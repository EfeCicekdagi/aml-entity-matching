import pandas as pd
import logging
import math
import sys
import os
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# alias_utils & text_utils: fix import path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from alias_utils import generate_acronym
from text_utils import normalize_text, remove_company_suffixes

logger = logging.getLogger(__name__)


# Jenerik is kelimeleri — rule_score icin anlamsiz overlap yaratir
_RULE_STOPWORDS = {
    "services", "service", "group", "holding", "holdings",
    "international", "global", "enterprises", "enterprise",
    "solutions", "solution", "industries", "industry",
    "management", "investments", "investment",
    "trading", "trade", "export", "import",
    "logistics", "transport", "energy", "petroleum",
}


def _acronym_score(explanation: str, variant_name: str) -> float:
    """
    Checks whether the EFT explanation contains the acronym of the candidate
    company name. Returns 1.0 on match, 0.0 otherwise.
    Example: 'nst payment' vs. 'North Star Trading' -> acronym='nst' -> 1.0
    """
    acronym = generate_acronym(variant_name)
    if acronym and len(acronym) >= 2 and acronym in explanation.split():
        return 1.0
    return 0.0


def _rule_score(explanation: str, variant_name: str) -> float:
    """
    Token-overlap score between EFT explanation and candidate company name.
    Filters out:
      - Generic business words (_RULE_STOPWORDS)
      - Very short tokens (<= 3 chars) — avoids 'ltd', 'co', 'of' noise
    Returns fraction of *distinctive* variant tokens found in explanation.
    """
    clean_variant = remove_company_suffixes(normalize_text(variant_name))
    # Keep only long, distinctive tokens
    variant_tokens = {
        t for t in clean_variant.split()
        if len(t) > 3 and t not in _RULE_STOPWORDS
    }
    if not variant_tokens:
        return 0.0
    exp_tokens = set(explanation.split())
    overlap = variant_tokens & exp_tokens
    return len(overlap) / len(variant_tokens)


def _exact_name_score(explanation: str, variant_name: str) -> float:
    """
    Bonus score: returns 1.0 if the normalized variant name appears
    as a substring in the EFT explanation.
    Catches cases like 'TR TO Indiaforensic SERVICES IN' where the
    full company name is explicitly written in the description.
    """
    norm_variant = normalize_text(variant_name)
    # Need at least 2 meaningful words to avoid trivial matches
    tokens = [t for t in norm_variant.split() if len(t) > 3]
    if len(tokens) < 2:
        return 0.0
    # Check if all meaningful tokens appear in the explanation
    exp_tokens = set(explanation.split())
    if all(t in exp_tokens for t in tokens):
        return 1.0
    return 0.0


class BatchProcessor:
    def __init__(self, repository, config, retriever, reranker, scorer):
        self.repo = repository
        self.config = config
        self.retriever = retriever
        self.reranker = reranker
        self.scorer = scorer

        self.batch_size = self.config.get("embedding", {}).get("batch_size", 32)
        self.embedding_model_name = self.config.get("embedding", {}).get("model_name", "BAAI/bge-m3")
        self.embedding_model = None

        # Device selection: config > auto-detect (same pattern as Reranker)
        cfg_device = self.config.get("embedding", {}).get("device", "auto")
        if cfg_device == "auto" or cfg_device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg_device
        logger.info(f"Embedding model will use device: {self.device}")

        # Pre-filter threshold before sending to Reranker (configurable)
        self.reranker_prefilter_score = (
            self.config.get("retrieval", {}).get("reranker_prefilter_score", 0.60)
        )

    def _load_embedding_model(self):
        if not self.embedding_model:
            logger.info(f"Loading embedding model: {self.embedding_model_name} on {self.device}")
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device
            )
            logger.info(f"Embedding model loaded on {self.device}.")

    def process_file_in_chunks(self, file_path: str, run_id: str, batch_id: str, chunk_size: int = 10000):
        """Reads a CSV file in chunks and processes each chunk."""
        self._load_embedding_model()

        metrics = {
            "input_row_count":     0,
            "processed_row_count": 0,
            "candidate_count":     0,
            "alert_count":         0
        }

        self.repo.start_run_log(run_id, batch_id, pipeline_name="AML_Production_Pipeline")

        import concurrent.futures

        # ── Pre-compute total chunk count for progress logging ────────────────
        try:
            total_rows = sum(1 for _ in open(file_path, encoding="utf-8")) - 1  # -1 for header
        except Exception:
            total_rows = None
        total_chunks = math.ceil(total_rows / chunk_size) if total_rows else "?"

        try:
            logger.info(f"Starting batch processing for {file_path} "
                        f"(~{total_rows or '?'} rows, chunk_size={chunk_size})")
            chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)

            for chunk_idx, chunk in enumerate(chunk_iter):
                pct = f"{100*(chunk_idx+1)/total_chunks:.1f}%" if isinstance(total_chunks, int) else "?%"
                logger.info(f"[Chunk {chunk_idx+1}/{total_chunks} | {pct}] Processing {len(chunk)} rows...")
                metrics["input_row_count"] += len(chunk)

                # ── STEP 1: Normalize text ────────────────────────────────────
                chunk['normalized_explanation'] = chunk['description'].astype(str).str.lower()
                chunk_start_idx = chunk.index[0]

                # ── STEP 2: Batch embed the entire chunk in one GPU/CPU call ──
                explanations = chunk['normalized_explanation'].tolist()
                embeddings = self.embedding_model.encode(
                    explanations,
                    batch_size=self.batch_size,
                    show_progress_bar=False
                )

                # ── STEP 3: Build row list for batch DB query ─────────────────
                rows_for_batch = []
                for row_idx, row in chunk.iterrows():
                    rows_for_batch.append({
                        "row_id":                str(row_idx),
                        "normalized_explanation": row["normalized_explanation"],
                        "embedding":             embeddings[row_idx - chunk_start_idx].tolist(),
                    })

                # O(1) lookup dict: row_id → normalized_explanation
                row_lookup = {r["row_id"]: r["normalized_explanation"] for r in rows_for_batch}

                # ── STEP 4: ONE batch query → replaces 30,000 individual queries
                all_candidates = self.retriever.batch_get_candidates(rows_for_batch)
                metrics["candidate_count"] += sum(len(v) for v in all_candidates.values())

                # ── STEP 5: Rerank only the strong survivors (parallel) ───────
                run_id_closure = run_id  # closure-safe reference
                chunk_alerts   = []      # collect alerts for bulk insert

                def _rerank_and_score(row_id, candidates):
                    row_result = {"alert_count": 0, "alerts": []}
                    norm_exp = row_lookup.get(row_id, "")

                    # Pre-filter: pass candidates that EITHER
                    #   (a) have a high retrieval score, OR
                    #   (b) have the company name explicitly in the EFT text
                    #       (exact_name_score=1.0) — catches fuzzy-miss but name-match cases
                    strong = [
                        c for c in candidates
                        if c.get("candidate_score", 0.0) >= self.reranker_prefilter_score
                        or _exact_name_score(norm_exp, c["variant_name"]) == 1.0
                    ]
                    if not strong:
                        return row_result

                    strong = self.reranker.score_candidates(norm_exp, strong)

                    for cand in strong:
                        fuzzy_score  = cand["candidate_score"] if cand["source"] in ["pg_trgm", "combined"] else 0.0
                        vector_score = cand["candidate_score"] if cand["source"] in ["pgvector", "combined"] else 0.0
                        scores_dict = {
                            "fuzzy_score":    fuzzy_score,
                            "vector_score":   vector_score,
                            "acronym_score":  _acronym_score(norm_exp, cand["variant_name"]),
                            # rule_score: filtered token overlap (no generic words)
                            # exact_name_score folded into rule_score as max
                            "rule_score":     max(
                                _rule_score(norm_exp, cand["variant_name"]),
                                _exact_name_score(norm_exp, cand["variant_name"])
                            ),
                            "reranker_score": cand.get("reranker_score", 0.0),
                        }
                        final_score = self.scorer.calculate_final_score(scores_dict)
                        risk_level  = self.scorer.assign_risk_level(final_score)
                        if risk_level in ["HIGH", "MEDIUM"]:
                            row_result["alert_count"] += 1
                            row_result["alerts"].append({
                                "run_id":      run_id_closure,
                                "eft_id":      int(row_id),
                                "company_id":  cand["company_id"],
                                "variant_id":  cand["variant_id"],
                                "final_score": final_score,
                                "risk_level":  risk_level,
                            })

                    return row_result

                with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                    future_map = {
                        executor.submit(_rerank_and_score, rid, cands): rid
                        for rid, cands in all_candidates.items()
                    }
                    for future in concurrent.futures.as_completed(future_map):
                        try:
                            res = future.result()
                            metrics["alert_count"]         += res["alert_count"]
                            metrics["processed_row_count"] += 1
                            chunk_alerts.extend(res["alerts"])
                        except Exception as e:
                            logger.error(f"Row scoring failed: {e}")
                            metrics["processed_row_count"] += 1

                # ── STEP 6: Bulk insert all alerts for this chunk ─────────────
                if chunk_alerts:
                    self.repo.insert_alerts_bulk(chunk_alerts)
                    logger.info(f"  → {len(chunk_alerts)} alert(s) written to DB.")

            self.repo.finish_run_log(run_id, metrics)
            logger.info(
                f"Batch processing finished. "
                f"Rows: {metrics['input_row_count']} | "
                f"Candidates: {metrics['candidate_count']} | "
                f"Alerts: {metrics['alert_count']}"
            )

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            self.repo.fail_run_log(run_id, str(e))

