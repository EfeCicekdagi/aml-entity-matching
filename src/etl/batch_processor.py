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
from src.utils.alias_utils import generate_acronym
from src.utils.text_utils import normalize_text, remove_company_suffixes
from src.utils.ner_extractor import NERExtractor
from src.config.db_tables import TABLES

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

        # EFT Pre-screening threshold (trigram min score to be considered suspicious)
        self.min_trgm_score = (
            self.config.get("retrieval", {}).get("min_trgm_score", 0.25)
        )

        # NER Initialization
        self.ner_enabled = self.config.get("ner", {}).get("enabled", False)
        self.ner_extractor = None
        if self.ner_enabled:
            ner_model = self.config.get("ner", {}).get("model_name", "savasy/bert-base-turkish-ner-cased")
            ner_dev = self.config.get("ner", {}).get("device", "auto")
            dev_id = 0 if (ner_dev == "auto" and torch.cuda.is_available()) or ner_dev == "cuda" else -1
            self.ner_extractor = NERExtractor(model_name=ner_model, device=dev_id)

    def _load_embedding_model(self):
        if not self.embedding_model:
            logger.info(f"Loading embedding model: {self.embedding_model_name} on {self.device}")
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=self.device
            )
            logger.info(f"Embedding model loaded on {self.device}.")

    def _prescreen_eft_chunk(self, norm_explanations: list, row_ids: list) -> set:
        """
        EFT PRE-SCREENING (Ön Eleme):
        Embedding'den ÖNCE tek bir hızlı SQL trigram sorgusu çalıştırır.

        Yöntem: Her EFT için şirket listesindeki en iyi eşleşmenin MAX trigram
        skoru, min_trgm_score eşiğinin üzerindeyse EFT şüpheli sayılır.
        Böylece çok zayıf benzerlikler (örn. tek bir harf örtüşmesi) olan
        EFT'ler elenerek NER + Embedding + Reranker yükü azaltılır.

        Returns:
            Set of row_ids whose BEST company match score exceeds min_trgm_score.
        """
        if not norm_explanations:
            return set()

        conn = self.repo.get_connection()
        if not conn:
            return set(row_ids)  # Fail-safe: process all if DB unavailable

        suspicious_ids = set()
        try:
            with conn.cursor() as cur:
                # MAX score bazlı pre-screen:
                # Her EFT için şirket listesindeki en yüksek trigram skoru bulunur.
                # Sadece MAX skor >= min_trgm_score olan EFT'ler geçer.
                prescreen_query = f"""
                    SELECT input.row_id
                    FROM (
                        SELECT
                            UNNEST(%s::text[]) AS norm_exp,
                            UNNEST(%s::text[]) AS row_id
                    ) AS input
                    JOIN LATERAL (
                        SELECT similarity(input.norm_exp, v.normalized_variant_name) AS sim
                        FROM {TABLES['company_variant']} v
                        WHERE v.is_active = true
                        ORDER BY input.norm_exp <-> v.normalized_variant_name
                        LIMIT 1
                    ) AS nearest ON nearest.sim >= %s
                """
                cur.execute(prescreen_query, (
                    norm_explanations,
                    [str(rid) for rid in row_ids],
                    self.min_trgm_score
                ))
                for row in cur.fetchall():
                    suspicious_ids.add(row[0])
        except Exception as e:
            logger.warning(f"Pre-screening failed (will process all rows): {e}")
            return set(str(rid) for rid in row_ids)  # Fail-safe
        finally:
            self.repo.release_connection(conn)

        return suspicious_ids

    def process_db_table_in_chunks(self, run_id: str, batch_id: str, table_name: str = None, chunk_size: int = 2000):
        """Reads EFT records from PostgreSQL in chunks and processes each chunk."""
        if table_name is None:
            table_name = TABLES["eft_input"]
            
        self._load_embedding_model()

        import time
        start_time = time.time()

        metrics = {
            "input_row_count":     0,
            "processed_row_count": 0,
            "candidate_count":     0,
            "alert_count":         0,
            "prescreen_skipped_count": 0
        }

        embedding_model_name = self.config.get("embedding", {}).get("model_name", "UNKNOWN")
        reranker_model_name = self.config.get("reranker", {}).get("model_name", "UNKNOWN")

        self.repo.start_run_log(
            run_id, 
            pipeline_name="AML_Production_Pipeline",
            embedding_model=embedding_model_name,
            reranker_model=reranker_model_name
        )

        import concurrent.futures

        # ── Pre-compute total chunk count for progress logging ────────────────
        try:
            conn = self.repo.get_connection()
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                total_rows = cur.fetchone()[0]
        except Exception as e:
            logger.warning(f"Could not get total rows from {table_name}: {e}")
            total_rows = None
        finally:
            if 'conn' in locals() and conn:
                self.repo.release_connection(conn)

        total_chunks = math.ceil(total_rows / chunk_size) if total_rows else "?"

        try:
            logger.info(f"Starting batch processing for {table_name} "
                        f"(~{total_rows or '?'} rows, chunk_size={chunk_size})")
            
            conn_for_read = self.repo.get_connection()
            try:
                # We suppress pandas warnings about SQLAlchemy
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    chunk_iter = pd.read_sql(f"SELECT * FROM {table_name} ORDER BY eft_id", con=conn_for_read, chunksize=chunk_size)

                    for chunk_idx, chunk in enumerate(chunk_iter):
                        pct = f"{100*(chunk_idx+1)/total_chunks:.1f}%" if isinstance(total_chunks, int) else "?%"
                        logger.info(f"[Chunk {chunk_idx+1}/{total_chunks} | {pct}] Processing {len(chunk)} rows...")
                        
                        if len(chunk) == 0:
                            logger.info("Chunk is empty, skipping.")
                            continue

                        metrics["input_row_count"] += len(chunk)

                        # ── STEP 1: Normalize text ────────────────────────────────────
                        chunk['normalized_explanation'] = chunk['explanation'].astype(str).str.lower()
                        explanations = chunk['normalized_explanation'].tolist()
                        search_queries = explanations

                        # ── STEP 1.5: EFT PRE-SCREENING (Ön Eleme) ──────────────────
                        # NER'den ONCE çalıştır: Sadece şüpheli EFT'ler
                        # sonraki ağır işlemlere (NER + Embedding + Reranker) taşınır.
                        raw_row_ids = [
                            str(row['eft_id']) if 'eft_id' in row else str(idx)
                            for idx, (_, row) in enumerate(chunk.iterrows())
                        ]
                        suspicious_ids = self._prescreen_eft_chunk(search_queries, raw_row_ids)

                        skipped = len(search_queries) - len(suspicious_ids)
                        metrics["prescreen_skipped_count"] += skipped

                        logger.info(
                            f"  [Pre-Screen] {len(suspicious_ids)}/{len(search_queries)} EFT şüpheli "
                            f"({skipped} EFT elendi — NER+Embedding atlandı 🚀)"
                        )

                        if not suspicious_ids:
                            logger.info("  [Pre-Screen] Bu chunk'ta şüpheli EFT yok, atlandı.")
                            metrics["processed_row_count"] += len(chunk)
                            continue

                        # Filtrele: sadece şüpheli EFT'lerin query'lerini al
                        filtered_queries     = []
                        filtered_raw_row_ids = []
                        for i, rid in enumerate(raw_row_ids):
                            if rid in suspicious_ids:
                                filtered_queries.append(search_queries[i])
                                filtered_raw_row_ids.append(rid)

                        # ── STEP 1.9: NER ──────────────────────────────────
                        # Sadece pre-screen'den geçen şüpheli EFT'lere uygulanır!
                        filtered_entities = [None] * len(filtered_queries)
                        if self.ner_enabled and self.ner_extractor:
                            filtered_entities = self.ner_extractor.batch_extract_entities(filtered_queries)

                        # ── STEP 2: Batch embed ONLY suspicious EFTs ──────────────────
                        embeddings = self.embedding_model.encode(
                            filtered_queries,
                            batch_size=self.batch_size,
                            show_progress_bar=False
                        )

                        # ── STEP 3: Build row list for batch DB query ─────────────────
                        rows_for_batch = []
                        for list_idx, rid in enumerate(filtered_raw_row_ids):
                            rows_for_batch.append({
                                "row_id":                rid,
                                "normalized_explanation": filtered_queries[list_idx],
                                "embedding":             embeddings[list_idx].tolist(),
                                "extracted_entity":      filtered_entities[list_idx]
                            })

                        # O(1) lookup dict: row_id → normalized_explanation
                        row_lookup = {r["row_id"]: r["normalized_explanation"] for r in rows_for_batch}
                        entity_lookup = {r["row_id"]: r["extracted_entity"] for r in rows_for_batch}

                        # ── STEP 4: ONE batch query → replaces 30,000 individual queries
                        all_candidates = self.retriever.batch_get_candidates(rows_for_batch)
                        
                        trgm_count, vector_count, combined_count, total_count = 0, 0, 0, sum(len(v) for v in all_candidates.values())
                        for cands in all_candidates.values():
                            for c in cands:
                                srcs = c.get("sources", [])
                                if "pg_trgm" in srcs and "pgvector" in srcs:
                                    combined_count += 1
                                elif "pg_trgm" in srcs:
                                    trgm_count += 1
                                elif "pgvector" in srcs:
                                    vector_count += 1
                                    
                        logger.info(f"Retrieved Candidates - Total Merged: {total_count}, TRGM-only: {trgm_count}, Vector-only: {vector_count}, Both: {combined_count}")
                        metrics["candidate_count"] += total_count

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

                            extracted = entity_lookup.get(row_id, None)

                            for cand in strong:
                                fuzzy_score  = cand.get("trgm_score", 0.0)
                                vector_score = cand.get("vector_score", 0.0)
                                raw_reranker = cand.get("raw_reranker_score", 0.0)
                                norm_reranker = cand.get("normalized_reranker_score", 0.0)
                                
                                logger.debug(f"Candidate: {cand['company_name']} - Sources: {cand.get('sources', [])}, TRGM: {fuzzy_score:.4f}, Vector: {vector_score:.4f}, RawReranker: {raw_reranker:.4f}, NormReranker: {norm_reranker:.4f}")

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
                                        "fuzzy_score": scores_dict["fuzzy_score"],
                                        "vector_score": scores_dict["vector_score"],
                                        "reranker_score": scores_dict["reranker_score"],
                                        "risk_level":  risk_level,
                                        "extracted_entity": extracted
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

                duration = time.time() - start_time
                self.repo.finish_run_log(run_id, metrics, duration_seconds=duration)
                logger.info(
                    f"Batch processing finished. "
                    f"Rows: {metrics['input_row_count']} | "
                    f"Candidates: {metrics['candidate_count']} | "
                    f"Alerts: {metrics['alert_count']}"
                )

            finally:
                if 'conn_for_read' in locals() and conn_for_read:
                    self.repo.release_connection(conn_for_read)

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            self.repo.fail_run_log(run_id, str(e))

