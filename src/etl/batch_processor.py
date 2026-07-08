import pandas as pd
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, repository, config, retriever, reranker, scorer):
        self.repo = repository
        self.config = config
        self.retriever = retriever
        self.reranker = reranker
        self.scorer = scorer
        
        self.batch_size = self.config.get("embedding", {}).get("batch_size", 128)
        self.embedding_model_name = self.config.get("embedding", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model = None

    def _load_embedding_model(self):
        if not self.embedding_model:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded.")

    def process_file_in_chunks(self, file_path: str, run_id: str, batch_id: str, chunk_size: int = 10000):
        """Reads a CSV file in chunks and processes each chunk."""
        self._load_embedding_model()
        
        metrics = {
            "input_row_count": 0,
            "processed_row_count": 0,
            "candidate_count": 0,
            "alert_count": 0
        }
        
        self.repo.start_run_log(run_id, batch_id, pipeline_name="AML_Production_Pipeline")
        
        import concurrent.futures

        try:
            logger.info(f"Starting batch processing for {file_path}")
            chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)

            for chunk_idx, chunk in enumerate(chunk_iter):
                logger.info(f"Processing chunk {chunk_idx+1} ({len(chunk)} rows)...")
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

                # ── STEP 4: ONE batch query → replaces 30,000 individual queries
                all_candidates = self.retriever.batch_get_candidates(rows_for_batch)
                metrics["candidate_count"] += sum(len(v) for v in all_candidates.values())

                # ── STEP 5: Rerank only the strong survivors (parallel) ───────
                run_id_closure = run_id  # closure-safe reference

                def _rerank_and_score(row_id, candidates):
                    row_result = {"alert_count": 0}
                    # Pre-filter: pass candidates with score >= 0.70 to Reranker
                    strong = [c for c in candidates if c.get("candidate_score", 0.0) >= 0.70]
                    if not strong:
                        return row_result

                    norm_exp = next(
                        (r["normalized_explanation"] for r in rows_for_batch if r["row_id"] == row_id),
                        ""
                    )
                    strong = self.reranker.score_candidates(norm_exp, strong)

                    for cand in strong:
                        fuzzy_score  = cand["candidate_score"] if cand["source"] in ["pg_trgm", "combined"] else 0.0
                        vector_score = cand["candidate_score"] if cand["source"] in ["pgvector", "combined"] else 0.0
                        scores_dict = {
                            "fuzzy_score":    fuzzy_score,
                            "vector_score":   vector_score,
                            "acronym_score":  0.0,
                            "rule_score":     0.0,
                            "reranker_score": cand.get("reranker_score", 0.0),
                        }
                        final_score = self.scorer.calculate_final_score(scores_dict)
                        risk_level  = self.scorer.assign_risk_level(final_score)
                        if risk_level in ["HIGH", "MEDIUM"]:
                            row_result["alert_count"] += 1
                            # ✅ Write alert to database
                            self.repo.insert_alert(
                                run_id=run_id_closure,
                                eft_id=int(row_id),
                                company_id=cand["company_id"],
                                variant_id=cand["variant_id"],
                                final_score=final_score,
                                risk_level=risk_level,
                            )

                    return row_result

                with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                    future_map = {
                        executor.submit(_rerank_and_score, rid, cands): rid
                        for rid, cands in all_candidates.items()
                    }
                    for future in concurrent.futures.as_completed(future_map):
                        try:
                            res = future.result()
                            metrics["alert_count"]     += res["alert_count"]
                            metrics["processed_row_count"] += 1
                        except Exception as e:
                            logger.error(f"Row scoring failed: {e}")
                            metrics["processed_row_count"] += 1

            self.repo.finish_run_log(run_id, metrics)
            logger.info("Batch processing finished successfully.")

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            self.repo.fail_run_log(run_id, str(e))
