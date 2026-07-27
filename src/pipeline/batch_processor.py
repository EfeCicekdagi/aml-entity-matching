import pandas as pd
import logging
import math
import time
import numpy as np

from src.config.db_tables import TABLES
from src.pipeline.inference_service import AMLInferenceService

logger = logging.getLogger(__name__)


def _compute_latency_percentiles(latencies_ms: list) -> dict:
    """P50, P95, P99 latency hesaplar (toplu işleme nedeniyle chunk ortalaması üzerinden tahmini satır latency)."""
    if not latencies_ms:
        return {"p50": None, "p95": None, "p99": None}
    arr = np.array(latencies_ms)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


class BatchProcessor:
    """AML batch işleme motoru. DB'den chunk halinde veri okur ve inference service'e iletir."""

    def __init__(self, repository, config: dict, inference_service: AMLInferenceService):
        self.repo = repository
        self.config = config
        self.inference_service = inference_service
        self.chunk_size = config.get("processing", {}).get("chunk_size", 2000)

    def process_db_table_in_chunks(
        self,
        run_id: str,
        batch_id: str,
        table_name: str = None,
        chunk_size: int = None
    ) -> None:
        """
        EFT verilerini chunk'lar halinde okuyup işler.
        """
        if table_name is None:
            table_name = TABLES["eft_input"]
            
        if chunk_size is None:
            chunk_size = self.chunk_size

        pipeline_start = time.time()

        metrics = {
            "input_row_count":        0,
            "input_count":            0,
            "processed_row_count":    0,
            "candidate_count":        0,  # Legacy alias for scored_candidate_count
            "scored_candidate_count": 0,
            "alert_count":            0,
            "high_alert_count":       0,  # Legacy alias for high_candidate_count
            "high_candidate_count":   0,
            "high_eft_count":         0,
            "medium_alert_count":     0,  # Legacy alias for medium_candidate_count
            "medium_candidate_count": 0,
            "medium_eft_count":       0,
            "no_match_count":         0,
            "no_candidate_count":     0,
            "match_result_count":     0,
            "error_count":            0,
            "ner_duration_s":         0.0,
            "embedding_duration_s":   0.0,
            "retrieval_duration_s":   0.0,
            "reranker_duration_s":    0.0,
            "scoring_duration_s":     0.0,
        }
        per_row_latencies = []

        embedding_model_name = self.config.get("embedding", {}).get("model_name", "UNKNOWN")
        reranker_model_name  = self.config.get("reranker",  {}).get("model_name", "UNKNOWN")

        self.repo.start_run_log(
            run_id,
            pipeline_name    = "AML_Production_Pipeline",
            embedding_model  = embedding_model_name,
            reranker_model   = reranker_model_name,
            scoring_config_version = self.config.get("scoring", {}).get("scoring_config_version"),
            threshold_version      = self.config.get("scoring", {}).get("threshold_config_version"),
            pipeline_version       = "aml_pipeline_v4",
            ner_model_name   = self.config.get("ner", {}).get("model_name"),
            watchlist_version= self.config.get("watchlist", {}).get("version"),
        )

        total_rows = None
        conn = self.repo.get_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    total_rows = cur.fetchone()[0]
            except Exception:
                pass
            finally:
                self.repo.release_connection(conn)

        total_chunks = math.ceil(total_rows / chunk_size) if total_rows else "?"

        conn_for_read = self.repo.get_connection()
        if not conn_for_read:
            logger.error("Could not get DB connection for reading.")
            self.repo.finish_run_log(run_id, metrics, duration_seconds=time.time() - pipeline_start, status="FAILED", error_message="Could not get DB connection for reading.")
            return

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                chunk_iter = pd.read_sql(
                    f"SELECT * FROM {table_name} ORDER BY eft_id",
                    con=conn_for_read,
                    chunksize=chunk_size
                )

                for chunk_idx, chunk in enumerate(chunk_iter):
                    pct = (
                        f"{100*(chunk_idx+1)/total_chunks:.1f}%"
                        if isinstance(total_chunks, int) else "?%"
                    )
                    logger.info(f"[Chunk {chunk_idx+1}/{total_chunks} | {pct}] Processing {len(chunk)} rows...")
                    
                    if len(chunk) == 0:
                        continue

                    metrics["input_row_count"] += len(chunk)
                    metrics["input_count"]     += len(chunk)

                    raw_explanations = chunk["explanation"].tolist()
                    row_ids = chunk.get("row_id", chunk["eft_id"]).astype(str).tolist()
                    eft_ids = chunk["eft_id"].tolist()

                    chunk_start = time.time()
                    try:
                        batch_results = self.inference_service.analyze_batch(
                            raw_explanations=raw_explanations,
                            row_ids=row_ids,
                            run_id=run_id,
                            eft_ids=eft_ids
                        )
                        chunk_duration_s = time.time() - chunk_start
                        chunk_duration_ms = chunk_duration_s * 1000
                        avg_row_latency_ms = chunk_duration_ms / len(chunk) if len(chunk) > 0 else 0.0
                        per_row_latencies.extend([avg_row_latency_ms] * len(chunk))

                        if batch_results and "metrics" in batch_results[0]:
                            c_metrics = batch_results[0]["metrics"]
                            ner_d = c_metrics.get("ner_duration_s", 0.0)
                            emb_d = c_metrics.get("embedding_duration_s", 0.0)
                            ret_d = c_metrics.get("retrieval_duration_s", 0.0)
                            rer_d = c_metrics.get("reranker_duration_s", 0.0)
                            metrics["ner_duration_s"] += ner_d
                            metrics["embedding_duration_s"] += emb_d
                            metrics["retrieval_duration_s"] += ret_d
                            metrics["reranker_duration_s"] += rer_d
                            metrics["scoring_duration_s"] += max(0.0, chunk_duration_s - (ner_d + emb_d + ret_d + rer_d))
                    except Exception as e:
                        logger.error(f"Failed to process chunk: {e}", exc_info=True)
                        metrics["error_count"] += len(chunk)
                        continue

                    chunk_match_results = []
                    chunk_alerts = []

                    for res in batch_results:
                        metrics["processed_row_count"] += 1
                        valid_cands = sum(1 for mr in res["match_results"] if mr.get("decision_status") != "NO_CANDIDATE_FOUND")
                        metrics["candidate_count"] += valid_cands
                        metrics["scored_candidate_count"] += valid_cands

                        if res["no_candidate"]:
                            metrics["no_candidate_count"] += 1
                            
                        metrics["high_alert_count"] += res["high_count"]
                        metrics["high_candidate_count"] += res["high_count"]
                        if res["high_count"] > 0:
                            metrics["high_eft_count"] += 1

                        metrics["medium_alert_count"] += res["medium_count"]
                        metrics["medium_candidate_count"] += res["medium_count"]
                        if res["medium_count"] > 0:
                            metrics["medium_eft_count"] += 1

                        metrics["no_match_count"] += res["no_match_count"]
                        metrics["alert_count"] += res["high_count"] + res["medium_count"]
                        
                        chunk_match_results.extend(res["match_results"])
                        chunk_alerts.extend(res["alerts"])

                    if chunk_match_results:
                        self.repo.insert_match_results_bulk(chunk_match_results)
                        metrics["match_result_count"] += len(chunk_match_results)
                        logger.info(f"  → {len(chunk_match_results)} match result(s) written.")

                    if chunk_alerts:
                        self.repo.insert_alerts_bulk(chunk_alerts)
                        logger.info(f"  → {len(chunk_alerts)} alert(s) written.")

        except Exception as exc:
            total_duration = time.time() - pipeline_start
            self.repo.finish_run_log(run_id, metrics, duration_seconds=total_duration, status="FAILED", error_message=str(exc))
            raise
        finally:
            self.repo.release_connection(conn_for_read)

        latency_stats = _compute_latency_percentiles(per_row_latencies)
        total_duration = time.time() - pipeline_start
        metrics["rows_per_second"] = (
            metrics["processed_row_count"] / total_duration if total_duration > 0 else 0.0
        )
        metrics["avg_candidate_per_row"] = (
            metrics["candidate_count"] / max(metrics["processed_row_count"], 1)
        )

        err_cnt = metrics.get("error_count", 0)
        inp_cnt = metrics.get("input_count", 0)
        if err_cnt == 0:
            final_status = "SUCCESS"
        elif 0 < err_cnt < inp_cnt:
            final_status = "PARTIAL_SUCCESS"
        else:
            final_status = "FAILED"
        self.repo.finish_run_log(run_id, metrics, duration_seconds=total_duration, status=final_status)
        self._update_latency_metrics(run_id, latency_stats)
        
        try:
            self.repo.populate_alert_export(run_id, table_name)
        except Exception as e:
            logger.error(f"Error populating export table: {e}")

        logger.info(f"Batch complete. Rows: {metrics['input_row_count']} | Scored Cands: {metrics['scored_candidate_count']} | HIGH Cands: {metrics['high_candidate_count']} (EFTs: {metrics['high_eft_count']}) | MEDIUM Cands: {metrics['medium_candidate_count']} (EFTs: {metrics['medium_eft_count']})")

    def _update_latency_metrics(self, run_id: str, latency_stats: dict) -> None:
        conn = self.repo.get_connection()
        if not conn: return
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {TABLES['run_log']}
                    SET p50_latency_ms = %s,
                        p95_latency_ms = %s,
                        p99_latency_ms = %s
                    WHERE run_id = %s
                """, (latency_stats.get("p50"), latency_stats.get("p95"), latency_stats.get("p99"), run_id))
            conn.commit()
        except Exception as e:
            logger.error(f"Error updating latency metrics: {e}")
            conn.rollback()
        finally:
            self.repo.release_connection(conn)
