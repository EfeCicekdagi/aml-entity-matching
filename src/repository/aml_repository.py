import psycopg2
import logging
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool
from src.config.db_tables import TABLES

logger = logging.getLogger(__name__)

class AMLRepository:
    def __init__(self, host, port, dbname, user, password):
        try:
            self.pool = ThreadedConnectionPool(
                minconn=1, maxconn=50,
                host=host, port=port, dbname=dbname, user=user, password=password
            )
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            self.pool = None

    def get_connection(self):
        if self.pool:
            try:
                return self.pool.getconn()
            except Exception as e:
                logger.error(f"Failed to get connection from pool: {e}")
        return None

    def release_connection(self, conn):
        if self.pool and conn:
            self.pool.putconn(conn)

    def execute_script(self, script_path):
        conn = self.get_connection()
        if not conn:
            return
        
        try:
            with open(script_path, "r", encoding="utf-8") as f:
                sql = f.read()
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            logger.info(f"Successfully executed script: {script_path}")
        except Exception as e:
            logger.error(f"Error executing script {script_path}: {e}")
            conn.rollback()
        finally:
            self.release_connection(conn)

    def start_run_log(self, run_id, pipeline_name="AML_Pipeline", embedding_model=None, reranker_model=None,
                      scoring_config_version=None, threshold_version=None, pipeline_version=None):
        conn = self.get_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {TABLES['run_log']} (run_id, pipeline_name, started_at, status, embedding_model, reranker_model,
                                                     scoring_config_version, threshold_version, pipeline_version, embedding_model_version, reranker_model_version)
                    VALUES (%s, %s, now(), 'STARTED', %s, %s, %s, %s, %s, %s, %s)
                """, (run_id, pipeline_name, embedding_model, reranker_model,
                      scoring_config_version, threshold_version, pipeline_version, embedding_model, reranker_model))
            conn.commit()
        except Exception as e:
            logger.error(f"Error starting run log: {e}")
            conn.rollback()
        finally:
            self.release_connection(conn)

    def finish_run_log(self, run_id, metrics, duration_seconds=None):
        conn = self.get_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {TABLES['run_log']} 
                    SET finished_at = now(),
                        processed_row_count = %s,
                        candidate_count = %s,
                        alert_count = %s,
                        status = 'SUCCESS'
                    WHERE run_id = %s
                """, (
                    metrics.get("processed_row_count", 0),
                    metrics.get("candidate_count", 0),
                    metrics.get("alert_count", 0),
                    run_id
                ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error finishing run log: {e}")
            conn.rollback()
        finally:
            self.release_connection(conn)

    def update_run_metrics(self, run_id, precision, recall, f1, exact_match):
        conn = self.get_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {TABLES['run_log']} 
                    SET precision_score = %s,
                        recall_score = %s,
                        f1_score = %s,
                        exact_match_score = %s
                    WHERE run_id = %s
                """, (precision, recall, f1, exact_match, run_id))
            conn.commit()
        except Exception as e:
            logger.error(f"Error updating run metrics: {e}")
            conn.rollback()
        finally:
            self.release_connection(conn)

    def fail_run_log(self, run_id, error_message):
        conn = self.get_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {TABLES['run_log']} 
                    SET finished_at = now(),
                        status = 'FAILED',
                        error_message = %s
                    WHERE run_id = %s
                """, (str(error_message), run_id))
            conn.commit()
        except Exception as e:
            logger.error(f"Error failing run log: {e}")
            conn.rollback()
        finally:
            self.release_connection(conn)

    def insert_alert(self, run_id, eft_id, company_id, variant_id, final_score, risk_level, extracted_entity=None):
        """Writes a single AML alert to the aml_alert table. Prefer insert_alerts_bulk for batch use."""
        self.insert_alerts_bulk([{
            "run_id": run_id, "eft_id": eft_id, "company_id": company_id,
            "variant_id": variant_id, "final_score": final_score, "risk_level": risk_level,
            "extracted_entity": extracted_entity
        }])

    def insert_alerts_bulk(self, alerts: list):
        """
        Bulk-inserts a list of alert dicts in a single transaction.
        Each dict must have: run_id, eft_id, company_id, variant_id, final_score, risk_level.
        Much faster than calling insert_alert() per row — avoids one commit per alert.
        """
        if not alerts:
            return
        conn = self.get_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                execute_values(cur, f"""
                    INSERT INTO {TABLES['alert']}
                        (run_id, eft_id, company_id, variant_id, final_score, fuzzy_score, vector_score, reranker_score, risk_level, alert_status, extracted_entity, match_reason,
                         entity_extraction_status, matched_variant_name, variant_type, watchlist_company_name)
                    VALUES %s
                    ON CONFLICT DO NOTHING
                """, [
                    (a["run_id"], a["eft_id"], a["company_id"],
                     a["variant_id"], a["final_score"], a.get("fuzzy_score", 0), a.get("vector_score", 0), a.get("reranker_score", 0),
                     a["risk_level"], "OPEN", a.get("extracted_entity"), a.get("match_reason"),
                     a.get("entity_extraction_status"), a.get("matched_variant_name"), a.get("variant_type"), a.get("watchlist_company_name"))
                    for a in alerts
                ])
            conn.commit()
            logger.debug(f"Bulk inserted {len(alerts)} alerts.")
        except Exception as e:
            logger.error(f"Error bulk inserting alerts: {e}")
            conn.rollback()
        finally:
            self.release_connection(conn)

    def update_alert_status(self, alert_id, status, reviewed_by=None, review_result=None, analyst_note=None, false_positive_reason=None):
        conn = self.get_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {TABLES['alert']}
                    SET alert_status = %s,
                        reviewed_by = %s,
                        reviewed_at = now(),
                        status_updated_at = now(),
                        review_result = %s,
                        analyst_note = %s,
                        false_positive_reason = %s
                    WHERE alert_id = %s
                """, (status, reviewed_by, review_result, analyst_note, false_positive_reason, alert_id))
            conn.commit()
            logger.info(f"Updated status for alert_id {alert_id} to {status}")
        except Exception as e:
            logger.error(f"Error updating alert status: {e}")
            conn.rollback()
        finally:
            self.release_connection(conn)
