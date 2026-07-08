import psycopg2
import logging
from psycopg2.extras import execute_values
from psycopg2.pool import ThreadedConnectionPool

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

    def start_run_log(self, run_id, batch_id, pipeline_name="AML_Pipeline"):
        conn = self.get_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO aml_run_log (run_id, pipeline_name, batch_id, started_at, status)
                    VALUES (%s, %s, %s, now(), 'STARTED')
                """, (run_id, pipeline_name, batch_id))
            conn.commit()
        except Exception as e:
            logger.error(f"Error starting run log: {e}")
            conn.rollback()
        finally:
            self.release_connection(conn)

    def finish_run_log(self, run_id, metrics):
        conn = self.get_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE aml_run_log 
                    SET finished_at = now(),
                        input_row_count = %s,
                        processed_row_count = %s,
                        candidate_count = %s,
                        alert_count = %s,
                        status = 'SUCCESS'
                    WHERE run_id = %s
                """, (
                    metrics.get("input_row_count", 0),
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

    def fail_run_log(self, run_id, error_message):
        conn = self.get_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE aml_run_log 
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

    def insert_alert(self, run_id, eft_id, company_id, variant_id, final_score, risk_level):
        """Writes a detected AML alert to the aml_alert table."""
        conn = self.get_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO aml_alert
                        (run_id, eft_id, company_id, variant_id, final_score, risk_level, alert_status)
                    VALUES (%s, %s, %s, %s, %s, %s, 'OPEN')
                """, (run_id, eft_id, company_id, variant_id, final_score, risk_level))
            conn.commit()
        except Exception as e:
            logger.error(f"Error inserting alert: {e}")
            conn.rollback()
        finally:
            self.release_connection(conn)
