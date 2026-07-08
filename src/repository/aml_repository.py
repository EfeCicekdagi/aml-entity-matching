import psycopg2
import logging
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

class AMLRepository:
    def __init__(self, host, port, dbname, user, password):
        self.conn_params = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password
        }

    def get_connection(self):
        try:
            return psycopg2.connect(**self.conn_params)
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return None

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
            conn.close()

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
            conn.close()

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
            conn.close()

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
            conn.close()

    # TODO: Add other insert functions (bronze, silver, gold, alert, etc.) later when needed.
