import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository

logging.basicConfig(level=logging.INFO)

def migrate_db():
    config_loader = ConfigLoader()
    cfg = config_loader.get_db_config()
    repo = AMLRepository(host=cfg['host'], port=cfg['port'], dbname=cfg['name'], user=cfg['user'], password=cfg['password'])
    
    conn = repo.get_connection()
    if not conn:
        logging.error("Failed to connect to db")
        return
        
    try:
        with conn.cursor() as cur:
            logging.info("Altering aml_ml.reranker_cache...")
            cur.execute("""
                ALTER TABLE aml_ml.reranker_cache 
                ADD COLUMN IF NOT EXISTS raw_reranker_score NUMERIC(8,5),
                ADD COLUMN IF NOT EXISTS normalized_reranker_score NUMERIC(8,5);
            """)
            
            logging.info("Altering aml_core.scoring_result...")
            cur.execute("""
                ALTER TABLE aml_core.scoring_result 
                ADD COLUMN IF NOT EXISTS raw_reranker_score NUMERIC(8,5),
                ADD COLUMN IF NOT EXISTS normalized_reranker_score NUMERIC(8,5);
            """)

            logging.info("Inserting new scoring_weight...")
            cur.execute("""
                INSERT INTO aml_config.scoring_weight (config_version, fuzzy_weight, vector_weight, acronym_weight, rule_weight, reranker_weight)
                VALUES ('scoring_v2_reranker', 0.20, 0.25, 0.00, 0.05, 0.50)
                ON CONFLICT (config_version) DO UPDATE SET
                    fuzzy_weight = EXCLUDED.fuzzy_weight,
                    vector_weight = EXCLUDED.vector_weight,
                    acronym_weight = EXCLUDED.acronym_weight,
                    rule_weight = EXCLUDED.rule_weight,
                    reranker_weight = EXCLUDED.reranker_weight,
                    is_active = true;
            """)

            logging.info("Inserting new thresholds...")
            thresholds = [
                ('threshold_v3_bge_m3', 'NO_MATCH', 0.00, 0.50),
                ('threshold_v3_bge_m3', 'LOW', 0.50, 0.65),
                ('threshold_v3_bge_m3', 'MEDIUM', 0.65, 0.80),
                ('threshold_v3_bge_m3', 'HIGH', 0.80, 1.00)
            ]
            for t in thresholds:
                cur.execute("""
                    INSERT INTO aml_config.threshold (config_version, risk_level, min_score, max_score)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (config_version, risk_level) DO UPDATE SET
                        min_score = EXCLUDED.min_score,
                        max_score = EXCLUDED.max_score,
                        is_active = true;
                """, t)
            
        conn.commit()
        logging.info("Migration successful.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Migration failed: {e}")
    finally:
        repo.release_connection(conn)

if __name__ == "__main__":
    migrate_db()
