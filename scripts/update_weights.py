import sys
import os

# Add src to python path for easier imports if running from root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository

def main():
    config_loader = ConfigLoader()
    db_config = config_loader.get_db_config()
    repo = AMLRepository(
        host=db_config.get("host"),
        port=db_config.get("port"),
        dbname=db_config.get("name"),
        user=db_config.get("user"),
        password=db_config.get("password")
    )
    conn = repo.get_connection()
    cur = conn.cursor()
    # ── 1. Update Scoring Weights ──
    cur.execute("""
        INSERT INTO aml_config.scoring_weight (
            config_version, fuzzy_weight, vector_weight, acronym_weight, rule_weight, reranker_weight, is_active
        )
        VALUES ('scoring_v2_reranker', 0.20, 0.60, 0.00, 0.00, 0.20, true)
        ON CONFLICT (config_version) DO UPDATE SET
            fuzzy_weight = EXCLUDED.fuzzy_weight,
            vector_weight = EXCLUDED.vector_weight,
            acronym_weight = EXCLUDED.acronym_weight,
            rule_weight = EXCLUDED.rule_weight,
            reranker_weight = EXCLUDED.reranker_weight,
            is_active = true;
    """)

    # ── 2. Update Thresholds ──
    for t_version in ['threshold_v2_reranker', 'threshold_v3_bge_m3']:
        cur.execute("""
            INSERT INTO aml_config.threshold (config_version, risk_level, min_score, max_score, is_active)
            VALUES 
                (%s, 'HIGH', 0.70, 1.00, true),
                (%s, 'MEDIUM', 0.60, 0.70, true),
                (%s, 'NO_MATCH', 0.00, 0.60, true)
            ON CONFLICT (config_version, risk_level) DO UPDATE SET
                min_score = EXCLUDED.min_score,
                max_score = EXCLUDED.max_score,
                is_active = true;
        """, (t_version, t_version, t_version))
    conn.commit()
    cur.close()
    repo.release_connection(conn)
    print("Database updated.")

if __name__ == "__main__":
    main()
