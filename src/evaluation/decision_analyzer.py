import sys
import os
import json
import logging
from collections import defaultdict
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository
from src.scoring.final_scorer import FinalScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DecisionAnalyzer:
    def __init__(self):
        self.config_loader = ConfigLoader()
        db_cfg = self.config_loader.get_db_config()
        self.repo = AMLRepository(
            host=db_cfg.get("host"), port=db_cfg.get("port"),
            dbname=db_cfg.get("name"), user=db_cfg.get("user"), password=db_cfg.get("password")
        )
        self.scorer = FinalScorer(self.repo)
        
    def load_cached_raw_scores(self, cache_file="outputs/raw_scores.json"):
        if not os.path.exists(cache_file):
            logger.error(f"Cache file {cache_file} not found. Run WeightOptimizer first.")
            sys.exit(1)
        with open(cache_file, "r") as f:
            return json.load(f)

    def analyze(self, raw_scores_list):
        logger.info("Fetching best weight and threshold configurations...")
        conn = self.repo.get_connection()
        cur = conn.cursor()
        
        # Get best thresholds based on F1
        cur.execute("""
            SELECT w.experiment_id, w.fuzzy_weight, w.vector_weight, w.reranker_weight, t.high_threshold, t.medium_threshold 
            FROM aml_experiment.threshold_analysis t
            JOIN aml_experiment.weight_analysis w ON t.base_weight_id = w.id
            ORDER BY t.f1_score DESC 
            LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            # Fallback to weights only
            cur.execute("SELECT experiment_id, fuzzy_weight, vector_weight, reranker_weight FROM aml_experiment.weight_analysis ORDER BY f1_score DESC LIMIT 1")
            w_row = cur.fetchone()
            if not w_row:
                logger.error("No experiment data found.")
                sys.exit(1)
            row = (w_row[0], w_row[1], w_row[2], w_row[3], 0.70, 0.60) # Default thresholds
        
        exp_id, f_w, v_w, r_w, high_th, med_th = row
        logger.info(f"Using: F={f_w}, V={v_w}, R={r_w} | High={high_th}, Med={med_th}")
        
        self.scorer.weights = {
            "fuzzy_weight": float(f_w),
            "vector_weight": float(v_w),
            "reranker_weight": float(r_w),
            "acronym_weight": 0.0,
            "rule_weight": 0.0
        }
        
        analysis_records = []
        score_comparison_stats = defaultdict(lambda: {"count": 0, "tp": 0, "fp": 0, "fn": 0})
        
        for item in raw_scores_list:
            expected = item["should_match"]
            raw = item["raw_scores"]
            
            final_score, reason, codes = self.scorer.calculate_final_score(raw, 1.0)
            
            if final_score >= high_th:
                predicted_label = "HIGH_ALERT"
                is_alert = True
            elif final_score >= med_th:
                predicted_label = "MEDIUM_ALERT"
                is_alert = True
            else:
                predicted_label = "NO_MATCH"
                is_alert = False
                
            is_correct = (expected == is_alert)
            error_type = None
            if expected and not is_alert: error_type = "FN"
            elif not expected and is_alert: error_type = "FP"
            
            # Score Comparison Categories
            f_score = raw.get("fuzzy_score", 0.0)
            v_score = raw.get("vector_score", 0.0)
            r_score = raw.get("reranker_score", 0.0)
            
            cat = "OTHER"
            if f_score >= 0.8 and v_score < 0.5:
                cat = "Fuzzy High, Vector Low"
            elif f_score < 0.5 and v_score >= 0.8:
                cat = "Fuzzy Low, Vector High"
            elif f_score >= 0.8 and v_score >= 0.8 and r_score < 0.6:
                cat = "Both High, Reranker Low"
            elif f_score >= 0.7 and v_score >= 0.7 and r_score >= 0.7:
                cat = "All High"
                
            score_comparison_stats[cat]["count"] += 1
            if expected and is_alert: score_comparison_stats[cat]["tp"] += 1
            if not expected and is_alert: score_comparison_stats[cat]["fp"] += 1
            if expected and not is_alert: score_comparison_stats[cat]["fn"] += 1
            
            # Save all FP and FN to DB
            if not is_correct:
                analysis_records.append((
                    exp_id, item["eft_id"], "Unknown", item["best_cand_name"],
                    f_score, v_score, r_score, final_score,
                    predicted_label, str(expected), is_correct, error_type, reason
                ))
        
        # Save to DB
        cur.execute("DELETE FROM aml_experiment.decision_analysis WHERE experiment_id = %s", (exp_id,))
        from psycopg2.extras import execute_values
        execute_values(cur, """
            INSERT INTO aml_experiment.decision_analysis 
            (experiment_id, eft_id, expected_company, retrieved_company, fuzzy_score, vector_score, reranker_score, final_score, predicted_label, expected_label, is_correct, error_type, reason_code)
            VALUES %s
        """, analysis_records)
        conn.commit()
        
        cur.close()
        self.repo.release_connection(conn)
        
        logger.info(f"Saved {len(analysis_records)} error cases to decision_analysis.")
        
        # Print report
        print("\n=== SCORE COMPARISON REPORT ===")
        for cat, stats in score_comparison_stats.items():
            print(f"\nCategory: {cat}")
            print(f"Total Count: {stats['count']}")
            print(f"True Positives: {stats['tp']}")
            print(f"False Positives: {stats['fp']}")
            print(f"False Negatives: {stats['fn']}")

if __name__ == "__main__":
    analyzer = DecisionAnalyzer()
    raw = analyzer.load_cached_raw_scores()
    analyzer.analyze(raw)
