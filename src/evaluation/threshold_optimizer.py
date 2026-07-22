import sys
import os
import json
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, auc, precision_recall_curve
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository
from src.scoring.final_scorer import FinalScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ThresholdOptimizer:
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

    def get_best_weights(self):
        logger.info("Fetching best weight configuration based on F1 Score...")
        conn = self.repo.get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, experiment_id, fuzzy_weight, vector_weight, reranker_weight 
            FROM aml_experiment.weight_analysis 
            ORDER BY f1_score DESC 
            LIMIT 1
        """)
        row = cur.fetchone()
        cur.close()
        self.repo.release_connection(conn)
        
        if not row:
            logger.error("No weight configurations found in aml_experiment.weight_analysis.")
            sys.exit(1)
            
        return {
            "id": row[0],
            "experiment_id": row[1],
            "fuzzy_weight": float(row[2]),
            "vector_weight": float(row[3]),
            "reranker_weight": float(row[4])
        }

    def grid_search_thresholds(self, raw_scores_list, best_weights):
        logger.info(f"Using weights: Fuzzy={best_weights['fuzzy_weight']}, Vector={best_weights['vector_weight']}, Reranker={best_weights['reranker_weight']}")
        
        self.scorer.weights = {
            "fuzzy_weight": best_weights['fuzzy_weight'],
            "vector_weight": best_weights['vector_weight'],
            "reranker_weight": best_weights['reranker_weight'],
            "acronym_weight": 0.0,
            "rule_weight": 0.0
        }
        
        # Precompute final scores for all samples with the best weights
        y_true = []
        final_scores_list = []
        
        for item in raw_scores_list:
            y_true.append(1 if item["should_match"] else 0)
            f_score, _, _ = self.scorer.calculate_final_score(item["raw_scores"], 1.0)
            final_scores_list.append(f_score)
            
        # Calculate Base AUC
        roc_auc = roc_auc_score(y_true, final_scores_list)
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, final_scores_list)
        pr_auc = auc(recall_arr, precision_arr)
        
        logger.info(f"Model Base Metrics - ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
        
        # Test Thresholds
        high_thresholds = [round(x, 2) for x in np.arange(0.70, 0.96, 0.05)]
        medium_thresholds = [round(x, 2) for x in np.arange(0.60, 0.81, 0.05)]
        
        conn = self.repo.get_connection()
        cur = conn.cursor()
        
        for high in high_thresholds:
            for medium in medium_thresholds:
                if medium >= high: continue
                
                y_pred = []
                tp = fp = tn = fn = 0
                for i, f_score in enumerate(final_scores_list):
                    is_alert = 1 if f_score >= medium else 0  # Any alert (High or Medium)
                    y_pred.append(is_alert)
                    
                    if y_true[i] == 1 and is_alert == 1: tp += 1
                    elif y_true[i] == 0 and is_alert == 1: fp += 1
                    elif y_true[i] == 0 and is_alert == 0: tn += 1
                    elif y_true[i] == 1 and is_alert == 0: fn += 1

                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                cur.execute("""
                    INSERT INTO aml_experiment.threshold_analysis 
                    (experiment_id, base_weight_id, high_threshold, medium_threshold, precision_score, recall_score, f1_score, roc_auc, pr_auc, tp, fp, tn, fn)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (best_weights["experiment_id"], best_weights["id"], high, medium, precision, recall, f1, roc_auc, pr_auc, tp, fp, tn, fn))
                
                logger.info(f"Threshold High={high}, Medium={medium} | Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                
        conn.commit()
        cur.close()
        self.repo.release_connection(conn)
        logger.info("Threshold optimization complete.")

if __name__ == "__main__":
    opt = ThresholdOptimizer()
    raw_scores = opt.load_cached_raw_scores()
    best_w = opt.get_best_weights()
    opt.grid_search_thresholds(raw_scores, best_w)
