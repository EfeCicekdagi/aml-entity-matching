import sys
import os
import pandas as pd
import itertools
import json
from datetime import datetime
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository
from src.retrieval.postgres_candidate_retriever import PostgresCandidateRetriever
from src.reranker.reranker import Reranker
from src.scoring.final_scorer import FinalScorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class WeightOptimizer:
    def __init__(self):
        self.config_loader = ConfigLoader()
        db_cfg = self.config_loader.get_db_config()
        self.repo = AMLRepository(
            host=db_cfg.get("host"), port=db_cfg.get("port"),
            dbname=db_cfg.get("name"), user=db_cfg.get("user"), password=db_cfg.get("password")
        )
        
        self.retriever = PostgresCandidateRetriever(self.repo, self.config_loader.get_retrieval_config())
        self.reranker = Reranker(self.repo, self.config_loader.get_reranker_config())
        self.scorer = FinalScorer(self.repo)
        
    def load_ground_truth(self, csv_path):
        df = pd.read_csv(csv_path)
        # Assuming columns: eft_id, expected_company, should_match
        return df

    def extract_raw_scores(self, df, cache_file="outputs/raw_scores.json"):
        if os.path.exists(cache_file):
            logger.info(f"Loading cached raw scores from {cache_file}...")
            with open(cache_file, "r") as f:
                return json.load(f)
                
        logger.info("Extracting raw scores using Retriever and Reranker (this will happen only once)...")
        raw_scores_list = []
        conn = self.repo.get_connection()
        cur = conn.cursor()
        
        # We need the EFT details. We can fetch them from bronze_eft_raw
        for idx, row in df.iterrows():
            eft_id = row['eft_id']
            should_match = row['should_match']
            
            cur.execute("SELECT explanation FROM bronze_eft_raw WHERE eft_id = %s", (eft_id,))
            res = cur.fetchone()
            if not res:
                continue
            
            explanation = res[0]
            
            # Retrieve candidates
            candidates = self.retriever.get_merged_candidates(explanation)
            
            # We only care if we found candidates
            best_cand = None
            if candidates:
                # Rerank
                reranked = self.reranker.score_candidates(explanation, candidates)
                if reranked:
                    best_cand = reranked[0]
            
            if best_cand:
                # Prepare score dict
                raw_scores = {
                    "fuzzy_score": best_cand.get("fuzzy_score", best_cand.get("trgm_score", 0.0)),
                    "vector_score": best_cand.get("vector_score", 0.0),
                    "reranker_score": best_cand.get("reranker_score", 0.0)
                }
            else:
                raw_scores = {"fuzzy_score": 0.0, "vector_score": 0.0, "reranker_score": 0.0}
            
            raw_scores_list.append({
                "eft_id": eft_id,
                "should_match": should_match,
                "raw_scores": raw_scores,
                "best_cand_name": best_cand["variant_name"] if best_cand else None
            })
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1} / {len(df)} records.")
                
        cur.close()
        self.repo.release_connection(conn)
        
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(raw_scores_list, f)
        logger.info(f"Saved raw scores to {cache_file}")
        
        return raw_scores_list

    def grid_search(self, raw_scores_list, fuzzy_range, vector_range, reranker_range):
        logger.info("Starting Grid Search for Weights...")
        
        # Generate combinations summing to 1.0
        combinations = []
        for f in fuzzy_range:
            for v in vector_range:
                for r in reranker_range:
                    if round(f + v + r, 2) == 1.00:
                        combinations.append({"fuzzy_weight": f, "vector_weight": v, "reranker_weight": r})
        
        logger.info(f"Generated {len(combinations)} valid weight combinations.")
        
        conn = self.repo.get_connection()
        cur = conn.cursor()
        
        # Create experiment run
        run_name = f"Weight-Opt-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        cur.execute("INSERT INTO aml_experiment.experiment_run (run_name, description) VALUES (%s, %s) RETURNING experiment_id",
                    (run_name, "Grid Search for weights"))
        experiment_id = cur.fetchone()[0]
        
        results = []
        for combo in combinations:
            self.scorer.weights = {
                "fuzzy_weight": combo["fuzzy_weight"],
                "vector_weight": combo["vector_weight"],
                "reranker_weight": combo["reranker_weight"],
                "acronym_weight": 0.0,
                "rule_weight": 0.0
            }
            
            y_true = []
            y_pred = []
            
            tp = fp = tn = fn = 0
            
            for item in raw_scores_list:
                y_true.append(1 if item["should_match"] else 0)
                
                # compute final score
                final_score, reason, codes = self.scorer.calculate_final_score(item["raw_scores"], 1.0)
                
                # simulate threshold assignment (assume High threshold is 0.70 for this phase)
                is_alert = 1 if final_score >= 0.70 else 0
                y_pred.append(is_alert)
                
                if item["should_match"] and is_alert: tp += 1
                elif not item["should_match"] and is_alert: fp += 1
                elif not item["should_match"] and not is_alert: tn += 1
                elif item["should_match"] and not is_alert: fn += 1
                
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            
            # Save to db
            cur.execute("""
                INSERT INTO aml_experiment.weight_analysis 
                (experiment_id, fuzzy_weight, vector_weight, reranker_weight, high_threshold, medium_threshold, precision_score, recall_score, f1_score, accuracy, tp, fp, tn, fn)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (experiment_id, combo["fuzzy_weight"], combo["vector_weight"], combo["reranker_weight"], 0.70, 0.60, precision, recall, f1, acc, tp, fp, tn, fn))
            
            logger.info(f"Combo F={combo['fuzzy_weight']:.2f}, V={combo['vector_weight']:.2f}, R={combo['reranker_weight']:.2f} | F1={f1:.4f}")
            
        conn.commit()
        cur.close()
        self.repo.release_connection(conn)
        logger.info(f"Grid search completed. Experiment ID: {experiment_id}")

if __name__ == "__main__":
    import numpy as np
    opt = WeightOptimizer()
    df = opt.load_ground_truth("tests/aml_eft_challenge_ground_truth_1100 (1).csv")
    raw_scores = opt.extract_raw_scores(df)
    
    # Example ranges: 0.1 to 0.8 with 0.1 step
    r = [round(x, 1) for x in np.arange(0.1, 0.9, 0.1)]
    opt.grid_search(raw_scores, fuzzy_range=r, vector_range=r, reranker_range=r)
