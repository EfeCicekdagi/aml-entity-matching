import sys
import os
import pandas as pd
import json
from datetime import datetime
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config.config_loader import ConfigLoader
from src.config.db_tables import TABLES
from src.repository.aml_repository import AMLRepository
from src.retrieval.postgres_candidate_retriever import PostgresCandidateRetriever
from src.models.reranker import Reranker
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

    def _current_cache_metadata(self) -> dict:
        """Mevcut konfigürasyondan cache metadata üretir."""
        emb_model_name = self.config_loader.get_embedding_config().get("model_name", "BAAI/bge-m3")
        rer_model_name = self.config_loader.get_reranker_config().get("model_name", "BAAI/bge-reranker-v2-m3")
        scoring_ver = self.config_loader.get_scoring_config().get("scoring_config_version", "scoring_v2_reranker")
        norm_ver = self.config_loader.config.get("entity_extraction", {}).get("version", "v1")
        return {
            "embedding_model": emb_model_name,
            "reranker_model": rer_model_name,
            "scoring_version": scoring_ver,
            "normalization_version": norm_ver,
        }

    def extract_raw_scores(self, df, cache_file="outputs/raw_scores.json"):
        current_meta = self._current_cache_metadata()
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cached = json.load(f)
            if isinstance(cached, list):
                logger.warning(f"Old cache format detected in {cache_file}. Using as-is without metadata validation.")
                return cached

            # Validate metadata — reject stale cache
            cached_meta = cached.get("__metadata__", {})
            mismatches = {k: (cached_meta.get(k), current_meta[k])
                         for k in current_meta if cached_meta.get(k) != current_meta[k]}
            if mismatches:
                logger.warning(
                    f"Cache metadata mismatch — ignoring stale cache. Differences: {mismatches}. "
                    f"Delete '{cache_file}' manually if you want to force refresh."
                )
            else:
                logger.info(f"Loading cached raw scores from {cache_file} (metadata OK)...")
                return cached.get("records", cached)  # backwards-compat

        logger.info("Extracting raw scores using MatchEngine (this will happen only once)...")
        raw_scores_list = []
        conn = self.repo.get_connection()
        cur = conn.cursor()

        # We need the EFT details. We can fetch them from bronze_eft_raw
        explanations = []
        row_ids = []
        should_match_list = []

        for idx, row in df.iterrows():
            eft_id = row['eft_id']
            should_match = row['should_match']

            cur.execute("SELECT explanation FROM bronze_eft_raw WHERE eft_id = %s", (eft_id,))
            res = cur.fetchone()
            if not res:
                continue

            explanations.append(res[0])
            row_ids.append(str(eft_id))
            should_match_list.append(should_match)

        # Initialize MatchEngine if not done
        if not hasattr(self, 'match_engine'):
            from src.pipeline.match_engine import MatchEngine
            from src.models.ner_extractor import NERExtractor
            from src.models.entity_extractor import EntityExtractor
            from sentence_transformers import SentenceTransformer

            ner_enabled = self.config_loader.config.get("ner", {}).get("enabled", False)
            entity_extractor = None
            if ner_enabled:
                ner_model = self.config_loader.config.get("ner", {}).get("model_name", "savasy/bert-base-turkish-ner-cased")
                ner_extractor = NERExtractor(model_name=ner_model, device=-1)
                entity_extractor = EntityExtractor(ner_extractor=ner_extractor)

            emb_model_name = self.config_loader.get_embedding_config().get("model_name", "BAAI/bge-m3")
            emb_model = SentenceTransformer(emb_model_name)

            self.match_engine = MatchEngine(
                config=self.config_loader.config,
                retriever=self.retriever,
                reranker=self.reranker,
                entity_extractor=entity_extractor,
                embedding_model=emb_model
            )

        # Process in batches
        logger.info(f"Processing {len(explanations)} explanations through MatchEngine...")
        engine_results = self.match_engine.process_batch(explanations, row_ids)

        from src.utils.text_utils import normalize_text, get_normalized_core_name

        for i, rid in enumerate(row_ids):
            eft_id = int(rid)
            should_match = should_match_list[i]
            res = engine_results.get(rid, {})
            candidates = res.get("candidates", [])
            clean_text = res.get("clean_text", "")

            best_cand = candidates[0] if candidates else None

            if best_cand:
                norm_query = normalize_text(clean_text)
                norm_cand = normalize_text(best_cand.get("variant_name", ""))
                core_query = get_normalized_core_name(norm_query)
                core_cand = get_normalized_core_name(norm_cand)
                exact_normalized_match = (norm_query == norm_cand and bool(norm_query))
                exact_core_match = (core_query == core_cand and bool(core_query))

                raw_scores = {
                    "fuzzy_score": best_cand.get("fuzzy_score", best_cand.get("trgm_score", 0.0)),
                    "vector_score": best_cand.get("vector_score", 0.0),
                    "reranker_score": best_cand.get("reranker_score", 0.0),
                    "acronym_score": best_cand.get("acronym_score", 0.0),
                    "rule_score": best_cand.get("rule_score", 0.0),
                    "exact_normalized_match": exact_normalized_match,
                    "exact_core_match": exact_core_match,
                    "legal_suffix_only_difference": exact_core_match and not exact_normalized_match,
                    "consonant_match": best_cand.get("consonant_match", False),
                    "query_token_count": len(norm_query.split()),
                    "candidate_company_id": best_cand.get("company_id") or best_cand.get("candidate_company_id"),
                    "alias_confidence": best_cand.get("alias_confidence", 1.0),
                    "_query_str": norm_query,
                    "_variant_str": norm_cand,
                }
                raw_scores_list.append({
                    "eft_id": eft_id,
                    "should_match": should_match,
                    "raw_scores": raw_scores,
                    "best_cand_name": best_cand.get("variant_name") or best_cand.get("company_name"),
                })
            else:
                raw_scores_list.append({
                    "eft_id": eft_id,
                    "should_match": should_match,
                    "raw_scores": {
                        "fuzzy_score": 0.0, "vector_score": 0.0, "reranker_score": 0.0,
                        "acronym_score": 0.0, "rule_score": 0.0,
                        "exact_normalized_match": False, "exact_core_match": False,
                        "legal_suffix_only_difference": False, "consonant_match": False,
                        "query_token_count": len(clean_text.split()),
                        "candidate_company_id": None, "alias_confidence": 1.0,
                        "_query_str": clean_text, "_variant_str": "",
                    },
                    "best_cand_name": None,
                })

            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1} / {len(row_ids)} records.")

        cur.close()
        self.repo.release_connection(conn)

        # Save with metadata so stale caches are detected
        cache_payload = {
            "__metadata__": {**current_meta, "created_at": datetime.utcnow().isoformat()},
            "records": raw_scores_list,
        }
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(cache_payload, f)
        logger.info(f"Saved raw scores to {cache_file} (with metadata)")

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
        cur.execute(f"INSERT INTO {TABLES['experiment_run']} (run_name, description) VALUES (%s, %s) RETURNING experiment_id",
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

        # ── SHORT_QUERY_PROFILE uyarısı ──────────────────────────────────────
        # FinalScorer, query_token_count <= 2 olan sorgular için farklı ağırlıklar kullanır:
        #   w_vector=0.10, w_reranker=0.40, w_fuzzy=0.30, w_rule=0.20
        # Bu grid search tüm kayıtlar için tek bir ağırlık kombiyasyonu arar.
        # Ancak kısa sorguların (tek kelime, kısaltmalar) bir bölümü bu ağırlıklara tabi DEĞİLDİR.
        # Dolayısıyla bulunan optimal ağırlıklar yalnızca DEFAULT_QUERY_PROFILE için geçerlidir.
        #
        # SHORT_QUERY_PROFILE'ı ayrı optimize etmek için bu dataset'i şu şekilde bölebilirsiniz:
        #   default_records = [r for r in raw_scores_list if r["raw_scores"].get("query_token_count", 3) > 2]
        #   short_records   = [r for r in raw_scores_list if r["raw_scores"].get("query_token_count", 3) <= 2]
        short_count = sum(
            1 for r in raw_scores_list
            if r["raw_scores"].get("query_token_count", 3) <= 2
        )
        if short_count:
            logger.warning(
                f"[SHORT_QUERY_PROFILE] Dataset'te {short_count} kısa sorgu (token_count <= 2) var. "
                f"FinalScorer bu sorgular için farklı sabit ağırlıklar kullanır. "
                f"Bulunan optimal ağırlıklar bu sorgular için GEÇERLİ DEĞİLDİR (DEFAULT_QUERY_PROFILE)."
            )


if __name__ == "__main__":
    import numpy as np
    opt = WeightOptimizer()
    df = opt.load_ground_truth("tests/aml_eft_challenge_ground_truth_1100 (1).csv")
    raw_scores = opt.extract_raw_scores(df)
    
    # Example ranges: 0.1 to 0.8 with 0.1 step
    r = [round(x, 1) for x in np.arange(0.1, 0.9, 0.1)]
    opt.grid_search(raw_scores, fuzzy_range=r, vector_range=r, reranker_range=r)
