import logging
from src.config.db_tables import TABLES

logger = logging.getLogger(__name__)

class FinalScorer:
    def __init__(self, repository, config_version="scoring_v2_reranker", threshold_version="threshold_v2_reranker"):
        self.repo = repository
        self.config_version = config_version
        self.threshold_version = threshold_version
        
        self.weights = self._load_weights()
        self.thresholds = self._load_thresholds()

    def _load_weights(self):
        conn = self.repo.get_connection()
        weights = {
            "fuzzy_weight": 0.0,
            "vector_weight": 0.30,
            "acronym_weight": 0.0,
            "rule_weight": 0.0,
            "reranker_weight": 0.70
        }
        if not conn:
            return weights
            
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT fuzzy_weight, vector_weight, acronym_weight, rule_weight, reranker_weight
                    FROM {TABLES['scoring_weight']}
                    WHERE config_version = %s AND is_active = true
                    ORDER BY created_at DESC LIMIT 1
                """, (self.config_version,))
                res = cur.fetchone()
                if res:
                    weights = {
                        "fuzzy_weight": float(res[0]),
                        "vector_weight": float(res[1]),
                        "acronym_weight": float(res[2]),
                        "rule_weight": float(res[3]),
                        "reranker_weight": float(res[4])
                    }
        except Exception as e:
            logger.error(f"Error loading weights from DB: {e}")
        finally:
            self.repo.release_connection(conn)
        return weights

    def _load_thresholds(self):
        conn = self.repo.get_connection()
        thresholds = []
        if not conn:
            return thresholds
            
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT risk_level, min_score, max_score
                    FROM {TABLES['threshold']}
                    WHERE config_version = %s AND is_active = true
                """, (self.threshold_version,))
                for row in cur.fetchall():
                    thresholds.append({
                        "risk_level": row[0],
                        "min_score": float(row[1]),
                        "max_score": float(row[2])
                    })
        except Exception as e:
            logger.error(f"Error loading thresholds from DB: {e}")
        finally:
            self.repo.release_connection(conn)
        return thresholds

    def calculate_final_score(self, scores: dict):
        """
        scores dict must contain:
        fuzzy_score, vector_score, acronym_score, rule_score, reranker_score
        """
        final_score = (
            scores.get("fuzzy_score", 0.0) * self.weights.get("fuzzy_weight", 0.0) +
            scores.get("vector_score", 0.0) * self.weights.get("vector_weight", 0.0) +
            scores.get("acronym_score", 0.0) * self.weights.get("acronym_weight", 0.0) +
            scores.get("rule_score", 0.0) * self.weights.get("rule_weight", 0.0) +
            scores.get("reranker_score", 0.0) * self.weights.get("reranker_weight", 0.0)
        )
        return min(max(final_score, 0.0), 1.0)

    def assign_risk_level(self, final_score: float):
        for t in self.thresholds:
            if t["min_score"] <= final_score <= t["max_score"]:
                return t["risk_level"]
        return "NO_MATCH"
