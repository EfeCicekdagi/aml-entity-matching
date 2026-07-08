import logging
import hashlib
import threading
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, repository, config):
        self.repo = repository
        self.config = config
        
        self.enabled = self.config.get("enabled", True)
        self.model_name = self.config.get("model_name", "BAAI/bge-reranker-v2-m3")
        self.model_version = self.config.get("model_version", "v1")
        self.use_cache = self.config.get("use_cache", True)
        self.model = None
        self._lock = threading.Lock()

    def _load_model(self):
        if not self.model and self.enabled:
            with self._lock:
                if not self.model:
                    logger.info(f"Loading Reranker Model: {self.model_name}")
                    try:
                        # Use sentence_transformers CrossEncoder
                        self.model = CrossEncoder(self.model_name, max_length=512)
                        logger.info("Reranker model loaded successfully.")
                    except Exception as e:
                        logger.error(f"Failed to load reranker model: {e}")
                        self.enabled = False

    def _generate_cache_key(self, normalized_explanation: str, variant_id: int):
        raw_key = f"{normalized_explanation}_{variant_id}_{self.model_version}"
        return hashlib.sha256(raw_key.encode('utf-8')).hexdigest()

    def _get_cached_score(self, cache_key: str):
        if not self.use_cache:
            return None
            
        conn = self.repo.get_connection()
        if not conn:
            return None
            
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT reranker_score FROM aml_reranker_cache WHERE cache_key = %s", (cache_key,))
                res = cur.fetchone()
                if res:
                    return float(res[0])
        except Exception as e:
            logger.error(f"Error reading reranker cache: {e}")
        finally:
            self.repo.release_connection(conn)
            
        return None

    def _save_cached_score(self, cache_key: str, normalized_explanation: str, variant_id: int, score: float):
        if not self.use_cache:
            return
            
        conn = self.repo.get_connection()
        if not conn:
            return
            
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO aml_reranker_cache (cache_key, normalized_explanation, variant_id, reranker_score, reranker_model_version)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (cache_key) DO NOTHING
                """, (cache_key, normalized_explanation, variant_id, score, self.model_version))
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving reranker cache: {e}")
            conn.rollback()
        finally:
            self.repo.release_connection(conn)

    def score_candidates(self, normalized_explanation: str, candidates: list):
        """
        Rerank a list of candidates.
        Each candidate dict must have 'variant_name' and 'variant_id'.
        Returns the candidates with 'reranker_score' added.
        """
        if not self.enabled or not candidates:
            for cand in candidates:
                cand['reranker_score'] = 0.0
            return candidates

        self._load_model()
        if not self.model:
            for cand in candidates:
                cand['reranker_score'] = 0.0
            return candidates

        pairs_to_score = []
        indices_to_score = []

        # Check cache first
        for idx, cand in enumerate(candidates):
            cache_key = self._generate_cache_key(normalized_explanation, cand["variant_id"])
            cached_score = self._get_cached_score(cache_key)
            
            if cached_score is not None:
                cand['reranker_score'] = cached_score
            else:
                pairs_to_score.append((normalized_explanation, cand["variant_name"]))
                indices_to_score.append((idx, cache_key))

        # Score missing pairs
        if pairs_to_score:
            try:
                # bge-reranker-v2-m3 returns logits. CrossEncoder predict handles it.
                # Usually we might want to apply sigmoid if the model doesn't output probabilities.
                # CrossEncoder with apply_softmax=False returns raw scores.
                # BAAI models usually need a sigmoid or are just raw logits. 
                # Let's normalize to 0-1 if they are logits, but SentenceTransformers might already handle it.
                scores = self.model.predict(pairs_to_score, show_progress_bar=False)
                
                # Apply simple normalization if scores are outside 0-1.
                # Sigmoid function for logits
                import math
                def sigmoid(x):
                    return 1 / (1 + math.exp(-x))

                for score, (idx, cache_key) in zip(scores, indices_to_score):
                    # Check if score needs sigmoid (e.g. if it can be negative)
                    normalized_score = float(sigmoid(score))
                    
                    candidates[idx]['reranker_score'] = normalized_score
                    self._save_cached_score(cache_key, normalized_explanation, candidates[idx]["variant_id"], normalized_score)
            except Exception as e:
                logger.error(f"Error scoring candidates with reranker: {e}")
                for idx, _ in indices_to_score:
                    candidates[idx]['reranker_score'] = 0.0

        return candidates
