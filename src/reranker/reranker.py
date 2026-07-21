import logging
import hashlib
import threading
import torch
from sentence_transformers import CrossEncoder
from src.config.db_tables import TABLES

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self, repository, config):
        self.repo = repository
        self.config = config

        self.enabled      = self.config.get("enabled", True)
        self.model_name   = self.config.get("model_name", "BAAI/bge-reranker-v2-m3")
        self.model_version = self.config.get("model_version", "v1")
        self.use_cache    = self.config.get("use_cache", True)
        self.model        = None
        self._lock        = threading.Lock()
        self._predict_lock = threading.Lock()

        # Device selection: config > auto-detect
        cfg_device = self.config.get("device", "auto")
        if cfg_device == "auto" or cfg_device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg_device
        logger.info(f"Reranker will use device: {self.device}")

    def _load_model(self):
        if not self.model and self.enabled:
            with self._lock:
                if not self.model:
                    logger.info(f"Loading Reranker Model: {self.model_name} on {self.device}")
                    try:
                        self.model = CrossEncoder(
                            self.model_name,
                            max_length=512,
                            device=self.device
                        )
                        logger.info(f"Reranker model loaded successfully on {self.device}.")
                    except Exception as e:
                        logger.error(f"Failed to load reranker model: {e}")
                        self.enabled = False

    def _generate_cache_key(self, normalized_explanation: str, variant_id: int):
        raw_key = f"{normalized_explanation}_{variant_id}_{self.model_version}"
        return hashlib.sha256(raw_key.encode('utf-8')).hexdigest()

    def _get_cached_scores_batch(self, cache_keys: list) -> dict:
        """Fetch multiple reranker cache entries in a single SQL round-trip."""
        if not self.use_cache or not cache_keys:
            return {}

        conn = self.repo.get_connection()
        if not conn:
            return {}

        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT cache_key, raw_reranker_score, normalized_reranker_score FROM {TABLES['reranker_cache']} WHERE cache_key = ANY(%s)",
                    (cache_keys,)
                )
                return {row[0]: {"raw": float(row[1]) if row[1] is not None else 0.0, "norm": float(row[2]) if row[2] is not None else 0.0} for row in cur.fetchall()}
        except Exception as e:
            logger.error(f"Error batch-reading reranker cache: {e}")
            return {}
        finally:
            self.repo.release_connection(conn)

    def _save_cached_scores_batch(self, entries: list):
        """Bulk-insert reranker cache entries: entries = list of (cache_key, norm_exp, variant_id, score)."""
        if not self.use_cache or not entries:
            return

        conn = self.repo.get_connection()
        if not conn:
            return

        try:
            with conn.cursor() as cur:
                from psycopg2.extras import execute_values
                execute_values(cur, f"""
                    INSERT INTO {TABLES['reranker_cache']}
                        (cache_key, normalized_explanation, variant_id, raw_reranker_score, normalized_reranker_score, reranker_model_version)
                    VALUES %s
                    ON CONFLICT (cache_key) DO NOTHING
                """, [
                    (e["cache_key"], e["norm_exp"], e["variant_id"],
                     e["raw_score"], e["norm_score"], self.model_version)
                    for e in entries
                ])
            conn.commit()
        except Exception as e:
            logger.error(f"Error bulk-saving reranker cache: {e}")
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
                cand['raw_reranker_score'] = 0.0
                cand['normalized_reranker_score'] = 0.0
                cand['reranker_score'] = 0.0
            return candidates

        self._load_model()
        if not self.model:
            for cand in candidates:
                cand['raw_reranker_score'] = 0.0
                cand['normalized_reranker_score'] = 0.0
                cand['reranker_score'] = 0.0
            return candidates

        # ── Batch cache lookup (single SQL round-trip) ──────────────────────────
        cache_keys = [
            self._generate_cache_key(normalized_explanation, cand["variant_id"])
            for cand in candidates
        ]
        cached_map = self._get_cached_scores_batch(cache_keys)

        pairs_to_score  = []
        indices_to_score = []  # (candidate_idx, cache_key, variant_id)

        for idx, (cand, cache_key) in enumerate(zip(candidates, cache_keys)):
            if cache_key in cached_map:
                cand['raw_reranker_score'] = cached_map[cache_key]["raw"]
                cand['normalized_reranker_score'] = cached_map[cache_key]["norm"]
                cand['reranker_score'] = cached_map[cache_key]["norm"]
            else:
                pairs_to_score.append((normalized_explanation, cand["variant_name"]))
                indices_to_score.append((idx, cache_key, cand["variant_id"]))

        # ── Score missing pairs ─────────────────────────────────────────────────
        if pairs_to_score:
            try:
                with self._predict_lock:
                    scores = self.model.predict(pairs_to_score, show_progress_bar=False)

                new_cache_entries = []
                for score, (idx, cache_key, variant_id) in zip(scores, indices_to_score):
                    raw_score = float(score)
                    # BGE-M3 Reranker via CrossEncoder predict() already returns probabilities (0-1).
                    # Applying sigmoid again compresses the scores (e.g., 0.999 -> 0.73, 0 -> 0.5).
                    normalized_score = raw_score
                    logger.debug(f"Reranker computed - raw_score: {raw_score:.5f}, normalized_score: {normalized_score:.5f}")
                    
                    candidates[idx]['raw_reranker_score'] = raw_score
                    candidates[idx]['normalized_reranker_score'] = normalized_score
                    candidates[idx]['reranker_score'] = normalized_score
                    
                    new_cache_entries.append({
                        "cache_key":  cache_key,
                        "norm_exp":   normalized_explanation,
                        "variant_id": variant_id,
                        "raw_score":  raw_score,
                        "norm_score": normalized_score,
                    })

                # ── Bulk-write cache entries (single SQL round-trip) ────────────
                self._save_cached_scores_batch(new_cache_entries)

            except Exception as e:
                logger.error(f"Error scoring candidates with reranker: {e}")
                for idx, _, _ in indices_to_score:
                    candidates[idx]['raw_reranker_score'] = 0.0
                    candidates[idx]['normalized_reranker_score'] = 0.0
                    candidates[idx]['reranker_score'] = 0.0

        return candidates

    def score_candidates_bulk(self, bulk_data: dict) -> dict:
        """
        Rerank a bulk set of candidates across multiple rows.
        bulk_data: {row_id: {"norm_exp": "...", "candidates": [...]}}
        Modifies candidates in-place to add 'reranker_score'.
        """
        if not self.enabled or not bulk_data:
            return bulk_data

        self._load_model()
        if not self.model:
            return bulk_data

        flat_candidates = []
        cache_keys = []
        
        for row_id, data in bulk_data.items():
            norm_exp = data.get("norm_exp", "")
            for cand in data.get("candidates", []):
                flat_candidates.append((norm_exp, cand))
                cache_keys.append(self._generate_cache_key(norm_exp, cand["variant_id"]))
                
        if not flat_candidates:
            return bulk_data
            
        cached_map = self._get_cached_scores_batch(cache_keys)
        
        pairs_to_score = []
        indices_to_score = [] 
        
        for i, (norm_exp, cand) in enumerate(flat_candidates):
            ckey = cache_keys[i]
            if ckey in cached_map:
                cand['raw_reranker_score'] = cached_map[ckey]["raw"]
                cand['normalized_reranker_score'] = cached_map[ckey]["norm"]
                cand['reranker_score'] = cached_map[ckey]["norm"]
            else:
                pairs_to_score.append((norm_exp, cand["variant_name"]))
                indices_to_score.append((i, ckey, norm_exp, cand["variant_id"]))
                
        if pairs_to_score:
            try:
                with self._predict_lock:
                    scores = self.model.predict(pairs_to_score, batch_size=128, show_progress_bar=False)
                    
                new_cache_entries = []
                for score, (i, ckey, norm_exp, variant_id) in zip(scores, indices_to_score):
                    raw_score = float(score)
                    norm_score = raw_score
                    
                    cand = flat_candidates[i][1]
                    cand['raw_reranker_score'] = raw_score
                    cand['normalized_reranker_score'] = norm_score
                    cand['reranker_score'] = norm_score
                    
                    new_cache_entries.append({
                        "cache_key": ckey,
                        "norm_exp": norm_exp,
                        "variant_id": variant_id,
                        "raw_score": raw_score,
                        "norm_score": norm_score
                    })
                    
                self._save_cached_scores_batch(new_cache_entries)
            except Exception as e:
                logger.error(f"Error in bulk reranking: {e}")
                for i, _, _, _ in indices_to_score:
                    cand = flat_candidates[i][1]
                    cand['raw_reranker_score'] = 0.0
                    cand['normalized_reranker_score'] = 0.0
                    cand['reranker_score'] = 0.0
                    
        return bulk_data

