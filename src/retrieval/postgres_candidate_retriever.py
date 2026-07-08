import logging

logger = logging.getLogger(__name__)

class PostgresCandidateRetriever:
    def __init__(self, repository, config):
        self.repo = repository
        self.config = config
        
        self.trgm_top_k = self.config.get("pg_trgm_top_k", 20)
        self.full_text_top_k = self.config.get("full_text_top_k", 20)
        self.vector_top_k = self.config.get("pgvector_top_k", 20)
        self.merged_top_k = self.config.get("merged_top_k", 30)
        self.min_trgm_score = self.config.get("min_trgm_score", 0.25)
        self.min_vector_score = self.config.get("min_vector_score", 0.50)

    def retrieve_trgm_candidates(self, normalized_explanation: str):
        conn = self.repo.get_connection()
        if not conn:
            return []
            
        candidates = []
        try:
            with conn.cursor() as cur:
                query = """
                    SELECT 
                        v.company_id,
                        v.variant_id,
                        similarity(%s, v.normalized_variant_name) AS candidate_score,
                        v.original_company_name,
                        v.normalized_variant_name
                    FROM silver_company_variant v
                    WHERE v.is_active = true
                      AND similarity(%s, v.normalized_variant_name) >= %s
                    ORDER BY candidate_score DESC
                    LIMIT %s;
                """
                cur.execute(query, (normalized_explanation, normalized_explanation, self.min_trgm_score, self.trgm_top_k))
                for row in cur.fetchall():
                    candidates.append({
                        "company_id": row[0],
                        "variant_id": row[1],
                        "candidate_score": float(row[2]),
                        "company_name": row[3],
                        "variant_name": row[4],
                        "source": "pg_trgm"
                    })
        except Exception as e:
            logger.error(f"Error in trgm retrieval: {e}")
        finally:
            self.repo.release_connection(conn)
            
        return candidates

    def retrieve_full_text_candidates(self, normalized_explanation: str):
        conn = self.repo.get_connection()
        if not conn:
            return []
            
        candidates = []
        try:
            with conn.cursor() as cur:
                query = """
                    SELECT 
                        v.company_id,
                        v.variant_id,
                        ts_rank(
                            to_tsvector('simple', v.normalized_variant_name),
                            plainto_tsquery('simple', %s)
                        ) AS candidate_score,
                        v.original_company_name,
                        v.normalized_variant_name
                    FROM silver_company_variant v
                    WHERE v.is_active = true
                      AND to_tsvector('simple', v.normalized_variant_name) @@ plainto_tsquery('simple', %s)
                    ORDER BY candidate_score DESC
                    LIMIT %s;
                """
                cur.execute(query, (normalized_explanation, normalized_explanation, self.full_text_top_k))
                for row in cur.fetchall():
                    candidates.append({
                        "company_id": row[0],
                        "variant_id": row[1],
                        "candidate_score": float(row[2]),
                        "company_name": row[3],
                        "variant_name": row[4],
                        "source": "full_text"
                    })
        except Exception as e:
            logger.error(f"Error in full-text retrieval: {e}")
        finally:
            self.repo.release_connection(conn)
            
        return candidates

    def retrieve_vector_candidates(self, query_embedding: list):
        if not query_embedding:
            return []
            
        conn = self.repo.get_connection()
        if not conn:
            return []
            
        candidates = []
        try:
            with conn.cursor() as cur:
                query = """
                    SELECT
                        e.company_id,
                        e.variant_id,
                        1 - (e.embedding <=> %s::vector) AS candidate_score,
                        v.original_company_name,
                        v.normalized_variant_name
                    FROM gold_company_embedding e
                    JOIN silver_company_variant v ON e.variant_id = v.variant_id
                    WHERE v.is_active = true
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s;
                """
                cur.execute(query, (query_embedding, query_embedding, self.vector_top_k))
                for row in cur.fetchall():
                    score = float(row[2])
                    if score >= self.min_vector_score:
                        candidates.append({
                            "company_id": row[0],
                            "variant_id": row[1],
                            "candidate_score": score,
                            "company_name": row[3],
                            "variant_name": row[4],
                            "source": "pgvector"
                        })
        except Exception as e:
            logger.error(f"Error in vector retrieval: {e}")
        finally:
            self.repo.release_connection(conn)
            
        return candidates

    def get_merged_candidates(self, normalized_explanation: str, query_embedding: list = None):
        """
        Retrieves candidates from all 3 sources and merges them.
        Deduplicates by taking the highest score for the same variant.
        """
        
        trgm_cands = self.retrieve_trgm_candidates(normalized_explanation)
        fts_cands = self.retrieve_full_text_candidates(normalized_explanation)
        vec_cands = self.retrieve_vector_candidates(query_embedding)
        
        merged = {}
        
        for cand in trgm_cands + fts_cands + vec_cands:
            key = (cand["company_id"], cand["variant_id"])
            if key not in merged:
                merged[key] = cand
            else:
                # Update source to combined if found in multiple sources
                merged[key]["source"] = "combined"
                # Keep the max score
                if cand["candidate_score"] > merged[key]["candidate_score"]:
                    merged[key]["candidate_score"] = cand["candidate_score"]
                    
        # Sort by score descending
        sorted_candidates = sorted(merged.values(), key=lambda x: x["candidate_score"], reverse=True)
        
        # Return top K
        return sorted_candidates[:self.merged_top_k]
