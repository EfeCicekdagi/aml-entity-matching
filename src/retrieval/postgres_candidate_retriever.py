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
        self.reranker_prefilter_score = self.config.get("reranker_prefilter_score", 0.60)

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

    def batch_get_candidates(self, rows: list) -> dict:
        """
        HIGH PERFORMANCE: Process an entire chunk in just 2 SQL queries instead of
        3 queries per row (which would be 30,000 queries for a 10k chunk).
        
        Args:
            rows: List of dicts with keys: 'row_id', 'normalized_explanation', 'embedding'
        
        Returns:
            Dict mapping row_id -> list of candidate dicts
        """
        if not rows:
            return {}
        
        results = {row["row_id"]: [] for row in rows}
        
        conn = self.repo.get_connection()
        if not conn:
            return results
        
        try:
            with conn.cursor() as cur:
                # ── QUERY 1: Batch Trigram + Full-Text ──────────────────────────
                # UNNEST lets us send all 10,000 texts in a single round-trip.
                trgm_query = """
                    SELECT
                        input.row_id,
                        v.company_id,
                        v.variant_id,
                        similarity(input.norm_exp, v.normalized_variant_name) AS candidate_score,
                        v.original_company_name,
                        v.normalized_variant_name,
                        'pg_trgm' AS source
                    FROM
                        (SELECT UNNEST(%s::text[]) AS norm_exp,
                                UNNEST(%s::text[]) AS row_id) AS input
                    JOIN silver_company_variant v ON v.is_active = true
                    WHERE similarity(input.norm_exp, v.normalized_variant_name) >= %s
                    ORDER BY candidate_score DESC;
                """
                row_ids   = [str(r["row_id"])              for r in rows]
                norm_exps = [r["normalized_explanation"]    for r in rows]

                cur.execute(trgm_query, (norm_exps, row_ids, self.min_trgm_score))
                for row in cur.fetchall():
                    row_id = row[0]
                    if row_id in results:
                        results[row_id].append({
                            "company_id":    row[1],
                            "variant_id":    row[2],
                            "candidate_score": float(row[3]),
                            "company_name":  row[4],
                            "variant_name":  row[5],
                            "source":        row[6],
                        })

                # ── QUERY 2: Batch Vector Search (LATERAL JOIN) ─────────────────
                # Tüm chunk'ı TEK sorguda işler — eski yöntem 10k ayrı SQL'di.
                # Her embedding için LATERAL ile en yakın K sonuç çekilir.
                #
                # FIX: psycopg2, Python list[float]'ı numeric[][] olarak gönderir,
                # bu yüzden vector[] cast'i başarısız olur.
                # Çözüm: her embedding'i '[v1,v2,...]' string'ine çevirip
                # text[] olarak gönder, SQL içinde ::vector ile cast et.
                row_ids_arr    = [str(r["row_id"]) for r in rows]
                embeddings_arr = [
                    "[" + ",".join(str(x) for x in r["embedding"]) + "]"
                    for r in rows
                ]

                vec_batch_query = """
                    SELECT
                        input.row_id,
                        nearest.company_id,
                        nearest.variant_id,
                        nearest.candidate_score,
                        nearest.company_name,
                        nearest.variant_name
                    FROM
                        (SELECT
                             UNNEST(%s::text[])           AS row_id,
                             UNNEST(%s::text[])::vector   AS emb
                        ) AS input
                    CROSS JOIN LATERAL (
                        SELECT
                            e.company_id,
                            e.variant_id,
                            1 - (e.embedding <=> input.emb) AS candidate_score,
                            v.original_company_name        AS company_name,
                            v.normalized_variant_name      AS variant_name
                        FROM gold_company_embedding e
                        JOIN silver_company_variant v
                          ON e.variant_id = v.variant_id
                        WHERE v.is_active = true
                          AND 1 - (e.embedding <=> input.emb) >= %s
                        ORDER BY e.embedding <=> input.emb
                        LIMIT %s
                    ) AS nearest;
                """
                cur.execute(vec_batch_query,
                            (row_ids_arr, embeddings_arr,
                             self.min_vector_score, self.vector_top_k))

                for vrow in cur.fetchall():
                    rid = vrow[0]
                    if rid in results:
                        results[rid].append({
                            "company_id":      vrow[1],
                            "variant_id":      vrow[2],
                            "candidate_score": float(vrow[3]),
                            "company_name":    vrow[4],
                            "variant_name":    vrow[5],
                            "source":          "pgvector",
                        })

        except Exception as e:
            logger.error(f"Error in batch_get_candidates: {e}")
        finally:
            self.repo.release_connection(conn)

        # Merge & deduplicate per row
        merged_results = {}
        for rid, cands in results.items():
            merged = {}
            for cand in cands:
                key = (cand["company_id"], cand["variant_id"])
                if key not in merged:
                    merged[key] = cand
                else:
                    merged[key]["source"] = "combined"
                    if cand["candidate_score"] > merged[key]["candidate_score"]:
                        merged[key]["candidate_score"] = cand["candidate_score"]
            sorted_cands = sorted(merged.values(), key=lambda x: x["candidate_score"], reverse=True)
            merged_results[rid] = sorted_cands[:self.merged_top_k]

        return merged_results
