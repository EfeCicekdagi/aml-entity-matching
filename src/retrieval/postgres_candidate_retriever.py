"""
postgres_candidate_retriever.py — PostgreSQL tabanlı çok kanallı aday getirici (Trigram, FTS, Vektör).

Özellikler:
  - Toplu (batch) arama yapısı ile veritabanı sorgularını optimize eder.
  - Her işlem ve kanal için detaylı pipeline_status (TRIGRAM_NO_RESULT, FTS_NO_RESULT, VECTOR_NO_RESULT, ALL_RETRIEVAL_CHANNELS_EMPTY, CANDIDATES_FOUND) döndürür.
  - Trigram ön eleme eşiğinin altında kalan işlemler için FTS ve Vektör kanallarını çalıştırarak kapsama alanını maksimize eder.
"""

import logging
from typing import Optional
from src.config.db_tables import TABLES

logger = logging.getLogger(__name__)


class PostgresCandidateRetriever:
    """PostgreSQL tabanlı çok kanallı aday getirici (trigram, FTS, vector)."""

    def __init__(self, repository, config: dict):
        self.repo = repository
        self.config = config

        self.trgm_top_k    = self.config.get("pg_trgm_top_k", 20)
        self.full_text_top_k = self.config.get("full_text_top_k", 20)
        self.vector_top_k  = self.config.get("pgvector_top_k", 20)
        self.merged_top_k  = self.config.get("merged_top_k", 30)
        self.min_trgm_score   = self.config.get("min_trgm_score", 0.25)
        self.min_vector_score = self.config.get("min_vector_score", 0.50)
        self.reranker_prefilter_score = self.config.get("reranker_prefilter_score", 0.60)
        self.vector_enabled = self.config.get("vector_enabled", True)

    def disable_vector_retrieval(self, reason: str = "") -> None:
        """Vektör arama kanalını devre dışı bırakır (Graceful degradation)."""
        logger.warning(
            "Disabling vector retrieval channel in PostgresCandidateRetriever"
            + (f": {reason}" if reason else ".")
        )
        self.vector_enabled = False

    def validate_vector_dimension(self, model_dim: int) -> bool:
        """
        Veritabanındaki company_embedding.embedding kolonunun vektör boyutunu doğrular.
        Model boyutu (model_dim) ile DB kolonu eşleşmiyorsa False döner.
        """
        conn = self.repo.get_connection()
        if not conn:
            return False

        try:
            full_table = TABLES.get("company_embedding", "aml_ml.company_embedding")
            if "." in full_table:
                schema_name, table_name = full_table.split(".", 1)
            else:
                schema_name, table_name = "public", full_table

            with conn.cursor() as cur:
                cur.execute("""
                    SELECT format_type(a.atttypid, a.atttypmod), a.atttypmod
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = %s AND c.relname = %s AND a.attname = 'embedding' AND NOT a.attisdropped;
                """, (schema_name, table_name))
                row = cur.fetchone()
                if not row:
                    logger.warning(f"Column 'embedding' not found in table {full_table}.")
                    return True  # Kolon bulunamazsa engellemeyelim

                format_str, atttypmod = row[0], row[1]
                import re
                match = re.search(r"\((\d+)\)", str(format_str))
                if match:
                    db_dim = int(match.group(1))
                    if db_dim != model_dim:
                        logger.error(
                            f"Vector dimension mismatch! Database column {full_table}.embedding is {format_str} (dim={db_dim}), "
                            f"but Python embedding model generates {model_dim}-dimensional vectors. "
                            "INSERT and similarity queries (<=>) will fail!"
                        )
                        return False
                elif atttypmod > 0 and atttypmod != model_dim:
                    logger.error(
                        f"Vector dimension mismatch! Database column {full_table}.embedding has atttypmod={atttypmod}, "
                        f"but Python embedding model generates {model_dim}-dimensional vectors."
                    )
                    return False

            return True
        except Exception as e:
            logger.warning(f"Could not validate vector dimension: {e}")
            return True
        finally:
            self.repo.release_connection(conn)

    # ── Tekil sorgu metodları ─────────────────────────────────────────────────

    def retrieve_trgm_candidates(self, normalized_explanation: str) -> list:
        """
        pg_trgm ile trigram similarity tabanlı aday getirir.

        Args:
            normalized_explanation: Normalize edilmiş EFT açıklaması

        Returns:
            Aday listesi (trgm_score dahil)
        """
        conn = self.repo.get_connection()
        if not conn:
            return []

        candidates = []
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT
                        v.company_id,
                        v.variant_id,
                        similarity(%s, v.normalized_variant_name) AS candidate_score,
                        v.original_company_name,
                        v.normalized_variant_name,
                        v.variant_type,
                        COALESCE(v.alias_confidence, 1.0)
                    FROM {TABLES['company_variant']} v
                    WHERE v.is_active = true
                      AND similarity(%s, v.normalized_variant_name) >= %s
                    ORDER BY candidate_score DESC
                    LIMIT %s
                """, (normalized_explanation, normalized_explanation,
                      self.min_trgm_score, self.trgm_top_k))

                for row in cur.fetchall():
                    candidates.append({
                        "company_id":       row[0],
                        "variant_id":       row[1],
                        "trgm_score":       float(row[2]),
                        "candidate_score":  float(row[2]),
                        "company_name":     row[3],
                        "variant_name":     row[4],
                        "variant_type":     row[5],
                        "alias_confidence": float(row[6]) if row[6] is not None else 1.0,
                        "sources":          ["pg_trgm"],
                    })
        except Exception as e:
            logger.error(f"Error in trgm retrieval: {e}")
        finally:
            self.repo.release_connection(conn)

        return candidates

    def retrieve_full_text_candidates(self, normalized_explanation: str) -> list:
        """
        PostgreSQL Full-Text Search ile aday getirir.

        Args:
            normalized_explanation: Normalize edilmiş EFT açıklaması

        Returns:
            Aday listesi (full_text_score dahil)
        """
        conn = self.repo.get_connection()
        if not conn:
            return []

        candidates = []
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT
                        v.company_id,
                        v.variant_id,
                        ts_rank(
                            to_tsvector('simple', v.normalized_variant_name),
                            plainto_tsquery('simple', %s)
                        ) AS candidate_score,
                        v.original_company_name,
                        v.normalized_variant_name,
                        v.variant_type,
                        COALESCE(v.alias_confidence, 1.0)
                    FROM {TABLES['company_variant']} v
                    WHERE v.is_active = true
                      AND to_tsvector('simple', v.normalized_variant_name)
                          @@ plainto_tsquery('simple', %s)
                    ORDER BY candidate_score DESC
                    LIMIT %s
                """, (normalized_explanation, normalized_explanation, self.full_text_top_k))

                for row in cur.fetchall():
                    candidates.append({
                        "company_id":       row[0],
                        "variant_id":       row[1],
                        "full_text_score":  float(row[2]),
                        "candidate_score":  float(row[2]),
                        "company_name":     row[3],
                        "variant_name":     row[4],
                        "variant_type":     row[5],
                        "alias_confidence": float(row[6]) if row[6] is not None else 1.0,
                        "sources":          ["full_text"],
                    })
        except Exception as e:
            logger.error(f"Error in full-text retrieval: {e}")
        finally:
            self.repo.release_connection(conn)

        return candidates

    def retrieve_vector_candidates(self, query_embedding: list) -> list:
        """
        pgvector ile semantik benzerlik tabanlı aday getirir.

        Args:
            query_embedding: Sorgu vektörü (list of float)

        Returns:
            Aday listesi (vector_score dahil)
        """
        if not self.vector_enabled or not query_embedding:
            return []

        conn = self.repo.get_connection()
        if not conn:
            return []

        candidates = []
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT
                        e.company_id,
                        e.variant_id,
                        1 - (e.embedding <=> %s::vector) AS candidate_score,
                        v.original_company_name,
                        v.normalized_variant_name,
                        v.variant_type,
                        COALESCE(v.alias_confidence, 1.0)
                    FROM {TABLES['company_embedding']} e
                    JOIN {TABLES['company_variant']} v ON e.variant_id = v.variant_id
                    WHERE v.is_active = true
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, self.vector_top_k))

                for row in cur.fetchall():
                    score = float(row[2])
                    if score >= self.min_vector_score:
                        candidates.append({
                            "company_id":       row[0],
                            "variant_id":       row[1],
                            "vector_score":     score,
                            "candidate_score":  score,
                            "company_name":     row[3],
                            "variant_name":     row[4],
                            "variant_type":     row[5],
                            "alias_confidence": float(row[6]) if row[6] is not None else 1.0,
                            "sources":          ["pgvector"],
                        })
        except Exception as e:
            logger.error(f"Error in vector retrieval: {e}")
        finally:
            self.repo.release_connection(conn)

        return candidates

    def get_merged_candidates(
        self,
        normalized_explanation: str,
        query_embedding: Optional[list] = None
    ) -> list:
        """
        Tüm kanallardan aday getirir ve birleştirir.
        Her kanal bağımsız çalışır — birinin boş dönmesi diğerlerini durdurmaz.

        Args:
            normalized_explanation: Normalize edilmiş EFT açıklaması
            query_embedding: Embedding vektörü (opsiyonel)

        Returns:
            Birleştirilmiş, puanlanmış aday listesi
        """
        from src.utils.text_utils import normalize_leetspeak
        trgm_cands = self.retrieve_trgm_candidates(normalized_explanation)
        fts_cands  = self.retrieve_full_text_candidates(normalized_explanation)
        vec_cands  = self.retrieve_vector_candidates(query_embedding) if (query_embedding and self.vector_enabled) else []

        leet_exp = normalize_leetspeak(normalized_explanation)
        if leet_exp != normalized_explanation:
            trgm_cands += self.retrieve_trgm_candidates(leet_exp)
            fts_cands  += self.retrieve_full_text_candidates(leet_exp)

        return self._merge_candidates(trgm_cands + fts_cands + vec_cands)

    def _merge_candidates(self, candidates: list) -> list:
        """
        Aday listesini birleştirir ve tekrarlananları kaldırır.
        Her varyant için en yüksek kanal skoru korunur.

        Args:
            candidates: Ham aday listesi (farklı kanallardan)

        Returns:
            Birleştirilmiş, sıralanmış aday listesi
        """
        merged: dict = {}

        for cand in candidates:
            key = (cand["company_id"], cand["variant_id"])
            if key not in merged:
                merged[key] = cand.copy()
            else:
                # Kaynakları birleştir
                for src in cand.get("sources", []):
                    if src not in merged[key]["sources"]:
                        merged[key]["sources"].append(src)
                # En yüksek kanal skorunu sakla
                if "trgm_score" in cand:
                    merged[key]["trgm_score"] = max(
                        merged[key].get("trgm_score", 0.0), cand["trgm_score"]
                    )
                if "vector_score" in cand:
                    merged[key]["vector_score"] = max(
                        merged[key].get("vector_score", 0.0), cand["vector_score"]
                    )
                if "full_text_score" in cand:
                    merged[key]["full_text_score"] = max(
                        merged[key].get("full_text_score", 0.0), cand["full_text_score"]
                    )

        # Bileşik candidate_score: en yüksek kanal skoru
        for m in merged.values():
            m["candidate_score"] = max(
                m.get("trgm_score", 0.0),
                m.get("vector_score", 0.0),
                m.get("full_text_score", 0.0)
            )

        sorted_cands = sorted(merged.values(), key=lambda x: x["candidate_score"], reverse=True)
        return sorted_cands[:self.merged_top_k]

    def _compute_pipeline_status(self, row_candidates: list, channel_counts: dict) -> tuple[str, str]:
        """
        Retrieval sonucuna göre pipeline_status ve no_candidate_reason hesaplar.

        Args:
            row_candidates: Birleştirilmiş aday listesi
            channel_counts: Her kanal için bulunan aday sayısı {"trgm": N, "fts": M, "vector": K}

        Returns:
            (pipeline_status, no_candidate_reason)
        """
        if row_candidates:
            return "CANDIDATES_FOUND", None

        reasons = []
        if channel_counts.get("trgm", 0) == 0:
            reasons.append("TRIGRAM_NO_RESULT")
        if channel_counts.get("fts", 0) == 0:
            reasons.append("FTS_NO_RESULT")
        if self.vector_enabled and channel_counts.get("vector", 0) == 0:
            reasons.append("VECTOR_NO_RESULT")

        expected_channels = 3 if self.vector_enabled else 2
        if len(reasons) == expected_channels:
            return "ALL_RETRIEVAL_CHANNELS_EMPTY", "ALL_RETRIEVAL_CHANNELS_EMPTY"
        elif reasons:
            return reasons[0], ", ".join(reasons)

        return "CANDIDATES_FOUND", None

    def batch_get_candidates(self, rows: list) -> dict:
        """
        Tüm chunk'ı 2 optimize SQL sorgusu ile işler (batch verimli).
        Trigram eşik altında kalan satırlar için FTS ve vector kanalları devreye girerek aday aramaya devam eder.
        Hiçbir kanaldan aday bulunmazsa pipeline_status = ALL_RETRIEVAL_CHANNELS_EMPTY olarak işaretlenir.

        Args:
            rows: [{"row_id", "normalized_explanation", "embedding"}] listesi

        Returns:
            {row_id: {"candidates": [...], "pipeline_status": "...", ...}} dict
        """
        if not rows:
            return {}

        # Sonuç yapısı: candidates + kanal istatistikleri
        results: dict[str, list] = {row["row_id"]: [] for row in rows}
        channel_counts: dict[str, dict] = {
            row["row_id"]: {"trgm": 0, "fts": 0, "vector": 0}
            for row in rows
        }

        conn = self.repo.get_connection()
        if not conn:
            return results

        try:
            with conn.cursor() as cur:
                from src.utils.text_utils import normalize_leetspeak
                row_ids = []
                norm_exps = []
                for r in rows:
                    rid = str(r["row_id"])
                    n_exp = r["normalized_explanation"]
                    row_ids.append(rid)
                    norm_exps.append(n_exp)
                    l_exp = normalize_leetspeak(n_exp)
                    if l_exp != n_exp:
                        row_ids.append(rid)
                        norm_exps.append(l_exp)

                trgm_query = f"""
                    WITH input AS (
                        SELECT UNNEST(%s::text[]) AS norm_exp,
                               UNNEST(%s::text[]) AS row_id
                    )
                    SELECT
                        input.row_id,
                        nearest.company_id,
                        nearest.variant_id,
                        nearest.candidate_score,
                        nearest.original_company_name,
                        nearest.normalized_variant_name,
                        nearest.variant_type,
                        nearest.alias_confidence,
                        'pg_trgm' AS source
                    FROM input
                    CROSS JOIN LATERAL (
                        SELECT
                            v.company_id,
                            v.variant_id,
                            similarity(input.norm_exp, v.normalized_variant_name) AS candidate_score,
                            v.original_company_name,
                            v.normalized_variant_name,
                            v.variant_type,
                            COALESCE(v.alias_confidence, 1.0) AS alias_confidence
                        FROM {TABLES['company_variant']} v
                        WHERE v.is_active = true
                          AND similarity(input.norm_exp, v.normalized_variant_name) >= %s
                        ORDER BY similarity(input.norm_exp, v.normalized_variant_name) DESC
                        LIMIT %s
                    ) nearest
                """
                cur.execute(trgm_query, (norm_exps, row_ids, self.min_trgm_score, self.trgm_top_k))

                for row in cur.fetchall():
                    rid = row[0]
                    if rid in results:
                        results[rid].append({
                            "company_id":       row[1],
                            "variant_id":       row[2],
                            "trgm_score":       float(row[3]),
                            "candidate_score":  float(row[3]),
                            "company_name":     row[4],
                            "variant_name":     row[5],
                            "variant_type":     row[6],
                            "alias_confidence": float(row[7]) if row[7] is not None else 1.0,
                            "sources":          [row[8]],
                        })
                        channel_counts[rid]["trgm"] += 1

                # ── QUERY 2: Batch FTS ─────────────────────────────────────
                # Her norm_exp için FTS arama — UNNEST ile tek round-trip
                # NOT: PostgreSQL'de plainto_tsquery ile LATERAL kullanmak gerekir
                fts_query = f"""
                    WITH input AS (
                        SELECT UNNEST(%s::text[]) AS norm_exp,
                               UNNEST(%s::text[]) AS row_id
                    )
                    SELECT
                        input.row_id,
                        nearest.company_id,
                        nearest.variant_id,
                        nearest.candidate_score,
                        nearest.original_company_name,
                        nearest.normalized_variant_name,
                        nearest.variant_type,
                        nearest.alias_confidence
                    FROM input
                    CROSS JOIN LATERAL (
                        SELECT
                            v.company_id,
                            v.variant_id,
                            GREATEST(
                                ts_rank(to_tsvector('simple', v.normalized_variant_name), plainto_tsquery('simple', input.norm_exp)),
                                ts_rank(to_tsvector('simple', input.norm_exp), plainto_tsquery('simple', v.normalized_variant_name))
                            ) AS candidate_score,
                            v.original_company_name,
                            v.normalized_variant_name,
                            v.variant_type,
                            COALESCE(v.alias_confidence, 1.0) AS alias_confidence
                        FROM {TABLES['company_variant']} v
                        WHERE v.is_active = true
                          AND (to_tsvector('simple', v.normalized_variant_name) @@ plainto_tsquery('simple', input.norm_exp)
                               OR to_tsvector('simple', input.norm_exp) @@ plainto_tsquery('simple', v.normalized_variant_name))
                        ORDER BY GREATEST(
                            ts_rank(to_tsvector('simple', v.normalized_variant_name), plainto_tsquery('simple', input.norm_exp)),
                            ts_rank(to_tsvector('simple', input.norm_exp), plainto_tsquery('simple', v.normalized_variant_name))
                        ) DESC
                        LIMIT %s
                    ) nearest
                """
                cur.execute(fts_query, (norm_exps, row_ids, self.full_text_top_k))

                for row in cur.fetchall():
                    rid = row[0]
                    if rid in results:
                        results[rid].append({
                            "company_id":       row[1],
                            "variant_id":       row[2],
                            "full_text_score":  float(row[3]),
                            "candidate_score":  float(row[3]),
                            "company_name":     row[4],
                            "variant_name":     row[5],
                            "variant_type":     row[6],
                            "alias_confidence": float(row[7]) if row[7] is not None else 1.0,
                            "sources":          ["full_text"],
                        })
                        channel_counts[rid]["fts"] += 1

                if self.vector_enabled and any(r.get("embedding") is not None for r in rows):
                    valid_rows = [r for r in rows if r.get("embedding") is not None]
                    row_ids_arr    = [str(r["row_id"]) for r in valid_rows]
                    embeddings_arr = [
                        "[" + ",".join(str(x) for x in r["embedding"]) + "]"
                        for r in valid_rows
                    ]

                    vec_batch_query = f"""
                        SELECT
                            input.row_id,
                            nearest.company_id,
                            nearest.variant_id,
                            nearest.candidate_score,
                            nearest.company_name,
                            nearest.variant_name,
                            nearest.variant_type,
                            nearest.alias_confidence
                        FROM
                            (SELECT
                                 UNNEST(%s::text[])         AS row_id,
                                 UNNEST(%s::text[])::vector AS emb
                            ) AS input
                        CROSS JOIN LATERAL (
                            SELECT
                                e.company_id,
                                e.variant_id,
                                1 - (e.embedding <=> input.emb) AS candidate_score,
                                v.original_company_name AS company_name,
                                v.normalized_variant_name AS variant_name,
                                v.variant_type,
                                COALESCE(v.alias_confidence, 1.0) AS alias_confidence
                            FROM {TABLES['company_embedding']} e
                            JOIN {TABLES['company_variant']} v ON e.variant_id = v.variant_id
                            WHERE v.is_active = true
                              AND 1 - (e.embedding <=> input.emb) >= %s
                            ORDER BY e.embedding <=> input.emb
                            LIMIT %s
                        ) AS nearest
                    """
                    cur.execute(vec_batch_query,
                                (row_ids_arr, embeddings_arr,
                                 self.min_vector_score, self.vector_top_k))

                    for vrow in cur.fetchall():
                        rid = vrow[0]
                        if rid in results:
                            results[rid].append({
                                "company_id":       vrow[1],
                                "variant_id":       vrow[2],
                                "vector_score":     float(vrow[3]),
                                "candidate_score":  float(vrow[3]),
                                "company_name":     vrow[4],
                                "variant_name":     vrow[5],
                                "variant_type":     vrow[6],
                                "alias_confidence": float(vrow[7]) if vrow[7] is not None else 1.0,
                                "sources":          ["pgvector"],
                            })
                            channel_counts[rid]["vector"] += 1

        except Exception as e:
            logger.error(f"Error in batch_get_candidates: {e}")
        finally:
            self.repo.release_connection(conn)

        # ── Merge, deduplicate ve pipeline_status hesapla ─────────────────
        merged_results: dict = {}
        for rid, cands in results.items():
            merged = self._merge_candidates(cands)
            pipeline_status, no_candidate_reason = self._compute_pipeline_status(
                merged, channel_counts.get(rid, {})
            )
            merged_results[rid] = {
                "candidates":         merged,
                "pipeline_status":    pipeline_status,
                "no_candidate_reason": no_candidate_reason,
                "channel_counts":     channel_counts.get(rid, {}),
            }

        return merged_results


# ── Type alias ───────────────────────────────────────────────────────────────
from typing import Optional
