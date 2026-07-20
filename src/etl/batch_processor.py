"""
batch_processor.py — AML batch işleme motoru.

Değişiklikler (v3):
  - Pre-screening artık CLEAN kararı vermiyor.
    Trigram eşik altında olan EFT'ler için FTS + vector çalışmaya devam ediyor.
  - Hiçbir kanaldan aday bulunamazsa: decision_status = NO_CANDIDATE_FOUND
    (CLEAN değil!)
  - Tüm sonuçlar match_result tablosuna yazılıyor.
  - Sadece HIGH ve MEDIUM risk → alert tablosuna yazılıyor.
  - LOW kayıtlar artık alert tablosuna yazılmıyor.
  - Teknik pipeline_status ile iş decision_status ayrıştırıldı.
  - EntityExtractor (çok katmanlı) kullanılıyor.
  - Pipeline aşaması süresi ölçülüyor.
  - reason_codes, calibrated_probability, human_explanation üretiliyor.
"""

import pandas as pd
import logging
import math
import sys
import os
import time
import json
import torch
import concurrent.futures
import numpy as np
from typing import Optional
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.alias_utils import generate_acronym
from src.utils.text_utils import normalize_text, remove_company_suffixes, get_normalized_core_name, is_consonant_match
from src.utils.entity_extractor import EntityExtractor
from src.config.db_tables import TABLES
from src.scoring.reason_codes import (
    ReasonCode, build_human_explanation, codes_to_list, list_to_codes
)

logger = logging.getLogger(__name__)

# ── Rule-based scoring yardımcıları (geriye uyumlu) ──────────────────────────
_RULE_STOPWORDS = {
    "services", "service", "group", "holding", "holdings",
    "international", "global", "enterprises", "enterprise",
    "solutions", "solution", "industries", "industry",
    "management", "investments", "investment",
    "trading", "trade", "export", "import",
    "logistics", "transport", "energy", "petroleum",
}


def _acronym_score(explanation: str, variant_name: str) -> float:
    """EFT açıklamasında şirket kısaltması arar. 1.0 veya 0.0 döner."""
    acronym = generate_acronym(variant_name)
    if acronym and len(acronym) >= 2 and acronym in explanation.split():
        return 1.0
    return 0.0


def _rule_score(explanation: str, variant_name: str) -> float:
    """Token overlap skoru. Generic kelimeler ve kısa tokenler hariç."""
    from src.utils.text_utils import normalize_text, remove_company_suffixes
    clean_variant = remove_company_suffixes(normalize_text(variant_name))
    variant_tokens = {
        t for t in clean_variant.split()
        if len(t) > 3 and t not in _RULE_STOPWORDS
    }
    if not variant_tokens:
        return 0.0
    exp_tokens = set(explanation.split())
    overlap = variant_tokens & exp_tokens
    return len(overlap) / len(variant_tokens)


def _exact_name_score(explanation: str, variant_name: str) -> float:
    """Variant adı EFT'de tam geçiyor mu? En az 2 anlamlı token gerektirir."""
    norm_variant = normalize_text(variant_name)
    tokens = [t for t in norm_variant.split() if len(t) > 3]
    if len(tokens) < 2:
        return 0.0
    exp_tokens = set(explanation.split())
    if all(t in exp_tokens for t in tokens):
        return 1.0
    return 0.0


def _compute_latency_percentiles(latencies_ms: list) -> dict:
    """P50, P95, P99 latency hesaplar."""
    if not latencies_ms:
        return {"p50": None, "p95": None, "p99": None}
    arr = np.array(latencies_ms)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


class BatchProcessor:
    """AML batch işleme motoru. Chunk bazlı veri işleme ve alert üretimi."""

    def __init__(self, repository, config: dict, retriever, reranker, scorer,
                 calibration=None):
        """
        Args:
            repository: AMLRepository instance
            config: aml_config.yaml içeriği
            retriever: PostgresCandidateRetriever instance
            reranker: Reranker instance
            scorer: FinalScorer instance
            calibration: CalibrationWrapper instance (opsiyonel)
        """
        self.repo        = repository
        self.config      = config
        self.retriever   = retriever
        self.reranker    = reranker
        self.scorer      = scorer
        self.calibration = calibration  # Opsiyonel kalibrasyon

        self.batch_size          = config.get("embedding", {}).get("batch_size", 32)
        self.embedding_model_name = config.get("embedding", {}).get("model_name", "BAAI/bge-m3")
        self.embedding_model     = None

        # Device
        cfg_device = config.get("embedding", {}).get("device", "auto")
        self.device = "cuda" if (cfg_device in ("auto", None) and torch.cuda.is_available()) else (
            cfg_device if cfg_device not in ("auto", None) else "cpu"
        )
        logger.info(f"Embedding model will use device: {self.device}")

        self.reranker_prefilter_score = (
            config.get("retrieval", {}).get("reranker_prefilter_score", 0.60)
        )

        # Pre-screening için trigram eşiği — artık CLEAN kararı için değil,
        # sadece prescreen_skipped_count istatistiği için
        self.min_trgm_score = config.get("retrieval", {}).get("min_trgm_score", 0.25)

        # Multi-layer entity extractor
        self.ner_extractor = None
        self._entity_extractor = None
        ner_enabled = config.get("ner", {}).get("enabled", False)
        if ner_enabled:
            try:
                from src.utils.ner_extractor import NERExtractor
                ner_model = config.get("ner", {}).get("model_name", "savasy/bert-base-turkish-ner-cased")
                ner_dev = config.get("ner", {}).get("device", "auto")
                dev_id = 0 if (ner_dev == "auto" and torch.cuda.is_available()) or ner_dev == "cuda" else -1
                self.ner_extractor = NERExtractor(model_name=ner_model, device=dev_id)
            except Exception as e:
                logger.warning(f"NER model could not be loaded: {e}")

        # EntityExtractor (NER + fallback katmanları)
        self._entity_extractor = EntityExtractor(
            ner_extractor=self.ner_extractor,
            config=config.get("entity_extraction", {})
        )

    def _load_embedding_model(self) -> None:
        """Embedding modelini lazy yükler."""
        if not self.embedding_model:
            logger.info(f"Loading embedding model: {self.embedding_model_name} on {self.device}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
            logger.info("Embedding model loaded.")

    # ── Pre-screening (istatistik amaçlı, artık bloklayıcı değil) ───────────

    def _prescreen_eft_chunk(self, norm_explanations: list, row_ids: list) -> set:
        """
        Trigram pre-screening — artık CLEAN kararı için kullanılmıyor.

        Yalnızca prescreen istatistiği için: hangi EFT'lerin trigram eşiği
        altında kaldığını kaydet. Eşik altı EFT'ler yine de FTS + vector
        kanallarından geçecek.

        Returns:
            Set of row_ids with trigram score >= min_trgm_score
        """
        if not norm_explanations:
            return set()

        conn = self.repo.get_connection()
        if not conn:
            return set(row_ids)  # Fail-safe: hepsini işle

        suspicious_ids = set()
        try:
            with conn.cursor() as cur:
                prescreen_query = f"""
                    SELECT input.row_id
                    FROM (
                        SELECT
                            UNNEST(%s::text[]) AS norm_exp,
                            UNNEST(%s::text[]) AS row_id
                    ) AS input
                    JOIN LATERAL (
                        SELECT word_similarity(v.normalized_variant_name, input.norm_exp) AS sim
                        FROM {TABLES['company_variant']} v
                        WHERE v.is_active = true
                        ORDER BY v.normalized_variant_name <<-> input.norm_exp
                        LIMIT 1
                    ) AS nearest ON nearest.sim >= %s
                """
                cur.execute(prescreen_query, (
                    norm_explanations,
                    [str(rid) for rid in row_ids],
                    self.min_trgm_score
                ))
                for row in cur.fetchall():
                    suspicious_ids.add(row[0])
        except Exception as e:
            logger.warning(f"Pre-screening query failed (continuing with all rows): {e}")
            return set(str(rid) for rid in row_ids)
        finally:
            self.repo.release_connection(conn)

        return suspicious_ids

    def process_db_table_in_chunks(
        self,
        run_id: str,
        batch_id: str,
        table_name: str = None,
        chunk_size: int = 2000
    ) -> None:
        """
        EFT verilerini chunk'lar halinde okuyup işler.

        Yeni akış:
          1. Tüm EFT'lere trigram pre-screen (istatistik amaçlı)
          2. Hepsi için NER + embedding hesapla
          3. batch_get_candidates → pipeline_status al
          4. Eğer ALL_RETRIEVAL_CHANNELS_EMPTY → decision_status=NO_CANDIDATE_FOUND
          5. Reranker + scoring → reason_codes üret
          6. Tüm sonuçlar → match_result tablosu
          7. Sadece HIGH/MEDIUM → alert tablosu
        """
        if table_name is None:
            table_name = TABLES["eft_input"]

        self._load_embedding_model()

        pipeline_start = time.time()

        # ── Metrik sayaçları ─────────────────────────────────────────────────
        metrics = {
            "input_row_count":        0,
            "input_count":            0,
            "processed_row_count":    0,
            "candidate_count":        0,
            "alert_count":            0,
            "high_alert_count":       0,
            "medium_alert_count":     0,
            "no_match_count":         0,
            "no_candidate_count":     0,
            "match_result_count":     0,
            "error_count":            0,
            "prescreen_skipped_count": 0,
            "ner_duration_s":         0.0,
            "retrieval_duration_s":   0.0,
            "reranker_duration_s":    0.0,
            "scoring_duration_s":     0.0,
        }
        per_row_latencies = []  # ms cinsinden

        embedding_model_name = self.config.get("embedding", {}).get("model_name", "UNKNOWN")
        reranker_model_name  = self.config.get("reranker",  {}).get("model_name", "UNKNOWN")

        self.repo.start_run_log(
            run_id,
            pipeline_name    = "AML_Production_Pipeline",
            embedding_model  = embedding_model_name,
            reranker_model   = reranker_model_name,
            scoring_config_version = self.config.get("scoring", {}).get("scoring_config_version"),
            threshold_version      = self.config.get("scoring", {}).get("threshold_config_version"),
            pipeline_version       = "aml_pipeline_v3",
            ner_model_name   = self.config.get("ner", {}).get("model_name"),
            watchlist_version= self.config.get("watchlist", {}).get("version"),
        )

        # Toplam satır sayısı (progress için)
        try:
            conn = self.repo.get_connection()
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                total_rows = cur.fetchone()[0]
        except Exception:
            total_rows = None
        finally:
            if "conn" in locals() and conn:
                self.repo.release_connection(conn)

        total_chunks = math.ceil(total_rows / chunk_size) if total_rows else "?"

        try:
            conn_for_read = self.repo.get_connection()
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    chunk_iter = pd.read_sql(
                        f"SELECT * FROM {table_name} ORDER BY eft_id",
                        con=conn_for_read,
                        chunksize=chunk_size
                    )

                    for chunk_idx, chunk in enumerate(chunk_iter):
                        pct = (
                            f"{100*(chunk_idx+1)/total_chunks:.1f}%"
                            if isinstance(total_chunks, int) else "?%"
                        )
                        logger.info(
                            f"[Chunk {chunk_idx+1}/{total_chunks} | {pct}] "
                            f"Processing {len(chunk)} rows..."
                        )
                        if len(chunk) == 0:
                            continue

                        metrics["input_row_count"] += len(chunk)
                        metrics["input_count"]     += len(chunk)

                        chunk_match_results = []
                        chunk_alerts        = []

                        self._process_chunk(
                            chunk, run_id, metrics,
                            per_row_latencies, chunk_match_results, chunk_alerts
                        )

                        # ── Batch write ──────────────────────────────────────
                        if chunk_match_results:
                            self.repo.insert_match_results_bulk(chunk_match_results)
                            metrics["match_result_count"] += len(chunk_match_results)
                            logger.info(f"  → {len(chunk_match_results)} match result(s) written.")

                        if chunk_alerts:
                            self.repo.insert_alerts_bulk(chunk_alerts)
                            logger.info(f"  → {len(chunk_alerts)} alert(s) written.")

            finally:
                if "conn_for_read" in locals() and conn_for_read:
                    self.repo.release_connection(conn_for_read)

            # ── Latency percentiles ──────────────────────────────────────────
            latency_stats = _compute_latency_percentiles(per_row_latencies)
            total_duration = time.time() - pipeline_start
            metrics["rows_per_second"] = (
                metrics["processed_row_count"] / total_duration if total_duration > 0 else 0.0
            )
            metrics["avg_candidate_per_row"] = (
                metrics["candidate_count"] / max(metrics["processed_row_count"], 1)
            )

            self.repo.finish_run_log(
                run_id, metrics,
                duration_seconds=total_duration,
            )
            # Latency percentiles ayrıca güncellenir
            self._update_latency_metrics(run_id, latency_stats)

            logger.info(
                f"Batch complete. Rows: {metrics['input_row_count']} | "
                f"Candidates: {metrics['candidate_count']} | "
                f"HIGH: {metrics['high_alert_count']} | "
                f"MEDIUM: {metrics['medium_alert_count']} | "
                f"NO_CANDIDATE: {metrics['no_candidate_count']} | "
                f"Duration: {total_duration:.1f}s"
            )

        except Exception as e:
            logger.error(f"Fatal error in batch processing: {e}", exc_info=True)
            self.repo.fail_run_log(run_id, str(e))

    def _process_chunk(
        self,
        chunk: pd.DataFrame,
        run_id: str,
        metrics: dict,
        per_row_latencies: list,
        chunk_match_results: list,
        chunk_alerts: list
    ) -> None:
        """
        Tek chunk'ı işler: embedding, entity extraction, retrieval, reranking, scoring.
        Sonuçları chunk_match_results ve chunk_alerts listelerine ekler.
        """
        # ── 1. Text normalization ────────────────────────────────────────────
        chunk["normalized_explanation"] = chunk["explanation"].astype(str).str.casefold()
        explanations = chunk["normalized_explanation"].tolist()

        raw_row_ids = [
            str(row["eft_id"]) if "eft_id" in row else str(idx)
            for idx, (_, row) in enumerate(chunk.iterrows())
        ]

        # ── 1.5 Pre-screen (istatistik, bloklayıcı DEĞİL) ──────────────────
        suspicious_ids = self._prescreen_eft_chunk(explanations, raw_row_ids)
        prescreen_skipped = len(raw_row_ids) - len(suspicious_ids)
        metrics["prescreen_skipped_count"] += prescreen_skipped
        logger.info(
            f"  [Pre-Screen] {len(suspicious_ids)}/{len(raw_row_ids)} trigram eşiği üstünde "
            f"({prescreen_skipped} düşük trigram — FTS+Vector yine çalışacak)"
        )

        # ── 2. NER + Entity Extraction ───────────────────────────────────────
        ner_start = time.time()
        extraction_results = self._entity_extractor.batch_extract(explanations)
        metrics["ner_duration_s"] = metrics.get("ner_duration_s", 0) + (time.time() - ner_start)

        # ── 3. Embedding ─────────────────────────────────────────────────────
        embeddings = self.embedding_model.encode(
            explanations, batch_size=self.batch_size, show_progress_bar=False
        )

        # ── 4. Batch aday retrieval ──────────────────────────────────────────
        rows_for_batch = [
            {
                "row_id":                raw_row_ids[i],
                "normalized_explanation": explanations[i],
                "embedding":             embeddings[i].tolist(),
            }
            for i in range(len(raw_row_ids))
        ]

        retrieval_start = time.time()
        all_retrieval_data = self.retriever.batch_get_candidates(rows_for_batch)
        metrics["retrieval_duration_s"] = (
            metrics.get("retrieval_duration_s", 0) + (time.time() - retrieval_start)
        )

        total_candidates = sum(
            len(v["candidates"]) for v in all_retrieval_data.values()
        )
        metrics["candidate_count"] += total_candidates

        # ── 5. Row bazında rerank + score ────────────────────────────────────
        row_lookup       = {r["row_id"]: r["normalized_explanation"] for r in rows_for_batch}
        extraction_lookup = {raw_row_ids[i]: extraction_results[i] for i in range(len(raw_row_ids))}
        eft_id_lookup    = {}
        for _, row in chunk.iterrows():
            rid = str(row.get("eft_id", ""))
            eft_id_lookup[rid] = int(row.get("eft_id", 0))

        reranker_total_s = 0.0
        scoring_total_s  = 0.0

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            future_map = {
                executor.submit(
                    self._score_row,
                    rid,
                    retrieval_data,
                    row_lookup,
                    extraction_lookup,
                    run_id,
                    eft_id_lookup,
                ): rid
                for rid, retrieval_data in all_retrieval_data.items()
            }

            for future in concurrent.futures.as_completed(future_map):
                try:
                    row_result = future.result()
                    metrics["processed_row_count"] += 1
                    per_row_latencies.append(row_result.get("latency_ms", 0))
                    reranker_total_s += row_result.get("reranker_s", 0)
                    scoring_total_s  += row_result.get("scoring_s", 0)

                    if row_result["no_candidate"]:
                        metrics["no_candidate_count"] += 1

                    metrics["high_alert_count"]   += row_result["high_count"]
                    metrics["medium_alert_count"] += row_result["medium_count"]
                    metrics["no_match_count"]     += row_result["no_match_count"]
                    metrics["alert_count"]        += row_result["high_count"] + row_result["medium_count"]

                    chunk_match_results.extend(row_result["match_results"])
                    chunk_alerts.extend(row_result["alerts"])

                except Exception as e:
                    logger.error(f"Row scoring failed: {e}", exc_info=True)
                    metrics["processed_row_count"] += 1
                    metrics["error_count"]         += 1

        metrics["reranker_duration_s"] = metrics.get("reranker_duration_s", 0) + reranker_total_s
        metrics["scoring_duration_s"]  = metrics.get("scoring_duration_s", 0) + scoring_total_s

    def _score_row(
        self,
        row_id: str,
        retrieval_data: dict,
        row_lookup: dict,
        extraction_lookup: dict,
        run_id: str,
        eft_id_lookup: dict,
    ) -> dict:
        """
        Tek bir EFT satırını rerank + score eder.

        Returns:
            dict içinde: match_results, alerts, istatistikler
        """
        row_start = time.time()
        result = {
            "match_results": [],
            "alerts":        [],
            "no_candidate":  False,
            "high_count":    0,
            "medium_count":  0,
            "no_match_count": 0,
            "latency_ms":    0,
            "reranker_s":    0.0,
            "scoring_s":     0.0,
        }

        norm_exp   = row_lookup.get(row_id, "")
        extraction = extraction_lookup.get(row_id)
        eft_id     = eft_id_lookup.get(row_id, 0)

        candidates       = retrieval_data.get("candidates", [])
        pipeline_status  = retrieval_data.get("pipeline_status", "CANDIDATES_FOUND")
        no_cand_reason   = retrieval_data.get("no_candidate_reason")
        channel_counts   = retrieval_data.get("channel_counts", {})
        candidate_count  = len(candidates)

        # ── NO_CANDIDATE_FOUND: hiçbir kanaldan aday yok ─────────────────────
        if not candidates:
            result["no_candidate"] = True
            result["match_results"].append({
                "run_id":              run_id,
                "eft_id":              eft_id,
                "pipeline_status":     pipeline_status,
                "no_candidate_reason": no_cand_reason or "ALL_RETRIEVAL_CHANNELS_EMPTY",
                "decision_status":     "NO_CANDIDATE_FOUND",
                "candidate_count":     0,
                "extracted_entity":    extraction.extracted_entity if extraction else None,
                "entity_type":         extraction.entity_type if extraction else "UNKNOWN",
                "extraction_method":   extraction.extraction_method if extraction else None,
                "extraction_confidence": extraction.extraction_confidence if extraction else 0.0,
                "entity_extraction_status": extraction.entity_extraction_status if extraction else "NOT_FOUND",
                "reason_codes":        [ReasonCode.NO_CANDIDATE_FOUND.value],
                "human_explanation":   "Kara liste veya yaptırım listesinde bu metinle eşleşen kayıt bulunamadı.",
                "retrieval_sources":   channel_counts,
            })
            result["latency_ms"] = (time.time() - row_start) * 1000
            return result

        # ── Pre-filter: reranker'a göndermeden önce zayıfları eleme ─────────
        strong = [
            c for c in candidates
            if c.get("candidate_score", 0.0) >= self.reranker_prefilter_score
            or _exact_name_score(norm_exp, c.get("variant_name", "")) == 1.0
        ]

        # ── Reranker ─────────────────────────────────────────────────────────
        reranker_start = time.time()
        if strong:
            strong = self.reranker.score_candidates(norm_exp, strong)
        else:
            strong = candidates  # Hepsi düşükse yine de devam et
        result["reranker_s"] = time.time() - reranker_start

        # ── Scoring ──────────────────────────────────────────────────────────
        scoring_start = time.time()
        for rank_idx, cand in enumerate(strong):
            self._process_candidate(
                cand, rank_idx, norm_exp, extraction, eft_id, run_id,
                pipeline_status, no_cand_reason, channel_counts, candidate_count,
                result
            )
        result["scoring_s"] = time.time() - scoring_start

        result["latency_ms"] = (time.time() - row_start) * 1000
        return result

    def _process_candidate(
        self,
        cand: dict,
        rank_idx: int,
        norm_exp: str,
        extraction,
        eft_id: int,
        run_id: str,
        pipeline_status: str,
        no_cand_reason,
        channel_counts: dict,
        candidate_count: int,
        result: dict,
    ) -> None:
        """
        Tek bir aday için skor hesaplar ve sonuçları result'a ekler.
        """
        import difflib

        fuzzy_score  = cand.get("trgm_score", 0.0)
        vector_score = cand.get("vector_score", 0.0)
        fts_score    = cand.get("full_text_score", 0.0)
        raw_reranker = cand.get("raw_reranker_score", 0.0)
        norm_reranker = cand.get("normalized_reranker_score", 0.0)
        alias_confidence = cand.get("alias_confidence", 1.0)

        norm_cand   = normalize_text(cand.get("variant_name", ""))
        core_query  = get_normalized_core_name(norm_exp)
        core_cand   = get_normalized_core_name(cand.get("variant_name", ""))

        exact_normalized_match     = bool(norm_exp == norm_cand and norm_exp)
        exact_core_match           = bool(core_query == core_cand and core_query)
        legal_suffix_only_diff     = exact_core_match and not exact_normalized_match
        query_is_contained         = bool(norm_exp in norm_cand and norm_exp)
        cand_is_contained          = bool(norm_cand in norm_exp and norm_cand)

        extracted_entity    = extraction.extracted_entity if extraction else None
        entity_type         = extraction.entity_type if extraction else "UNKNOWN"
        extraction_method   = extraction.extraction_method if extraction else "ENTITY_NOT_FOUND"
        extraction_confidence = extraction.extraction_confidence if extraction else 0.0
        entity_status       = extraction.entity_extraction_status if extraction else "NOT_FOUND"

        # Fallback: variant adı EFT içinde geçiyorsa entity olarak kullan
        if not extracted_entity and cand_is_contained:
            extracted_entity  = cand.get("variant_name")
            extraction_method = "FALLBACK_MATCHED_VARIANT"
            entity_status     = "FALLBACK"

        query_token_count = len(norm_exp.split())

        scores_dict = {
            "fuzzy_score":    fuzzy_score,
            "vector_score":   vector_score,
            "acronym_score":  _acronym_score(norm_exp, cand.get("variant_name", "")),
            "rule_score":     max(
                _rule_score(norm_exp, cand.get("variant_name", "")),
                _exact_name_score(norm_exp, cand.get("variant_name", ""))
            ),
            "reranker_score": norm_reranker,
            "query_token_count": query_token_count,
            "exact_normalized_match": exact_normalized_match,
            "exact_core_match": exact_core_match,
            "legal_suffix_only_difference": legal_suffix_only_diff,
            "query_is_contained_in_candidate": query_is_contained,
            "candidate_is_contained_in_query": cand_is_contained,
            "consonant_match": is_consonant_match(core_query, core_cand),
            "_query_str":   norm_exp,
            "_variant_str": norm_cand,
        }

        # Extracted entity varsa fuzzy/rule skorlarını güncelle
        if extracted_entity:
            fuzzy_ext = difflib.SequenceMatcher(
                None, extracted_entity.lower(), cand.get("variant_name", "").lower()
            ).ratio()
            if core_cand:
                fuzzy_ext = max(
                    fuzzy_ext,
                    difflib.SequenceMatcher(None, extracted_entity.lower(), core_cand.lower()).ratio()
                )
            scores_dict["fuzzy_score"] = max(scores_dict["fuzzy_score"], fuzzy_ext)

            if is_consonant_match(extracted_entity, core_cand):
                scores_dict["consonant_match"] = True

            scores_dict["acronym_score"] = max(
                scores_dict["acronym_score"],
                _acronym_score(extracted_entity, cand.get("variant_name", ""))
            )
            scores_dict["rule_score"] = max(
                scores_dict["rule_score"],
                max(
                    _rule_score(extracted_entity, cand.get("variant_name", "")),
                    _exact_name_score(extracted_entity, cand.get("variant_name", ""))
                )
            )

        # ── Final scoring ────────────────────────────────────────────────────
        final_score, match_reason, reason_codes = self.scorer.calculate_final_score(
            scores_dict, alias_confidence=alias_confidence
        )
        risk_level      = self.scorer.assign_risk_level(final_score)
        decision_status = self.scorer.assign_decision_status(risk_level)

        # ── Calibration ──────────────────────────────────────────────────────
        calibrated_prob  = None
        calibration_applied = False
        calibration_method  = None
        calibration_version = None

        if self.calibration:
            cal_result = self.calibration.calibrate(norm_reranker)
            calibrated_prob       = cal_result.calibrated_probability
            calibration_applied   = cal_result.calibration_applied
            calibration_method    = cal_result.calibration_method
            calibration_version   = cal_result.calibration_version

            if not calibration_applied:
                if ReasonCode.CALIBRATION_NOT_APPLIED.value not in reason_codes:
                    reason_codes.append(ReasonCode.CALIBRATION_NOT_APPLIED.value)
            else:
                if ReasonCode.CALIBRATION_APPLIED.value not in reason_codes:
                    reason_codes.append(ReasonCode.CALIBRATION_APPLIED.value)

        # İnsan tarafından okunabilir açıklama
        from src.scoring.reason_codes import list_to_codes
        human_exp = build_human_explanation(
            entity       = extracted_entity,
            matched_name = cand.get("variant_name"),
            codes        = list_to_codes(reason_codes),
            final_score  = final_score,
            calibrated_probability = calibrated_prob
        )

        # ── match_result kaydı (tüm kararlar) ────────────────────────────────
        match_record = {
            "run_id":              run_id,
            "eft_id":              eft_id,
            "candidate_company_id": cand.get("company_id"),
            "variant_id":          cand.get("variant_id"),
            "extracted_entity":    extracted_entity,
            "entity_type":         entity_type,
            "extraction_method":   extraction_method,
            "extraction_confidence": extraction_confidence,
            "entity_extraction_status": entity_status,
            "trigram_score":       fuzzy_score,
            "full_text_score":     fts_score,
            "vector_score":        vector_score,
            "fuzzy_score":         scores_dict["fuzzy_score"],
            "reranker_raw_score":  raw_reranker,
            "reranker_normalized_score": norm_reranker,
            "calibrated_probability": calibrated_prob,
            "calibration_applied": calibration_applied,
            "calibration_method":  calibration_method,
            "calibration_version": calibration_version,
            "final_score":         final_score,
            "pipeline_status":     pipeline_status,
            "no_candidate_reason": no_cand_reason,
            "decision_status":     decision_status,
            "candidate_count":     candidate_count,
            "reason_codes":        reason_codes,
            "human_explanation":   human_exp,
            "retrieval_sources":   channel_counts,
            "candidate_rank":      rank_idx + 1,
            "matched_variant_name": cand.get("variant_name"),
            "variant_type":        cand.get("variant_type"),
            "watchlist_company_name": cand.get("company_name"),
        }
        result["match_results"].append(match_record)

        # ── Alert tablosu (sadece HIGH/MEDIUM) ────────────────────────────────
        if self.scorer.is_alert_worthy(risk_level):
            alert_record = {
                **match_record,
                "company_id":    cand.get("company_id"),
                "risk_level":    risk_level,
                "match_reason":  match_reason,
                "reranker_score": norm_reranker,
            }
            result["alerts"].append(alert_record)

            if risk_level == "HIGH":
                result["high_count"] += 1
            elif risk_level == "MEDIUM":
                result["medium_count"] += 1
        else:
            result["no_match_count"] += 1

    def _update_latency_metrics(self, run_id: str, latency_stats: dict) -> None:
        """Latency percentile değerlerini run_log'a yazar."""
        conn = self.repo.get_connection()
        if not conn:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE {TABLES['run_log']}
                    SET p50_latency_ms = %s,
                        p95_latency_ms = %s,
                        p99_latency_ms = %s
                    WHERE run_id = %s
                """, (
                    latency_stats.get("p50"),
                    latency_stats.get("p95"),
                    latency_stats.get("p99"),
                    run_id
                ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error updating latency metrics: {e}")
            conn.rollback()
        finally:
            self.repo.release_connection(conn)
