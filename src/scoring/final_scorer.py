"""
final_scorer.py — AML eşleştirme nihai skor hesaplayıcı.

Değişiklikler (v3):
  - Exact match override artık koşulsuz uygulanmıyor.
    Kısa/genel isimler (token sayısı < 2, uzunluk < 4) için
    final_score 1.0'a çıkarılmaz; ensemble sonucu korunur.
  - reason_codes listesi üretiliyor.
  - calibrated_probability alanı ayrı tutulıyor.
  - HIGH >= 0.70, MEDIUM >= 0.62, NO_MATCH < 0.62 sınırları
    assign_risk_level içinde net biçimde uygulanıyor.
  - decision_status alanı üretiliyor.
"""

import logging
from typing import Optional
from src.config.db_tables import TABLES
from src.scoring.reason_codes import ReasonCode, build_human_explanation, codes_to_list

logger = logging.getLogger(__name__)

# Kısa veya genel isimlere exact match override UYGULANMAZ
_AMBIGUOUS_SHORT_NAMES: set[str] = {
    "abc", "star", "global", "trust", "united", "first", "best",
    "prime", "apex", "nova", "alpha", "beta", "delta", "sigma",
    "omega", "ace", "pro", "max", "tech", "plus", "net",
}

# Exact override için minimum güvenli koşullar
_EXACT_OVERRIDE_MIN_TOKEN_COUNT = 2   # En az 2 token
_EXACT_OVERRIDE_MIN_CHAR_LENGTH = 5   # En az 5 karakter (suffix dahil)


class FinalScorer:
    """
    Adayları puanlayan ve nihai AML kararını veren sınıf.

    Weights ve threshold'lar DB'den yüklenir.
    """

    def __init__(
        self,
        repository,
        config_version: str = "scoring_v2_reranker",
        threshold_version: str = "threshold_v2_reranker",
        calibration_wrapper=None,
    ):
        """
        Args:
            repository: AMLRepository instance
            config_version: Kullanılacak scoring weight versiyonu
            threshold_version: Kullanılacak threshold versiyonu
            calibration_wrapper: CalibrationWrapper instance (opsiyonel)
        """
        self.repo = repository
        self.config_version = config_version
        self.threshold_version = threshold_version
        self.calibration = calibration_wrapper

        self.weights = self._load_weights()
        self.thresholds = self._load_thresholds()

    def _load_weights(self) -> dict:
        """DB'den scoring weight'leri yükler. Hata durumunda default kullanır."""
        conn = self.repo.get_connection()
        weights = {
            "fuzzy_weight":    0.0,
            "vector_weight":   0.30,
            "acronym_weight":  0.0,
            "rule_weight":     0.0,
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
                        "fuzzy_weight":    float(res[0]),
                        "vector_weight":   float(res[1]),
                        "acronym_weight":  float(res[2]),
                        "rule_weight":     float(res[3]),
                        "reranker_weight": float(res[4])
                    }
        except Exception as e:
            logger.error(f"Error loading weights from DB: {e}")
        finally:
            self.repo.release_connection(conn)
        return weights

    def _load_thresholds(self) -> list:
        """DB'den threshold değerlerini yükler."""
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
                    ORDER BY min_score ASC
                """, (self.threshold_version,))
                for row in cur.fetchall():
                    thresholds.append({
                        "risk_level": row[0],
                        "min_score":  float(row[1]),
                        "max_score":  float(row[2])
                    })
        except Exception as e:
            logger.error(f"Error loading thresholds from DB: {e}")
        finally:
            self.repo.release_connection(conn)
        return thresholds

    # ── Exact match güvenlik kontrolü ─────────────────────────────────────────

    def _is_safe_exact_override(
        self,
        query: str,
        variant_name: str,
        alias_confidence: float = 1.0
    ) -> tuple[bool, ReasonCode]:
        """
        Exact match override'ın güvenli olup olmadığını belirler.

        Override güvenli DEĞİLSE:
          - Kısa/tek token isimler (< 2 token veya < 5 karakter)
          - Genel/ambiguous kelimeler (ABC, STAR, GLOBAL vb.)
          - Alias confidence düşükse (algoritmik üretilmiş)

        Returns:
            (is_safe, reason_code)
        """
        q_stripped = query.strip().casefold()
        tokens = q_stripped.split()
        char_len = len(q_stripped.replace(" ", ""))

        # Çok kısa veya tek token
        if len(tokens) < _EXACT_OVERRIDE_MIN_TOKEN_COUNT:
            return False, ReasonCode.SHORT_AMBIGUOUS_EXACT_MATCH

        if char_len < _EXACT_OVERRIDE_MIN_CHAR_LENGTH:
            return False, ReasonCode.SHORT_AMBIGUOUS_EXACT_MATCH

        # Genel/ambiguous kelime
        if q_stripped in _AMBIGUOUS_SHORT_NAMES:
            return False, ReasonCode.SHORT_AMBIGUOUS_EXACT_MATCH

        # Alias confidence düşükse (algoritmik typo vb.)
        if alias_confidence < 0.6:
            return False, ReasonCode.EXACT_MATCH_REQUIRES_REVIEW

        return True, ReasonCode.EXACT_CORE_MATCH

    def calculate_final_score(
        self,
        scores: dict,
        alias_confidence: float = 1.0
    ) -> tuple[float, str, list[str]]:
        """
        Nihai skoru hesaplar ve reason code'ları üretir.

        Args:
            scores: Skor dict'i. Zorunlu anahtarlar:
                    fuzzy_score, vector_score, acronym_score, rule_score, reranker_score,
                    exact_normalized_match, exact_core_match, legal_suffix_only_difference,
                    query_token_count
            alias_confidence: Eşleşen alias'ın güven seviyesi (0.0-1.0)

        Returns:
            (final_score, match_reason, reason_codes_list)
        """
        query_token_count = scores.get("query_token_count", 3)
        reason_codes: list[ReasonCode] = []

        # ── Ağırlıklar ──────────────────────────────────────────────────────
        w_fuzzy    = self.weights.get("fuzzy_weight",    0.0)
        w_vector   = self.weights.get("vector_weight",   0.30)
        w_acronym  = self.weights.get("acronym_weight",  0.0)
        w_rule     = self.weights.get("rule_weight",     0.0)
        w_reranker = self.weights.get("reranker_weight", 0.70)

        # Dinamik ağırlık: kısa sorgular için lexical ağırlığı artır
        if query_token_count <= 2:
            w_vector   = 0.10
            w_reranker = 0.40
            w_fuzzy    = 0.30
            w_rule     = 0.20

        total_weight = w_fuzzy + w_vector + w_acronym + w_rule + w_reranker
        weight_factor = 1.0 / total_weight if total_weight > 0 else 1.0

        # ── Weighted ensemble ────────────────────────────────────────────────
        fuzzy_contrib    = scores.get("fuzzy_score",    0.0) * w_fuzzy    * weight_factor
        vector_contrib   = scores.get("vector_score",   0.0) * w_vector   * weight_factor
        acronym_contrib  = scores.get("acronym_score",  0.0) * w_acronym  * weight_factor
        rule_contrib     = scores.get("rule_score",     0.0) * w_rule     * weight_factor
        reranker_contrib = scores.get("reranker_score", 0.0) * w_reranker * weight_factor

        logger.debug(
            f"Score contributions — Fuzzy: {fuzzy_contrib:.4f}, Vector: {vector_contrib:.4f}, "
            f"Acronym: {acronym_contrib:.4f}, Rule: {rule_contrib:.4f}, "
            f"Reranker: {reranker_contrib:.4f}"
        )

        weighted_score = fuzzy_contrib + vector_contrib + acronym_contrib + rule_contrib + reranker_contrib
        final_score = float(min(max(weighted_score, 0.0), 1.0))

        # ── Sinyaller için reason code üretimi ──────────────────────────────
        match_reason = "LOW_CONFIDENCE"

        if scores.get("reranker_score", 0.0) > 0.75:
            reason_codes.append(ReasonCode.RERANKER_CONFIRMED)
            match_reason = "RERANKER_CONFIRMED"
        elif scores.get("reranker_score", 0.0) < 0.30 and final_score >= 0.50:
            reason_codes.append(ReasonCode.RERANKER_REJECTED)

        if scores.get("vector_score", 0.0) > 0.80:
            reason_codes.append(ReasonCode.HIGH_VECTOR_SIMILARITY)

        if scores.get("fuzzy_score", 0.0) > 0.85:
            reason_codes.append(ReasonCode.HIGH_FUZZY_SIMILARITY)
            match_reason = "HIGH_FUZZY_MATCH"

        if scores.get("acronym_score", 0.0) >= 1.0:
            reason_codes.append(ReasonCode.ACRONYM_MATCH)

        if final_score >= 0.60 and not reason_codes:
            reason_codes.append(ReasonCode.SEMANTIC_MATCH)

        # ── Consonant match override ─────────────────────────────────────────
        if scores.get("consonant_match"):
            final_score = max(final_score, 0.85)
            match_reason = "CONSONANT_ONLY_MATCH"
            reason_codes.append(ReasonCode.CONSONANT_ONLY_MATCH)

        # ── Legal suffix only difference override ────────────────────────────
        if scores.get("legal_suffix_only_difference"):
            query_str  = scores.get("_query_str", "")
            cand_str   = scores.get("_variant_str", "")
            is_safe, safe_code = self._is_safe_exact_override(query_str, cand_str, alias_confidence)

            if is_safe:
                final_score  = max(final_score, 0.92)
                match_reason = "LEGAL_SUFFIX_ONLY_DIFFERENCE"
                reason_codes.append(ReasonCode.LEGAL_SUFFIX_ONLY_DIFFERENCE)
            else:
                reason_codes.append(safe_code)

        # ── Exact core match override ────────────────────────────────────────
        if scores.get("exact_core_match"):
            query_str = scores.get("_query_str", "")
            cand_str  = scores.get("_variant_str", "")
            is_safe, safe_code = self._is_safe_exact_override(query_str, cand_str, alias_confidence)

            if is_safe:
                final_score  = max(final_score, 0.95)
                match_reason = "EXACT_CORE_MATCH"
                reason_codes.append(ReasonCode.EXACT_CORE_MATCH)
            else:
                # Güvenli değil — sadece reason code ekle, skor override etme
                reason_codes.append(safe_code)
                logger.debug(
                    f"Exact core match override skipped for '{query_str}' "
                    f"(reason: {safe_code.value})"
                )

        # ── Exact normalized match override (en güçlü) ──────────────────────
        if scores.get("exact_normalized_match"):
            query_str = scores.get("_query_str", "")
            cand_str  = scores.get("_variant_str", "")
            is_safe, safe_code = self._is_safe_exact_override(query_str, cand_str, alias_confidence)

            if is_safe:
                final_score  = 1.0
                match_reason = "EXACT_NORMALIZED_MATCH"
                reason_codes.append(ReasonCode.EXACT_OFFICIAL_NAME)
            else:
                # Güvenli değil — yüksek skor ver ama 1.0 yapmaz
                final_score  = max(final_score, 0.85)
                reason_codes.append(safe_code)
                match_reason = "EXACT_NORMALIZED_MATCH_WITH_CAUTION"
                logger.warning(
                    f"Exact normalized match for short/ambiguous name '{query_str}' — "
                    f"score capped at {final_score:.3f}"
                )

        # ── Exact compact match override ────────────────────────────────────────
        if scores.get("exact_compact_match"):
            compact_cand = scores.get("compact_matched_variant", "")
            # TODO: ConfigLoader kullanılarak dinamik yapılabilir, şu an default değerleri kullanıyoruz
            min_length = 10
            score_floor = 0.98
            
            if len(compact_cand) >= min_length:
                final_score = max(final_score, score_floor)
                match_reason = "EXACT_COMPACT_MATCH"
                reason_codes.append(ReasonCode.EXACT_COMPACT_MATCH)
                logger.info(f"Exact compact match triggered for '{compact_cand}'. Score overridden to {final_score}")
            else:
                logger.debug(f"Exact compact match ignored for '{compact_cand}' due to min_length ({len(compact_cand)} < {min_length})")

        if not reason_codes:
            reason_codes.append(ReasonCode.LOW_CONFIDENCE)

        return float(min(max(final_score, 0.0), 1.0)), match_reason, codes_to_list(reason_codes)

    def assign_risk_level(self, final_score: float) -> str:
        """
        Final skora göre risk seviyesi atar.

        Öncelik sırası:
          1. DB threshold tablosundaki değerler
          2. Fallback: HIGH >= 0.70, MEDIUM >= 0.62, NO_MATCH < 0.62

        Args:
            final_score: 0.0 - 1.0 arasında final skor

        Returns:
            'HIGH', 'MEDIUM', 'LOW', veya 'NO_MATCH'
        """
        if self.thresholds:
            for i, t in enumerate(self.thresholds):
                is_last = (i == len(self.thresholds) - 1)
                if is_last:
                    if t["min_score"] <= final_score <= t["max_score"]:
                        return t["risk_level"]
                else:
                    if t["min_score"] <= final_score < t["max_score"]:
                        return t["risk_level"]

        # DB'den threshold yüklenemezse veya aralık dışındaysa fallback
        if final_score >= 0.70:
            return "HIGH"
        elif final_score >= 0.62:
            return "MEDIUM"
        else:
            return "NO_MATCH"

    def assign_decision_status(self, risk_level: str) -> str:
        """
        Risk seviyesini iş kararı (decision_status) alanına dönüştürür.

        Args:
            risk_level: 'HIGH', 'MEDIUM', 'LOW', 'NO_MATCH'

        Returns:
            decision_status string
        """
        mapping = {
            "HIGH":     "HIGH_ALERT",
            "MEDIUM":   "MEDIUM_ALERT",
            "LOW":      "MATCH_BELOW_THRESHOLD",
            "NO_MATCH": "NO_MATCH",
        }
        return mapping.get(risk_level, "NO_MATCH")

    def is_alert_worthy(self, risk_level: str) -> bool:
        """
        Bu risk seviyesi için alert tablosuna kayıt oluşturulmalı mı?

        Yalnızca HIGH ve MEDIUM alertler alert tablosuna yazılır.
        LOW ve NO_MATCH sadece match_result'a yazılır.

        Args:
            risk_level: Risk seviyesi string

        Returns:
            True ise alert tablosuna yaz
        """
        return risk_level in ("HIGH", "MEDIUM")
