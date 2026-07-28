"""
final_scorer.py — AML eşleştirme nihai skor ve risk atama motoru.

Özellikler:
  - Kısa ve genel isimler (token sayısı < 2, uzunluk < 4) için exact match override uygulanmaz; yapay zeka ensemble sonucu korunur.
  - Denetim izi (audit trail) için reason_codes, decision_status ve calibrated_probability alanlarını üretir.
  - Konfigürasyon tabanlı eşik değerleriyle (HIGH, MEDIUM, LOW/NO_MATCH) katmanlı risk sınıflandırması sağlar.
"""

import logging
from typing import Optional
from src.config.db_tables import TABLES
from src.scoring.reason_codes import ReasonCode, build_human_explanation, codes_to_list
from src.utils.text_utils import _AMBIGUOUS_SHORT_NAMES

logger = logging.getLogger(__name__)

# Exact override için minimum güvenli koşullar
_EXACT_OVERRIDE_MIN_TOKEN_COUNT = 2   # En az 2 token (şirket eklerinden arındırılmış öz isim için)
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
            config_version: DB'den çekilecek model/ağırlık versiyonu
            threshold_version: DB'den çekilecek eşik (threshold) versiyonu
            calibration_wrapper: Seçimli ProbabilityCalibration sınıf örneği
        """
        self.repo = repository
        self.config_version = config_version
        self.threshold_version = threshold_version
        self.calibration_wrapper = calibration_wrapper

        # Default fallback değerler (DB'den okuma başarısız olursa)
        self.weights: dict[str, float] = {
            "fuzzy_weight":    0.10,
            "vector_weight":   0.50,
            "acronym_weight":  0.0,
            "rule_weight":     0.0,
            "reranker_weight": 0.40,
        }
        self.thresholds: list[dict] = []
        self._load_config_from_db()

    def _load_config_from_db(self) -> None:
        """Veritabanından ağırlıkları ve eşik değerlerini yükler."""
        if not self.repo:
            return

        conn = self.repo.get_connection()
        if not conn:
            return

        try:
            with conn.cursor() as cur:
                # ── Ağırlıklar ──────────────────────────────────────────────
                cur.execute(f"""
                    SELECT fuzzy_weight, vector_weight, acronym_weight, rule_weight, reranker_weight
                    FROM {TABLES['scoring_weight']}
                    WHERE config_version = %s AND is_active = true
                """, (self.config_version,))
                row = cur.fetchone()
                if row:
                    self.weights["fuzzy_weight"] = float(row[0] if row[0] is not None else 0.0)
                    self.weights["vector_weight"] = float(row[1] if row[1] is not None else 0.0)
                    self.weights["acronym_weight"] = float(row[2] if row[2] is not None else 0.0)
                    self.weights["rule_weight"] = float(row[3] if row[3] is not None else 0.0)
                    self.weights["reranker_weight"] = float(row[4] if row[4] is not None else 0.0)
                else:
                    logger.warning(
                        f"Config version '{self.config_version}' not found in DB. "
                        f"Using default fallback weights."
                    )

                # ── Threshold'lar ───────────────────────────────────────────
                thresholds = []
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
                self.thresholds = thresholds
        except Exception as e:
            logger.error(f"Error loading config from DB: {e}")
        finally:
            self.repo.release_connection(conn)

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
        from src.utils.text_utils import get_normalized_core_name

        # Alias confidence düşükse (algoritmik typo vb.)
        if alias_confidence < 0.6:
            return False, ReasonCode.EXACT_MATCH_REQUIRES_REVIEW

        for text in [query, variant_name]:
            if not text or not text.strip():
                return False, ReasonCode.SHORT_AMBIGUOUS_EXACT_MATCH
            stripped = text.strip().casefold()
            tokens = stripped.split()
            char_len = len(stripped.replace(" ", ""))

            # Çok kısa veya tek token
            if len(tokens) < _EXACT_OVERRIDE_MIN_TOKEN_COUNT:
                return False, ReasonCode.SHORT_AMBIGUOUS_EXACT_MATCH

            if char_len < _EXACT_OVERRIDE_MIN_CHAR_LENGTH:
                return False, ReasonCode.SHORT_AMBIGUOUS_EXACT_MATCH

            # Öz ismin (core name) tek token veya çok kısa olması (ör. Apple Corp -> apple)
            core_name = get_normalized_core_name(stripped).strip().casefold()
            core_tokens = core_name.split()
            if len(core_tokens) < _EXACT_OVERRIDE_MIN_TOKEN_COUNT:
                if len(core_tokens) == 1 and len(core_name) >= 6 and len(tokens) >= 2 and core_name not in _AMBIGUOUS_SHORT_NAMES and stripped not in _AMBIGUOUS_SHORT_NAMES:
                    pass  # Uzun, ayırt edici tek kelime + legal suffix (ör. Indiaforensic Services, Finspire Solutions)
                else:
                    return False, ReasonCode.SHORT_AMBIGUOUS_EXACT_MATCH
            elif len(core_name.replace(" ", "")) < _EXACT_OVERRIDE_MIN_CHAR_LENGTH:
                return False, ReasonCode.SHORT_AMBIGUOUS_EXACT_MATCH

            # Genel/ambiguous kelime (hem tam hem core ad)
            if stripped in _AMBIGUOUS_SHORT_NAMES or core_name in _AMBIGUOUS_SHORT_NAMES:
                return False, ReasonCode.SHORT_AMBIGUOUS_EXACT_MATCH

            # Core name içindeki herhangi bir token tek başına ambiguous listesindeyse ve sadece 1 kelime ise
            if len(core_tokens) == 1 and core_tokens[0] in _AMBIGUOUS_SHORT_NAMES:
                return False, ReasonCode.SHORT_AMBIGUOUS_EXACT_MATCH

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

        # ── Acronym / Abbreviation match override ────────────────────────────
        if scores.get("acronym_score", 0.0) >= 1.0:
            query_str = scores.get("_query_str", "")
            cand_str  = scores.get("_variant_str", "")
            from src.utils.text_utils import get_normalized_core_name
            core_cand = get_normalized_core_name(cand_str).strip().casefold()
            
            if cand_str.strip().casefold() not in _AMBIGUOUS_SHORT_NAMES and core_cand not in _AMBIGUOUS_SHORT_NAMES:
                is_safe, safe_code = self._is_safe_exact_override(query_str, cand_str, alias_confidence)

                if ReasonCode.ACRONYM_MATCH not in reason_codes:
                    reason_codes.append(ReasonCode.ACRONYM_MATCH)

                # Reranker skoru burada önemli: kısa tek-token adlarda (< 6 harf)
                # reranker onayı olmadan HIGH risk vermek hatalı eşleşmelere (false positive) yol açar.
                # Örn: "mehar" → "mehar entertainment" (mehar=düğün başlığı olabilir)
                reranker_ok = scores.get("reranker_score", 0.0) >= 0.35
                cand_compact_len = len(cand_str.replace(" ", ""))
                if is_safe and (cand_compact_len >= 6 or reranker_ok):
                    # Uzun/distinctive adlarda veya reranker onayladıysa yüksek risk
                    final_score = max(final_score, 0.88)
                    match_reason = "ACRONYM_MATCH"
                elif cand_compact_len >= 3:
                    # Kısa tek-token adlarda veya reranker şüpheli ise:
                    # Analist incelemesi (MEDIUM), otomatik HIGH risk yok
                    final_score = max(final_score, 0.65)
                    match_reason = "ACRONYM_MATCH"
                else:
                    # Çok kısa (2 harfli) ambiguous olabilecek kısaltmalarda orta risk ve inceleme kodu
                    final_score = max(final_score, 0.65)
                    if safe_code not in reason_codes:
                        reason_codes.append(safe_code)

        if final_score >= 0.60 and not reason_codes:
            reason_codes.append(ReasonCode.SEMANTIC_MATCH)

        # ── Consonant match override ─────────────────────────────────────────
        if scores.get("consonant_match"):
            query_str = scores.get("_query_str", "")
            cand_str  = scores.get("_variant_str", "")
            is_safe, safe_code = self._is_safe_exact_override(query_str, cand_str, alias_confidence)
            if is_safe:
                final_score = max(final_score, 0.85)
                match_reason = "CONSONANT_ONLY_MATCH"
                reason_codes.append(ReasonCode.CONSONANT_ONLY_MATCH)
            else:
                reason_codes.append(safe_code)

        # ── High fuzzy match boost (yazım hatası, eksik harf, typo desteği) ──
        fuzzy_val = scores.get("fuzzy_score", 0.0)
        if fuzzy_val >= 0.82:
            query_str = scores.get("_query_str", "")
            cand_str  = scores.get("_variant_str", "")
            is_safe, _ = self._is_safe_exact_override(query_str, cand_str, alias_confidence)
            
            if is_safe:
                if fuzzy_val >= 0.88:
                    final_score = max(final_score, min(0.92, fuzzy_val * 0.98))
                else:
                    final_score = max(final_score, fuzzy_val * 0.88)
                match_reason = "HIGH_FUZZY_MATCH"
                if ReasonCode.HIGH_FUZZY_SIMILARITY not in reason_codes:
                    reason_codes.append(ReasonCode.HIGH_FUZZY_SIMILARITY)
            else:
                # Güvenli değil (tek kelimelik veya genel kelime: ör. Apple, Oracle, Amazon)
                # Yalnızca hem sorgu hem aday çoklu kelimeden oluşuyorsa (ör. Amazn Technologies Inc)
                # ve token sayısı benzerse makul bir destek ver
                from src.utils.text_utils import get_normalized_core_name
                q_tokens = query_str.split()
                c_tokens = cand_str.split()
                core_c = get_normalized_core_name(cand_str).strip().casefold()
                core_c_toks = core_c.split()

                cand_eligible = False
                if len(core_c_toks) >= 2:
                    cand_eligible = True
                elif len(core_c_toks) == 1:
                    if core_c_toks[0] in _AMBIGUOUS_SHORT_NAMES:
                        cand_eligible = (len(c_tokens) >= 3)
                    else:
                        cand_eligible = (len(c_tokens) >= 2)

                if cand_eligible and len(q_tokens) >= 2 and abs(len(q_tokens) - len(c_tokens)) <= 1:
                    final_score = max(final_score, fuzzy_val * 0.90)
                    match_reason = "HIGH_FUZZY_MATCH"
                if ReasonCode.HIGH_FUZZY_SIMILARITY not in reason_codes:
                    reason_codes.append(ReasonCode.HIGH_FUZZY_SIMILARITY)

        # ── Leetspeak evasion override ───────────────────────────────────────
        if scores.get("leetspeak_evasion_detected"):
            if ReasonCode.LEETSPEAK_EVASION not in reason_codes:
                reason_codes.append(ReasonCode.LEETSPEAK_EVASION)
            final_score = max(final_score, 0.88)
            match_reason = "LEETSPEAK_EVASION_DETECTED"

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
                # Güvenli değil (Apple, Oracle vb. tek veya genel kelimeler)
                # Doğrudan yüksek risk (override) VERME! Bağlamı (ensemble skoru) değerlendir!
                reason_codes.append(safe_code)
                match_reason = "EXACT_NORMALIZED_MATCH_WITH_CAUTION"
                logger.debug(
                    f"Exact normalized match for short/ambiguous name '{query_str}' — "
                    f"no direct high risk override applied, relying on context ensemble (score: {final_score:.3f})"
                )

        # ── Exact compact match override ────────────────────────────────────────
        if scores.get("exact_compact_match"):
            compact_cand = scores.get("compact_matched_variant", "")
            query_str = scores.get("_query_str", "")
            cand_str  = scores.get("_variant_str", "")
            is_safe, safe_code = self._is_safe_exact_override(query_str, cand_str, alias_confidence)
            min_length = 10
            score_floor = 0.98
            
            if is_safe and len(compact_cand) >= min_length:
                final_score = max(final_score, score_floor)
                match_reason = "EXACT_COMPACT_MATCH"
                reason_codes.append(ReasonCode.EXACT_COMPACT_MATCH)
                logger.debug(f"Exact compact match triggered for '{compact_cand}'. Score overridden to {final_score}")
            else:
                if not is_safe:
                    reason_codes.append(safe_code)
                logger.debug(f"Exact compact match ignored for '{compact_cand}' due to safety check or min_length")

        # ── Substantial missing info (Eksik bilgi fazlaysa analist incelemesi) ────
        if scores.get("substantial_missing_info"):
            if ReasonCode.PARTIAL_MATCH_REQUIRES_REVIEW not in reason_codes:
                reason_codes.append(ReasonCode.PARTIAL_MATCH_REQUIRES_REVIEW)
            match_reason = "PARTIAL_MATCH_REQUIRES_REVIEW"
            # Doğrudan yüksek risk vermek yerine analist incelemesine (orta risk bandı: 0.45 - 0.64) gönder
            if final_score >= 0.65:
                final_score = 0.64
            elif final_score < 0.45 and scores.get("fuzzy_score", 0.0) >= 0.40:
                # Kısmi bilgi yeterliyse en azından analist incelemesi oluşturabilmeli (aday kaybolmamalı)
                final_score = max(final_score, 0.50)

        # ── Reranker Veto (Semantik red → lexical override'ı kısıtla) ──────────
        # Reranker semantik olarak açıkça reddettiyse (< 0.25) VE kesin bir
        # lexical kanıt yoksa (exact normalized / compact / leetspeak evasion),
        # lexical override'ların HIGH risk üretmesine izin verme.
        # Bu; tek-token eşleşme, compact evasion veya kısmi fuzzy'nin yol açtığı
        # false positive HIGH alert'leri engeller.
        reranker_val = scores.get("reranker_score", 0.0)
        _has_hard_lexical_evidence = (
            scores.get("exact_normalized_match") or
            scores.get("exact_compact_match") or
            scores.get("leetspeak_evasion_detected") or
            scores.get("exact_core_match") or          # compact_core eşleşmesi evasion'ı kapsar
            scores.get("acronym_score", 0.0) >= 1.0   # NST/IBM gibi kısaltma eşleşmesi
        )
        if reranker_val < 0.25 and not _has_hard_lexical_evidence:
            if final_score > 0.64:
                logger.debug(
                    f"Reranker veto applied: reranker={reranker_val:.4f} < 0.25, "
                    f"capping final_score {final_score:.4f} → 0.64 (MEDIUM cap, no HIGH allowed)"
                )
                final_score = 0.64
                if ReasonCode.RERANKER_REJECTED not in reason_codes:
                    reason_codes.append(ReasonCode.RERANKER_REJECTED)

        if not reason_codes:
            reason_codes.append(ReasonCode.LOW_CONFIDENCE)

        return float(min(max(final_score, 0.0), 1.0)), match_reason, codes_to_list(reason_codes)

    def assign_risk_level(self, final_score: float) -> str:
        """
        Final skora göre risk seviyesi atar.

        Öncelik sırası:
          1. DB threshold tablosundaki değerler
          2. Fallback: HIGH >= 0.70, MEDIUM >= 0.60, NO_MATCH < 0.60

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
        if final_score >= 0.65:
            return "HIGH"
        elif final_score >= 0.45:
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
