"""
entity_extractor.py — Çok katmanlı entity extraction orchestrator.

NER modeli başarısız olduğunda ya da yetersiz sonuç ürettiğinde
fallback katmanları devreye girer. Bu sayede NER tek karar
noktası olmaktan çıkar.

Katman sırası:
  1. NER model (mevcut NERExtractor)
  2. Regex / rule-based extraction
  3. Candidate-supported span extraction
  4. Exact variant fallback
  5. Full text fallback (tüm normalize metin)

Her sonuç için extraction metadata üretilir:
  - extracted_entity
  - entity_type: PERSON | ORGANIZATION | VESSEL | AIRCRAFT | LOCATION | UNKNOWN
  - extraction_method: NER_MODEL | RULE_BASED | CANDIDATE_SUPPORTED |
                        FALLBACK_MATCHED_VARIANT | FULL_TEXT_FALLBACK | ENTITY_NOT_FOUND
  - extraction_confidence: 0.0 - 1.0
  - extraction_start / extraction_end: karakter offset (NER varsa)
  - entity_extraction_status: EXTRACTED | FALLBACK | NOT_FOUND
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Regex pattern'ları ────────────────────────────────────────────────────────
# EFT açıklamalarında şirket adı sinyali verebilecek pattern'lar
_COMPANY_SUFFIX_PATTERN = re.compile(
    r"\b(ltd|limited|llc|inc|incorporated|corp|corporation|co|company|"
    r"plc|llp|pvt|a\.s\.|a\.ş\.|anonim|sirketi|holding|group|"
    r"international|intl|trading|import|export|logistics|energy)\b",
    re.IGNORECASE
)

# EFT içindeki kesin anlamlı bloklar (büyük harfli kısımlar şirket adı olabilir)
_CAPITALIZED_BLOCK_PATTERN = re.compile(
    r"\b([A-Z][A-Z\s&\-\.]{2,50}(?:LTD|LLC|INC|CORP|CO|PLC|GROUP|HOLDING)?)\b"
)

# IBAN, hesap numaraları vb. PII – bunlar entity değil, maskelenmeli
_IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b")
_ACCOUNT_PATTERN = re.compile(r"\b\d{8,26}\b")


@dataclass
class ExtractionResult:
    """Tek bir entity extraction sonucu."""

    extracted_entity: Optional[str] = None
    entity_type: str = "UNKNOWN"
    extraction_method: str = "ENTITY_NOT_FOUND"
    extraction_confidence: float = 0.0
    extraction_start: Optional[int] = None
    extraction_end: Optional[int] = None
    entity_extraction_status: str = "NOT_FOUND"


class EntityExtractor:
    """
    Çok katmanlı entity extraction orchestrator.

    NER modeli opsiyonel; kurulu değilse veya başarısız olursa
    sonraki katmanlar devreye girer.
    """

    def __init__(self, ner_extractor=None, config: Optional[dict] = None):
        """
        Args:
            ner_extractor: NERExtractor instance (opsiyonel)
            config: Extraction yapılandırması
        """
        self.ner_extractor = ner_extractor
        self.config = config or {}
        self.min_entity_length = self.config.get("min_entity_length", 3)
        self.max_entity_length = self.config.get("max_entity_length", 100)

    # ── Katman 1: NER ─────────────────────────────────────────────────────────

    def _extract_via_ner(self, text: str) -> Optional[ExtractionResult]:
        """
        NER modeli ile entity çıkarır.
        NERExtractor mevcut değilse None döner.
        """
        if not self.ner_extractor:
            return None

        try:
            entity = self.ner_extractor.extract_entity(text)
            if entity and len(entity.strip()) >= self.min_entity_length:
                return ExtractionResult(
                    extracted_entity=entity.strip(),
                    entity_type="ORGANIZATION",   # Mevcut model ORG/PER çıkarıyor
                    extraction_method="NER_MODEL",
                    extraction_confidence=0.90,
                    entity_extraction_status="EXTRACTED"
                )
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")

        return None

    # ── Katman 2: Rule-based ──────────────────────────────────────────────────

    def _extract_via_rules(self, text: str) -> Optional[ExtractionResult]:
        """
        Regex / rule-based entity çıkarma.
        Büyük harfli bloklar ve şirket suffix pattern'larına bakılır.
        """
        # Büyük harfli + suffix içeren blok ara
        match = _CAPITALIZED_BLOCK_PATTERN.search(text)
        if match:
            candidate = match.group(0).strip()
            if (self.min_entity_length <= len(candidate) <= self.max_entity_length
                    and not _IBAN_PATTERN.match(candidate)
                    and not _ACCOUNT_PATTERN.match(candidate)):
                return ExtractionResult(
                    extracted_entity=candidate,
                    entity_type="ORGANIZATION",
                    extraction_method="RULE_BASED",
                    extraction_confidence=0.65,
                    extraction_start=match.start(),
                    extraction_end=match.end(),
                    entity_extraction_status="EXTRACTED"
                )

        # Şirket suffix içeren kelime grubu ara
        suffix_match = _COMPANY_SUFFIX_PATTERN.search(text)
        if suffix_match:
            start_pos = max(0, suffix_match.start() - 60)
            snippet = text[start_pos:suffix_match.end()]
            # En az 2 kelimeden oluşan parça al
            words = [w for w in snippet.split() if len(w) > 2]
            if len(words) >= 2:
                entity = " ".join(words[-4:])  # Son 4 kelimeyi al
                return ExtractionResult(
                    extracted_entity=entity.strip(),
                    entity_type="ORGANIZATION",
                    extraction_method="RULE_BASED",
                    extraction_confidence=0.50,
                    entity_extraction_status="EXTRACTED"
                )

        return None

    # ── Katman 3: Candidate-supported ────────────────────────────────────────

    def _extract_via_candidates(
        self, text: str, candidates: list
    ) -> Optional[ExtractionResult]:
        """
        Aday listesindeki variant isimlerini EFT metninde arar.
        En uzun eşleşen ismi entity olarak döndürür.

        Args:
            text: EFT açıklaması (normalize edilmiş)
            candidates: retrieval'dan gelen aday listesi

        Returns:
            En uzun eşleşen variant adı
        """
        if not candidates:
            return None

        text_lower = text.lower()
        text_tokens = set(text_lower.split())
        best_match: Optional[tuple[str, int]] = None  # (variant_name, length)
        best_acronym_match: Optional[tuple[str, int]] = None  # (variant_name, length)

        from src.utils.alias_utils import generate_acronym, generate_abbreviated_aliases
        from src.utils.text_utils import normalize_leetspeak

        leet_text_lower = normalize_leetspeak(text_lower)
        leet_text_tokens = set(leet_text_lower.split())

        for cand in candidates:
            vname = cand.get("variant_name", "")
            if not vname or len(vname.strip()) < 2:
                continue
            vname_lower = vname.lower()

            # 1. Doğrudan variant adının veya leetspeak çevirisinin eşleşmesi
            in_text = vname_lower in text_lower
            in_leet = vname_lower in leet_text_lower and leet_text_lower != text_lower
            if in_text or in_leet:
                if len(vname_lower.split()) == 1 and len(vname_lower) <= 3 and vname_lower not in text_tokens and vname_lower not in leet_text_tokens:
                    pass  # 2-3 harflik kısa kelimelerde kelime sınırı (word boundary) kontrolü
                else:
                    if best_match is None or len(vname) > best_match[1]:
                        best_match = (vname, len(vname))

            # 2. Sistem tarafından üretilen baş harf kısaltması (acronym) veya kısaltmalı alias eşleşmesi
            if best_match is None:
                orig_name = cand.get("original_company_name", vname)
                for name_to_check in [vname, orig_name]:
                    if not name_to_check:
                        continue
                    acr = generate_acronym(name_to_check)
                    if acr and len(acr) >= 2 and (acr in text_tokens or acr in leet_text_tokens):
                        if best_acronym_match is None or len(name_to_check) > best_acronym_match[1]:
                            best_acronym_match = (name_to_check, len(name_to_check))
                    for abbr in generate_abbreviated_aliases(name_to_check, max_alias_count=10):
                        if abbr and len(abbr) >= 2 and (abbr in text_lower or abbr in leet_text_lower):
                            if len(abbr.split()) == 1 and abbr not in text_tokens and abbr not in leet_text_tokens:
                                continue
                            if best_acronym_match is None or len(name_to_check) > best_acronym_match[1]:
                                best_acronym_match = (name_to_check, len(name_to_check))

        if best_match:
            return ExtractionResult(
                extracted_entity=best_match[0],
                entity_type="ORGANIZATION",
                extraction_method="CANDIDATE_SUPPORTED",
                extraction_confidence=0.75,
                entity_extraction_status="EXTRACTED"
            )

        if best_acronym_match:
            return ExtractionResult(
                extracted_entity=best_acronym_match[0],
                entity_type="ORGANIZATION",
                extraction_method="ACRONYM_SUPPORTED",
                extraction_confidence=0.75,
                entity_extraction_status="EXTRACTED"
            )

        # Exact substring eşleşmediyse, adayın kendisiyle (veya öz adıyla) fuzzy ve compact benzerlik kontrolü yap (yazım hatası / typo / bitişik-ayrı yazım desteği)
        import difflib
        from src.utils.text_utils import get_normalized_core_name, get_compact_core_name
        text_core = get_normalized_core_name(text_lower)
        text_compact = get_compact_core_name(text_lower)
        best_fuzzy_cand: Optional[tuple[str, float]] = None

        for cand in candidates:
            vname = cand.get("variant_name", "")
            if not vname or len(vname) < self.min_entity_length:
                continue
            vname_core = get_normalized_core_name(vname.lower())
            vname_compact = get_compact_core_name(vname.lower())
            if not vname_core and not vname_compact:
                continue
            
            trgm = cand.get("trgm_score", 0.0)
            ratio = difflib.SequenceMatcher(None, text_core, vname_core).ratio() if vname_core else 0.0
            
            # Compact core benzerliği (bitişik veya ayrı yazılan kelimeler için)
            compact_sim = 0.0
            if vname_compact and len(vname_compact) >= 4:
                if vname_compact in text_compact:
                    compact_sim = 1.0
                elif text_compact:
                    compact_sim = difflib.SequenceMatcher(None, text_compact, vname_compact).ratio()

            # Token bazlı benzerlik (kelimeler arası en yüksek eşleşme)
            t_sim = 0.0
            c_words = [w for w in vname_core.split() if len(w) > 2]
            q_words = [w for w in text_core.split() if len(w) > 2]
            if c_words and q_words:
                t_sim = sum(max(difflib.SequenceMatcher(None, qw, cw).ratio() for qw in q_words) for cw in c_words) / len(c_words)
                
            sim = max(trgm, ratio, t_sim, compact_sim)
            if sim >= 0.80 and (best_fuzzy_cand is None or sim > best_fuzzy_cand[1]):
                best_fuzzy_cand = (vname, sim)

        if best_fuzzy_cand:
            return ExtractionResult(
                extracted_entity=best_fuzzy_cand[0],
                entity_type="ORGANIZATION",
                extraction_method="FUZZY_CANDIDATE_MATCH",
                extraction_confidence=0.70,
                entity_extraction_status="EXTRACTED"
            )

        return None

    # ── Katman 4: Exact variant fallback ─────────────────────────────────────

    def _extract_via_variant_fallback(
        self, text: str, candidates: list
    ) -> Optional[ExtractionResult]:
        """
        Kandidatlardaki en yüksek skorlu varyant adını fallback entity olarak kullanır.
        Katman 3'ten farklı olarak EFT metninde aramaz — direkt en iyi adayı kullanır.
        """
        if not candidates:
            return None

        # En yüksek candidate_score'a sahip olanı seç
        best = max(candidates, key=lambda c: c.get("candidate_score", 0.0), default=None)
        if best and best.get("variant_name"):
            return ExtractionResult(
                extracted_entity=best["variant_name"],
                entity_type="ORGANIZATION",
                extraction_method="FALLBACK_MATCHED_VARIANT",
                extraction_confidence=0.40,
                entity_extraction_status="FALLBACK"
            )

        return None

    # ── Katman 5: Full text fallback ──────────────────────────────────────────

    def _extract_full_text_fallback(self, text: str) -> ExtractionResult:
        """
        Hiçbir katman sonuç üretemediğinde normalize edilmiş metnin
        ilk 60 karakterini entity olarak döndürür.
        extraction_confidence çok düşük tutulur.
        """
        snippet = text.strip()[:60].strip() if text else ""
        if snippet:
            return ExtractionResult(
                extracted_entity=snippet,
                entity_type="UNKNOWN",
                extraction_method="FULL_TEXT_FALLBACK",
                extraction_confidence=0.10,
                entity_extraction_status="FALLBACK"
            )

        return ExtractionResult(
            extracted_entity=None,
            entity_type="UNKNOWN",
            extraction_method="ENTITY_NOT_FOUND",
            extraction_confidence=0.0,
            entity_extraction_status="NOT_FOUND"
        )

    # ── Ana orchestrator ──────────────────────────────────────────────────────

    def extract(
        self,
        text: str,
        candidates: Optional[list] = None,
        use_full_text_fallback: bool = False
    ) -> ExtractionResult:
        """
        Çok katmanlı extraction pipeline'ını çalıştırır.

        Katmanları sırayla dener; ilk başarılı sonucu döndürür.

        Args:
            text: EFT açıklaması (ham veya normalize edilmiş)
            candidates: Retrieval'dan gelen aday listesi (opsiyonel)
            use_full_text_fallback: True ise Katman 5 de denenecek

        Returns:
            ExtractionResult — en iyi extraction sonucu
        """
        if not text or not text.strip():
            return ExtractionResult(
                extraction_method="ENTITY_NOT_FOUND",
                entity_extraction_status="NOT_FOUND"
            )

        from src.utils.text_utils import clean_spaced_characters
        text = clean_spaced_characters(text)

        # Katman 1: NER
        result = self._extract_via_ner(text)
        if result:
            logger.debug(f"Entity extracted via NER: {result.extracted_entity}")
            return result

        # Katman 2: Candidate-supported (öncelikli)
        if candidates:
            result = self._extract_via_candidates(text, candidates)
            if result:
                logger.debug(f"Entity extracted via candidates: {result.extracted_entity}")
                return result

        # Katman 3: Rule-based heuristic fallback
        result = self._extract_via_rules(text)
        if result:
            logger.debug(f"Entity extracted via rules: {result.extracted_entity}")
            return result

        # Katman 4: Variant fallback
        if candidates:
            result = self._extract_via_variant_fallback(text, candidates)
            if result:
                logger.debug(f"Entity via variant fallback: {result.extracted_entity}")
                return result

        # Katman 5: Full text fallback (opsiyonel)
        if use_full_text_fallback:
            return self._extract_full_text_fallback(text)

        return ExtractionResult(
            extraction_method="ENTITY_NOT_FOUND",
            entity_extraction_status="NOT_FOUND"
        )

    def batch_extract(
        self,
        texts: list[str],
        candidates_per_row: Optional[dict] = None,
        use_full_text_fallback: bool = False
    ) -> list[ExtractionResult]:
        """
        Toplu extraction. NER için GPU batching kullanır.
        NER'in bulamadığı satırlar için fallback katmanları sırayla denenir.

        Args:
            texts: EFT metin listesi
            candidates_per_row: {row_id: [candidates]} haritası (opsiyonel)
            use_full_text_fallback: Katman 5 aktif edilsin mi

        Returns:
            ExtractionResult listesi (texts ile aynı uzunlukta)
        """
        if not texts:
            return []

        from src.utils.text_utils import clean_spaced_characters
        texts = [clean_spaced_characters(t) if t else "" for t in texts]

        results = [None] * len(texts)

        # 1. Katman 1 (NER): GPU Batch Inference
        ner_entities = [None] * len(texts)
        if self.ner_extractor:
            try:
                ner_entities = self.ner_extractor.batch_extract_entities(texts)
            except Exception as e:
                logger.error(f"Batched NER failed: {e}")

        # 2. Sonuçları birleştir & Fallback katmanları
        for i, text in enumerate(texts):
            candidates = candidates_per_row.get(str(i), []) if candidates_per_row else None
            
            if not text or not text.strip():
                results[i] = ExtractionResult(
                    extraction_method="ENTITY_NOT_FOUND",
                    entity_extraction_status="NOT_FOUND"
                )
                continue

            # Check if batched NER found something
            ner_val = ner_entities[i]
            if ner_val and len(ner_val.strip()) >= self.min_entity_length:
                results[i] = ExtractionResult(
                    extracted_entity=ner_val.strip(),
                    entity_type="ORGANIZATION",
                    extraction_method="NER_MODEL",
                    extraction_confidence=0.90,
                    entity_extraction_status="EXTRACTED"
                )
                continue

            # Fallbacks: Candidates, Rule-based, Variant fallback, Full text
            result = None
            if candidates:
                result = self._extract_via_candidates(text, candidates)
            if not result:
                result = self._extract_via_rules(text)
            if not result and candidates:
                result = self._extract_via_variant_fallback(text, candidates)
            if not result and use_full_text_fallback:
                result = self._extract_full_text_fallback(text)
                
            if not result:
                result = ExtractionResult(
                    extraction_method="ENTITY_NOT_FOUND",
                    entity_extraction_status="NOT_FOUND"
                )
            results[i] = result

        return results
