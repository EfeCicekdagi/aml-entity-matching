"""
reason_codes.py — AML eşleştirme karar gerekçe kodları.

Her eşleştirme sonucu için insan tarafından okunabilir
açıklama üretmek amacıyla kullanılır. reason_codes JSONB
alanında list[str] olarak saklanır.

Kullanım:
    from src.scoring.reason_codes import ReasonCode, build_human_explanation

    codes = [ReasonCode.EXACT_OFFICIAL_NAME, ReasonCode.HIGH_VECTOR_SIMILARITY]
    explanation = build_human_explanation(
        entity="Microsoft Corp",
        matched_name="Microsoft Corporation",
        codes=codes,
        final_score=0.97
    )
"""

from enum import Enum
from typing import Optional


class ReasonCode(str, Enum):
    """
    AML eşleştirme karar gerekçe kodları.

    Kategoriler:
      - EXACT_*: Tam eşleşme türleri
      - HIGH_*: Yüksek skor sinyalleri
      - SHORT_*/AMBIGUOUS_*: Düşük güven sinyalleri
      - NER_*: Entity extraction ile ilgili
      - COUNTRY_*/ENTITY_TYPE_*/DATE_*: Yardımcı alan çelişkileri
      - CALIBRATION_*: Kalibrasyon durumu
      - NO_*: Aday yok durumu
    """

    EXACT_OFFICIAL_NAME        = "EXACT_OFFICIAL_NAME"
    EXACT_OFFICIAL_ALIAS       = "EXACT_OFFICIAL_ALIAS"
    EXACT_CORE_MATCH           = "EXACT_CORE_MATCH"
    LEGAL_SUFFIX_ONLY_DIFFERENCE = "LEGAL_SUFFIX_ONLY_DIFFERENCE"
    EXACT_MATCH_REQUIRES_REVIEW  = "EXACT_MATCH_REQUIRES_REVIEW"
    EXACT_COMPACT_MATCH        = "EXACT_COMPACT_MATCH"

    # ── Tam eşleşme uyarıları ─────────────────────────────────────────────────
    SHORT_AMBIGUOUS_EXACT_MATCH = "SHORT_AMBIGUOUS_EXACT_MATCH"
    DUPLICATE_ALIAS_MATCH       = "DUPLICATE_ALIAS_MATCH"
    PARTIAL_MATCH_REQUIRES_REVIEW = "PARTIAL_MATCH_REQUIRES_REVIEW"

    # ── Yüksek skor sinyalleri ────────────────────────────────────────────────
    HIGH_TRIGRAM_SIMILARITY    = "HIGH_TRIGRAM_SIMILARITY"
    HIGH_VECTOR_SIMILARITY     = "HIGH_VECTOR_SIMILARITY"
    HIGH_FUZZY_SIMILARITY      = "HIGH_FUZZY_SIMILARITY"
    RERANKER_CONFIRMED         = "RERANKER_CONFIRMED"
    RERANKER_REJECTED          = "RERANKER_REJECTED"
    SEMANTIC_MATCH             = "SEMANTIC_MATCH"
    CONSONANT_ONLY_MATCH       = "CONSONANT_ONLY_MATCH"

    # ── Alias / variant türü ──────────────────────────────────────────────────
    OFFICIAL_ALIAS_MATCH       = "OFFICIAL_ALIAS_MATCH"
    TRANSLITERATION_MATCH      = "TRANSLITERATION_MATCH"
    FORMER_NAME_MATCH          = "FORMER_NAME_MATCH"
    ABBREVIATION_MATCH         = "ABBREVIATION_MATCH"
    ACRONYM_MATCH              = "ACRONYM_MATCH"
    LEETSPEAK_EVASION          = "LEETSPEAK_EVASION"

    # ── NER ve entity extraction ──────────────────────────────────────────────
    NER_FALLBACK_USED          = "NER_FALLBACK_USED"
    NO_ENTITY_EXTRACTED        = "NO_ENTITY_EXTRACTED"
    RULE_BASED_EXTRACTION      = "RULE_BASED_EXTRACTION"
    CANDIDATE_SUPPORTED_EXTRACTION = "CANDIDATE_SUPPORTED_EXTRACTION"
    FULL_TEXT_FALLBACK_USED    = "FULL_TEXT_FALLBACK_USED"

    # ── Yardımcı alan çelişkileri ─────────────────────────────────────────────
    COUNTRY_CONFLICT           = "COUNTRY_CONFLICT"
    DATE_OF_BIRTH_CONFLICT     = "DATE_OF_BIRTH_CONFLICT"
    IDENTIFIER_EXACT_MATCH     = "IDENTIFIER_EXACT_MATCH"
    ENTITY_TYPE_CONFLICT       = "ENTITY_TYPE_CONFLICT"
    ADDRESS_SUPPORTS_MATCH     = "ADDRESS_SUPPORTS_MATCH"

    # ── Kalibrasyon durumu ────────────────────────────────────────────────────
    CALIBRATION_NOT_APPLIED    = "CALIBRATION_NOT_APPLIED"
    CALIBRATION_APPLIED        = "CALIBRATION_APPLIED"

    # ── Retrieval ve aday durumu ──────────────────────────────────────────────
    NO_CANDIDATE_FOUND         = "NO_CANDIDATE_FOUND"
    LOW_CONFIDENCE             = "LOW_CONFIDENCE"
    MATCH_BELOW_THRESHOLD      = "MATCH_BELOW_THRESHOLD"


# ── İnsan tarafından okunabilir açıklama şablonları ──────────────────────────
_REASON_DESCRIPTIONS: dict[ReasonCode, str] = {
    ReasonCode.EXACT_COMPACT_MATCH:
        "EFT açıklamasında şirket adı boşluk ve noktalama işaretlerinden bağımsız normalizasyon sonrasında tam olarak tespit edildi.",
    ReasonCode.EXACT_OFFICIAL_NAME:
        "Resmi şirket adı tam olarak eşleşti.",
    ReasonCode.EXACT_OFFICIAL_ALIAS:
        "Resmi alias tam olarak eşleşti.",
    ReasonCode.EXACT_CORE_MATCH:
        "Şirketin kök adı (legal suffix hariç) tam eşleşti.",
    ReasonCode.LEGAL_SUFFIX_ONLY_DIFFERENCE:
        "Fark yalnızca hukuki ek (Ltd, Inc, Corp vb.) kaynaklıdır; kök isim aynı.",
    ReasonCode.EXACT_MATCH_REQUIRES_REVIEW:
        "Tam eşleşme tespit edildi ancak isim kısa veya genel olduğu için insan incelemesi önerilir.",
    ReasonCode.SHORT_AMBIGUOUS_EXACT_MATCH:
        "Eşleşen isim çok kısa veya genel bir ifade (örn. ABC, GLOBAL, STAR). Tam eşleşme olsa da skor güvenilir değil.",
    ReasonCode.DUPLICATE_ALIAS_MATCH:
        "Bu alias birden fazla farklı şirkette kullanılmaktadır. Eşleşme belirsiz olabilir.",
    ReasonCode.PARTIAL_MATCH_REQUIRES_REVIEW:
        "Aday üretimi için kısmi bilgi yeterli görülmüştür ancak eksik bilgi fazla olduğu için doğrudan yüksek risk yerine analist incelemesi önerilir.",
    ReasonCode.HIGH_TRIGRAM_SIMILARITY:
        "Trigram karakter benzerliği yüksek.",
    ReasonCode.HIGH_VECTOR_SIMILARITY:
        "Semantik vektör benzerliği yüksek.",
    ReasonCode.HIGH_FUZZY_SIMILARITY:
        "Fuzzy string benzerliği yüksek.",
    ReasonCode.RERANKER_CONFIRMED:
        "Cross-encoder reranker modeli eşleşmeyi güçlü biçimde onayladı.",
    ReasonCode.RERANKER_REJECTED:
        "Reranker modeli eşleşmeyi onaylamadı, skor düşürüldü.",
    ReasonCode.SEMANTIC_MATCH:
        "Anlamsal (semantik) benzerlik eşleşmeyi destekliyor.",
    ReasonCode.CONSONANT_ONLY_MATCH:
        "Sesli harfler çıkarıldığında konsonant dizileri eşleşiyor (kısaltma eşleşmesi).",
    ReasonCode.OFFICIAL_ALIAS_MATCH:
        "Resmi olarak kaydedilmiş bir alias eşleşti.",
    ReasonCode.TRANSLITERATION_MATCH:
        "Farklı alfabeden Latin'e çeviri sonucunda eşleşme sağlandı.",
    ReasonCode.FORMER_NAME_MATCH:
        "Şirketin eski ismiyle eşleşme sağlandı.",
    ReasonCode.ABBREVIATION_MATCH:
        "Kısaltma eşleşmesi bulundu.",
    ReasonCode.ACRONYM_MATCH:
        "Baş harf kısaltması (acronym) eşleşmesi bulundu.",
    ReasonCode.LEETSPEAK_EVASION:
        "Harflerin rakam veya sembollerle değiştirilerek gizlenmeye çalışıldığı (Leetspeak normalizasyonu) tespit edildi.",
    ReasonCode.NER_FALLBACK_USED:
        "NER modeli entity çıkaramadı; aday varyantı fallback olarak kullanıldı.",
    ReasonCode.NO_ENTITY_EXTRACTED:
        "Metinden hiçbir entity çıkarılamadı. Tüm EFT metni kara liste ile karşılaştırıldı.",
    ReasonCode.RULE_BASED_EXTRACTION:
        "Entity, regex/kural tabanlı yöntemle çıkarıldı.",
    ReasonCode.CANDIDATE_SUPPORTED_EXTRACTION:
        "Entity, aday listesi üzerinden metinde aranarak tespit edildi.",
    ReasonCode.FULL_TEXT_FALLBACK_USED:
        "Tüm extraction katmanları başarısız oldu; tam metin fallback kullanıldı.",
    ReasonCode.COUNTRY_CONFLICT:
        "Yardımcı alan çelişkisi: Ülke bilgileri uyuşmuyor.",
    ReasonCode.DATE_OF_BIRTH_CONFLICT:
        "Yardımcı alan çelişkisi: Doğum tarihleri uyuşmuyor.",
    ReasonCode.IDENTIFIER_EXACT_MATCH:
        "Kimlik numarası (pasaport, vergi no) tam olarak eşleşti.",
    ReasonCode.ENTITY_TYPE_CONFLICT:
        "Varlık türü uyuşmazlığı: Biri şirket, diğeri kişi veya gemi olabilir.",
    ReasonCode.ADDRESS_SUPPORTS_MATCH:
        "Adres bilgisi eşleşmeyi destekliyor.",
    ReasonCode.CALIBRATION_NOT_APPLIED:
        "Kalibrasyon modeli mevcut değil; normalize edilmiş reranker skoru olarak kullanıldı.",
    ReasonCode.CALIBRATION_APPLIED:
        "Kalibrasyon uygulandı; olasılık tahmini istatistiksel modelle hesaplandı.",
    ReasonCode.NO_CANDIDATE_FOUND:
        "Hiçbir retrieval kanalından aday bulunamadı.",
    ReasonCode.LOW_CONFIDENCE:
        "Eşleşme skorları düşük güven seviyesinde.",
    ReasonCode.MATCH_BELOW_THRESHOLD:
        "Eşleşme skoru eşik değerinin altında kaldı; alert oluşturulmadı.",
}


def get_description(code: ReasonCode) -> str:
    """
    Reason code için Türkçe açıklama döndürür.

    Args:
        code: ReasonCode enum değeri

    Returns:
        İnsan tarafından okunabilir açıklama
    """
    return _REASON_DESCRIPTIONS.get(code, f"Bilinmeyen gerekçe kodu: {code.value}")


def build_human_explanation(
    entity: Optional[str],
    matched_name: Optional[str],
    codes: list[ReasonCode],
    final_score: float,
    calibrated_probability: Optional[float] = None
) -> str:
    """
    Eşleştirme sonucu için insan tarafından okunabilir açıklama üretir.

    Args:
        entity: Metinden çıkarılan entity
        matched_name: Kara listede eşleşen varyant adı
        codes: Üretilen reason code listesi
        final_score: Final eşleştirme skoru
        calibrated_probability: Kalibrasyon sonrası olasılık (opsiyonel)

    Returns:
        Analist için okunabilir açıklama metni
    """
    parts = []

    # Başlık
    if entity and matched_name:
        parts.append(
            f'"{entity}" ile "{matched_name}" arasında '
            f'{"yüksek" if final_score >= 0.70 else "orta"} olasılıklı eşleşme bulundu.'
        )
    elif matched_name:
        parts.append(
            f'"{matched_name}" ile eşleşme tespit edildi (skor: {final_score:.3f}).'
        )
    else:
        parts.append(f"Eşleşme tespit edildi (skor: {final_score:.3f}).")

    # Skor
    if calibrated_probability is not None:
        parts.append(
            f"Kalibre edilmiş olasılık: %{calibrated_probability * 100:.1f}."
        )

    # Her reason code için kısa açıklama
    for code in codes:
        desc = get_description(code)
        if desc:
            parts.append(desc)

    return " ".join(parts)


def codes_to_list(codes: list[ReasonCode]) -> list[str]:
    """
    ReasonCode listesini string listesine çevirir (DB'ye yazmak için).

    Args:
        codes: ReasonCode listesi

    Returns:
        String listesi
    """
    return [c.value for c in codes]


def list_to_codes(code_strings: list[str]) -> list[ReasonCode]:
    """
    String listesini ReasonCode listesine çevirir (DB'den okumak için).
    Bilinmeyen kodları siler.

    Args:
        code_strings: String listesi

    Returns:
        ReasonCode listesi
    """
    result = []
    for s in code_strings:
        try:
            result.append(ReasonCode(s))
        except ValueError:
            pass  # Bilinmeyen kod — ignore
    return result
