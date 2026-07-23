"""
text_utils.py — Metin normalizasyon yardımcı fonksiyonları.

Normalizasyon akışı:
  1. Unicode NFKC normalizasyonu
  2. casefold (locale-agnostic lowercase)
  3. Noktalama ve gereksiz boşluk temizliği
  4. Legal suffix normalizasyonu
  5. Token normalizasyonu

Transliteration için ayrı modül: src/utils/transliteration.py
"""

import re
import string
import unicodedata
from typing import Optional

# ── Legal suffix haritası (normalize → standart form) ──────────────────────
COMPANY_SUFFIX_MAP: dict[str, str] = {
    # İngilizce
    "ltd": "limited",
    "ltd.": "limited",
    "llc": "llc",
    "inc": "incorporated",
    "inc.": "incorporated",
    "corp": "corporation",
    "corp.": "corporation",
    "co": "company",
    "co.": "company",
    "plc": "plc",
    "llp": "llp",
    "pvt": "private",
    "pvt.": "private",
    "intl": "international",
    # Türkçe
    "a.s.": "anonim sirketi",
    "as": "anonim sirketi",
    "sti": "sirketi",
    "ltd sti": "limited sirketi",
}

# ── Legal suffix kümesi (core name hesabı için kaldırılacaklar) ─────────────
LEGAL_SUFFIXES: set[str] = {
    # İngilizce
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "llp", "plc", "pvt", "private",
    "technologies", "technology", "intl", "international",
    "group", "holdings", "holding",
    # Türkçe
    "anonim", "sirketi", "limited",
}


def unicode_normalize(text: str) -> str:
    """
    NFKC normalizasyonu uygular.
    Aksan karakterleri ve kompozit Unicode formlarını standartlaştırır.
    Ör: ﬁ (fi ligature) → fi, ² → 2
    """
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)


def normalize_text(text: Optional[str]) -> str:
    """
    Tam metin normalizasyon pipeline'ı.

    Adımlar:
      1. Unicode NFKC normalizasyonu
      2. casefold (Türkçe I/İ için locale-agnostic)
      3. Noktalama işaretlerini boşluğa çevir
      4. Çoklu boşlukları temizle
      5. Legal suffix normalizasyonu

    Args:
        text: Normalize edilecek metin

    Returns:
        Normalize edilmiş metin (boş string hata yerine)
    """
    if text is None:
        return ""

    # 1. Unicode NFKC
    text = unicode_normalize(str(text))

    # 2. casefold (Türkçe "İ" → "i" için str.lower() yerine casefold)
    text = text.casefold()

    # 3. Noktalama işaretlerini boşluğa çevir
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    # 4. Fazla boşlukları temizle
    tokens = text.split()

    # 5. Legal suffix normalizasyonu
    normalized_tokens = [COMPANY_SUFFIX_MAP.get(token, token) for token in tokens]

    return " ".join(normalized_tokens)


def tokenize(text: str) -> list[str]:
    """
    Normalize edilmiş metni token listesine çevirir.

    Args:
        text: Ham metin

    Returns:
        Token listesi
    """
    return normalize_text(text).split()


def remove_company_suffixes(text: str) -> str:
    """
    Metinden yasal şirket eklerini kaldırır.
    Core name hesabında kullanılır.

    Args:
        text: Normalize edilmiş metin

    Returns:
        Legal suffix olmadan metin
    """
    tokens = tokenize(text)
    filtered = [token for token in tokens if token not in LEGAL_SUFFIXES]
    return " ".join(filtered)


def get_normalized_core_name(text: str) -> str:
    """
    Metni normalize edip legal suffix'leri kaldırarak şirket öz ismini döndürür.

    Ör: 'Apple Inc.'   → 'apple'
        'Apple Incorporated' → 'apple'
        'Microsoft Corp' → 'microsoft'

    Args:
        text: Şirket adı veya EFT açıklaması

    Returns:
        Sadece öz isim tokenleri içeren string
    """
    return remove_company_suffixes(text)


def is_consonant_match(text1: str, text2: str) -> bool:
    """
    İki string'in sesli harfler kaldırıldıktan sonra özdeş olup olmadığını kontrol eder.
    Kısaltma tespiti için kullanılır (ör. 'mcrsft' ↔ 'microsoft').
    3 karakterden kısa eşleşmeleri görmezden gelir.

    Args:
        text1: İlk string
        text2: İkinci string

    Returns:
        True eğer konsonant dizileri aynı ve en az 3 karakter ise
    """
    if not text1 or not text2:
        return False

    def _strip_vowels(s: str) -> str:
        return "".join(c for c in s if c not in "aeiouAEIOUıİüÜöÖ")

    c1 = _strip_vowels(text1.casefold())
    c2 = _strip_vowels(text2.casefold())

    return bool(c1 == c2 and len(c1) >= 3)


def detect_script(text: str) -> str:
    """
    Metindeki baskın Unicode alfabesini tespit eder.

    Returns:
        'ARABIC', 'CYRILLIC', 'LATIN', 'GEORGIAN', 'HANGUL', 'UNKNOWN'
    """
    if not text:
        return "UNKNOWN"

    script_counts: dict[str, int] = {}
    for char in text:
        cp = ord(char)
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            script_counts["ARABIC"] = script_counts.get("ARABIC", 0) + 1
        elif 0x0400 <= cp <= 0x04FF:
            script_counts["CYRILLIC"] = script_counts.get("CYRILLIC", 0) + 1
        elif 0x10A0 <= cp <= 0x10FF:
            script_counts["GEORGIAN"] = script_counts.get("GEORGIAN", 0) + 1
        elif 0xAC00 <= cp <= 0xD7AF:
            script_counts["HANGUL"] = script_counts.get("HANGUL", 0) + 1
        elif 0x0000 <= cp <= 0x007F or 0x00A0 <= cp <= 0x024F:
            script_counts["LATIN"] = script_counts.get("LATIN", 0) + 1

    if not script_counts:
        return "UNKNOWN"

    return max(script_counts, key=lambda k: script_counts[k])


def remove_accents(text: str) -> str:
    """
    Aksan ve diakritik işaretleri kaldırır (NFD + non-spacing mark filtresi).
    Ör: 'café' → 'cafe', 'naïve' → 'naive'

    Args:
        text: Ham metin

    Returns:
        Aksansız metin
    """
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")
