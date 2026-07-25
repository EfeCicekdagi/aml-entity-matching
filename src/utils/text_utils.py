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
    # İngilizce / Uluslararası
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
    "pte": "private",
    "pte.": "private",
    "pty": "private",
    "pty.": "private",
    "gmbh": "gmbh",
    "sa": "sa",
    "s.a.": "sa",
    "ag": "ag",
    "bv": "bv",
    "b.v.": "bv",
    "nv": "nv",
    "n.v.": "nv",
    "srl": "srl",
    "s.r.l.": "srl",
    "spa": "spa",
    "s.p.a.": "spa",
    "oy": "oy",
    "ab": "ab",
    "sdn": "sendirian",
    "bhd": "berhad",
    "bhd.": "berhad",
    "tbk": "tbk",
    "intl": "international",
    "ent": "enterprises",
    "sol": "solutions",
    "ind": "industries",
    "svc": "services",
    # Türkçe
    "a.s.": "anonim sirketi",
    "as": "anonim sirketi",
    "sti": "sirketi",
    "ltd sti": "limited sirketi",
    "ltd. sti.": "limited sirketi",
    "san": "sanayi",
    "san.": "sanayi",
    "tic": "ticaret",
    "tic.": "ticaret",
}

# ── Legal suffix kümesi (core name hesabı için kaldırılacaklar) ─────────────
LEGAL_SUFFIXES: set[str] = {
    # İngilizce / Uluslararası
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "llp", "plc", "pvt", "private", "pte", "pty",
    "gmbh", "sa", "ag", "bv", "nv", "srl", "spa", "oy", "ab", "sendirian", "berhad", "tbk",
    "technologies", "technology", "intl", "international",
    "group", "holdings", "holding", "enterprises", "enterprise", "solutions", "solution",
    "industries", "industry", "services", "service", "partners", "partner", "associates", "associate",
    # Türkçe
    "anonim", "sirketi", "limited", "sanayi", "ticaret", "ve",
}

# ── Compact match haritası ────────────────────────────────────────────────
COMPACT_SUFFIX_MAP: dict[str, str] = {
    "limited": "ltd",
    "private": "pvt",
    "incorporated": "inc",
    "corporation": "corp",
    "company": "co",
}

_ALL_SUFFIX_WORDS = sorted(list(LEGAL_SUFFIXES | {
    "entertainment", "enterprises", "technologies", "technology", "solutions",
    "services", "service", "international", "holding", "holdings", "investments",
    "investment", "industries", "industry", "trading", "trade", "group", "limited",
    "private", "pvt", "ltd", "inc", "corp", "corporation", "company", "llc",
    "anonim", "sirketi", "sanayi", "ticaret", "berhad", "sendirian", "pvtltd",
    "limitedsirketi", "anonimsirketi"
}), key=len, reverse=True)

_COMPACT_SUFFIX_STRINGS = sorted(list({
    re.sub(r'[\W_]', '', s.casefold()) for s in _ALL_SUFFIX_WORDS if len(s) > 2
}), key=len, reverse=True)

# ── Leetspeak haritası ───────────────────────────────────────────────────
LEETSPEAK_MAP: dict[str, str] = {
    '0': 'o',
    '1': 'i',
    '3': 'e',
    '4': 'a',
    '5': 's',
    '7': 't',
    '8': 'b',
    '@': 'a',
    '$': 's',
    '!': 'i',
    '|': 'l',
}


def normalize_leetspeak(text: Optional[str]) -> str:
    """
    Harflerin rakamlar veya sembollerle değiştirilerek (Leetspeak: 0->o, 1->i, 3->e, 4->a, 5->s, 7->t, 8->b vb.)
    yapılan gizleme (evasion/obfuscation) girişimlerini normalize eder.
    Ör: 'M!cr0s0ft C0rp0r4t!0n' -> 'microsoft corporation'
        '4ppl3 Inc' -> 'apple inc'
        '0r4cl3' -> 'oracle'
        'F!nsp!r3' -> 'finspire'
    """
    if not text:
        return ""
    text_str = str(text).casefold()
    trans_table = str.maketrans(LEETSPEAK_MAP)
    return text_str.translate(trans_table)


def check_leetspeak_evasion(query: Optional[str], candidate: Optional[str]) -> tuple[bool, float, str]:
    """
    Sorgu ile aday arasında Leetspeak (harf yerine rakam/sembol kullanımı) ile gizleme (evasion)
    yapılıp yapılmadığını kontrol eder.

    Returns:
        (evasion_detected: bool, improved_score: float, leet_normalized_query: str)
    """
    if not query or not candidate:
        return False, 0.0, ""

    norm_q = normalize_text(str(query))
    norm_c = normalize_text(str(candidate))

    if not norm_q or not norm_c:
        return False, 0.0, ""

    has_leet_chars = any(c in LEETSPEAK_MAP for c in norm_q)
    if not has_leet_chars:
        return False, 0.0, norm_q

    leet_q = normalize_text(normalize_leetspeak(query))
    leet_c = normalize_text(normalize_leetspeak(candidate))

    if leet_q == norm_q:
        return False, 0.0, leet_q

    import difflib
    def _tok_sim(q_str: str, c_str: str) -> float:
        c_toks = [w for w in c_str.split() if len(w) > 2 and w not in LEGAL_SUFFIXES]
        q_toks = [w for w in q_str.split() if len(w) > 2 and w not in LEGAL_SUFFIXES]
        if not c_toks or not q_toks:
            return 0.0
        return sum(max(difflib.SequenceMatcher(None, qt, ct).ratio() for qt in q_toks) for ct in c_toks) / len(c_toks)

    orig_sim = max(difflib.SequenceMatcher(None, norm_q, norm_c).ratio(), _tok_sim(norm_q, norm_c))
    leet_sim = max(difflib.SequenceMatcher(None, leet_q, leet_c).ratio(), _tok_sim(leet_q, leet_c))

    q_tokens = leet_q.split()
    c_tokens = set(leet_c.split()) if leet_c else set(norm_c.split())

    token_match = any(
        qt in c_tokens and len(qt) >= 3 and qt not in LEGAL_SUFFIXES
        for qt in q_tokens
    )

    evasion_detected = bool(
        (leet_sim > 0.80 and leet_sim > orig_sim + 0.05) or
        (token_match and orig_sim < 0.95 and leet_q != norm_q)
    )

    return evasion_detected, leet_sim, leet_q



def clean_spaced_characters(text: str) -> str:
    """
    Harfleri arasına boşluk veya noktalama konularak gizlenmeye çalışılmış (evasion/obfuscation)
    metinleri birleştirir. Ör: 'M i c r o s o f t', 'M.i.c.r.o.s.o.f.t', 'M-I-C-R-O-S-O-F-T', '(M)(i)(c)' -> 'Microsoft', 'MICROSOFT', 'Mic'.
    En az 2 tekli karakterin yanyana gelmesi durumunda çalışır.
    """
    if not text:
        return ""
    return re.sub(r'\b(?:[^\W_](?:[\.,\-/\(\)\|\*\:]+|[ \t])){1,}[^\W_]\b', lambda m: re.sub(r'[\s\.,\-/\(\)\|\*\:]+', '', m.group(0)), str(text))


def compact_normalize(text: Optional[str]) -> str:
    """
    Compact exact match için metni standardize eder.
    1. NFKC + casefold + spaced evasion temizliği
    2. Suffix'leri kısalt (limited -> ltd vb.)
    3. Harf ve rakam dışındaki her şeyi (boşluk, noktalama) kaldırır.
    """
    if not text:
        return ""
    
    text = unicode_normalize(str(text))
    text = clean_spaced_characters(text).casefold()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = clean_spaced_characters(text)
    
    tokens = text.split()
    compacted_tokens = [COMPACT_SUFFIX_MAP.get(t, t) for t in tokens]
    
    compact_text = "".join(compacted_tokens)
    compact_text = re.sub(r'[\W_]', '', compact_text)
    for k, v in COMPACT_SUFFIX_MAP.items():
        compact_text = compact_text.replace(k, v)
    return compact_text


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
      2. Harf arasına boşluk konularak gizleme (obfuscation/evasion) temizliği
      3. casefold (Türkçe I/İ için locale-agnostic)
      4. Noktalama işaretlerini boşluğa çevir
      5. Çoklu boşlukları ve tekrar eden boşluklu harfleri temizle
      6. Legal suffix normalizasyonu

    Args:
        text: Normalize edilecek metin

    Returns:
        Normalize edilmiş metin (boş string hata yerine)
    """
    if text is None:
        return ""

    # 1. Unicode NFKC
    text = unicode_normalize(str(text))

    # 1.5 Harf arasına boşluk konularak gizleme (obfuscation) temizliği
    text = clean_spaced_characters(text)

    # 2. casefold (Türkçe "İ" → "i" için str.lower() yerine casefold)
    text = text.casefold()

    # 3. Noktalama işaretlerini boşluğa çevir
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    # 3.5 Noktalama sonrasında oluşabilecek boşluklu harf öbeklerini tekrar birleştir
    text = clean_spaced_characters(text)

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


def get_compact_core_name(text: Optional[str]) -> str:
    """
    Kelimelerin bitişik veya ayrı yazılması (concatenated/split) durumlarında dahi
    şirket öz ismini saf alfanümerik compact formda döndürür.

    Ör: 'microsoftcorporation'     → 'microsoft'
        'micro soft corporation'   → 'microsoft'
        'indiaforensicservices'    → 'indiaforensic'
        'india forensic services'  → 'indiaforensic'
        'finspiresolutions'        → 'finspire'
        'fin spire solutions'      → 'finspire'
    """
    if not text:
        return ""
    c = compact_normalize(text)
    changed = True
    while changed and len(c) > 3:
        changed = False
        for s in _COMPACT_SUFFIX_STRINGS:
            if c.endswith(s) and len(c) - len(s) >= 3:
                c = c[:-len(s)]
                changed = True
    return c


def get_leetspeak_compact_core_name(text: Optional[str]) -> str:
    """
    Leetspeak karakterleri normalize edildikten sonra compact core name hesaplar.
    Ör: 'm1cr0s0ftc0rp0r4t10n' -> 'microsoft'
    """
    if not text:
        return ""
    return get_compact_core_name(normalize_leetspeak(text))


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
