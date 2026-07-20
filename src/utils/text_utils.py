import re
import string

# Extended Suffix Map for standardizing or removing legal suffixes
COMPANY_SUFFIX_MAP = {
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
    "pvt.": "private"
}

LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llc", "llp", "plc", "pvt", "private",
    "technologies", "technology"
}

def normalize_text(text: str) -> str:
    """
    Verilen metni normalize eder.
    Amaç: EFT açıklaması ve şirket isimlerini ortak formata getirmek.
    Noktalama işaretlerini kaldırır, küçük harfe çevirir, gereksiz boşlukları temizler.
    """
    if text is None:
        return ""

    text = str(text).lower()

    # Noktalama işaretlerini boşluğa çevir
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    # Fazla boşlukları temizle
    tokens = text.split()

    normalized_tokens = []
    for token in tokens:
        # ltd -> limited, corp -> corporation gibi dönüşümler
        token = COMPANY_SUFFIX_MAP.get(token, token)
        normalized_tokens.append(token)

    return " ".join(normalized_tokens)


def tokenize(text: str) -> list[str]:
    """
    Normalize edilmiş metni token listesine çevirir.
    """
    normalized = normalize_text(text)
    return normalized.split()


def remove_company_suffixes(text: str) -> str:
    """
    Şirket eklerini çıkarır. (Eski yapı uyumluluğu için tutuluyor)
    """
    tokens = tokenize(text)
    filtered_tokens = [token for token in tokens if token not in LEGAL_SUFFIXES]
    return " ".join(filtered_tokens)


def get_normalized_core_name(text: str) -> str:
    """
    Metni normalize eder ve sondaki veya içerideki yasal ekleri (legal suffixes)
    tamamen temizler. Şirket öz ismini bulmayı hedefler.
    Ör: 'Apple Inc' -> 'apple'
        'Apple Incorporated' -> 'apple'
    """
    return remove_company_suffixes(text)


def is_consonant_match(text1: str, text2: str) -> bool:
    """
    Checks if two strings are identical after removing all vowels.
    This helps match abbreviations like 'mcrsft' with 'microsoft'.
    Ignores matches that are too short (less than 3 consonants).
    """
    if not text1 or not text2:
        return False
        
    def _strip_vowels(s: str) -> str:
        return "".join([c for c in s if c not in "aeiou"])
        
    c1 = _strip_vowels(text1)
    c2 = _strip_vowels(text2)
    
    if c1 == c2 and len(c1) >= 3:
        return True
    return False