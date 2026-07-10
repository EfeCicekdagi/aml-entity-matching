import re
import string


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
}


def normalize_text(text: str) -> str:
    """
    Verilen metni normalize eder.
    Amaç: EFT açıklaması ve şirket isimlerini ortak formata getirmek.
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
    Şirket eklerini çıkarır.
    Örnek:
    'abc trading limited' -> 'abc trading'
    """

    suffixes = {
        "limited",
        "llc",
        "incorporated",
        "corporation",
        "company",
        "corp",
        "inc",
        "ltd",
        "co"
    }

    tokens = tokenize(text)
    filtered_tokens = [token for token in tokens if token not in suffixes]

    return " ".join(filtered_tokens)