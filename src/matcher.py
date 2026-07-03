from rapidfuzz import fuzz
from text_utils import normalize_text, tokenize
from alias_utils import generate_acronym
from config import (
    FUZZY_WEIGHT,
    VECTOR_WEIGHT,
    ACRONYM_WEIGHT,
    RULE_WEIGHT,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD
)

GENERAL_WORDS = {
    "global",
    "trade",
    "trading",
    "group",
    "international",
    "company",
    "limited",
    "corporation",
    "incorporated",
    "llc",
    "service",
    "services",
    "payment",
    "invoice",
    "export",
    "import",
    "transfer",
    "eft",
    "inv",
    "for",
    "to",
    "market",
    "random",
    "inc",
    "ltd",
    "co"
}


def calculate_fuzzy_score(description: str, alias: str) -> float:
    """
    EFT açıklaması ile şirket alias'ı arasında fuzzy skor üretir.
    0-1 arası skor döner.
    """

    normalized_description = normalize_text(description)
    normalized_alias = normalize_text(alias)

    score = fuzz.WRatio(normalized_description, normalized_alias)

    return score / 100


def calculate_acronym_score(description: str, company_name: str) -> float:
    """
    Şirket acronym'i EFT açıklamasında geçiyor mu kontrol eder.
    """

    normalized_description = normalize_text(description)
    acronym = generate_acronym(company_name)

    if not acronym:
        return 0.0

    description_tokens = normalized_description.split()

    if acronym in description_tokens:
        return 1.0

    return 0.0


def get_common_tokens(description: str, company_name: str) -> set[str]:
    """
    Açıklama ve şirket adı arasındaki ortak tokenları döndürür.
    """

    desc_tokens = set(tokenize(description))
    company_tokens = set(tokenize(company_name))

    return desc_tokens.intersection(company_tokens)


def get_important_common_tokens(description: str, company_name: str) -> set[str]:
    """
    Açıklama ve şirket adı arasındaki genel olmayan ortak tokenları döndürür.
    """

    common_tokens = get_common_tokens(description, company_name)

    important_common_tokens = {
        token for token in common_tokens
        if token not in GENERAL_WORDS
    }

    return important_common_tokens


def calculate_rule_score(description: str, company_name: str) -> float:
    """
    Basit kural bazlı skor üretir.
    """

    common_tokens = get_common_tokens(description, company_name)
    important_common_tokens = get_important_common_tokens(description, company_name)

    if not common_tokens:
        return 0.0

    score = 0.0

    # Önemli ortak token varsa skor artır
    if important_common_tokens:
        score += 0.5

    # Birden fazla ortak token varsa skor artır
    if len(common_tokens) >= 2:
        score += 0.3

    # İki veya daha fazla önemli ortak token varsa daha güçlü kabul et
    if len(important_common_tokens) >= 2:
        score += 0.2

    # Eşleşme sadece genel kelimelerden oluşuyorsa cezalandır
    if common_tokens and not important_common_tokens:
        score -= 0.4

    score = max(0.0, min(1.0, score))

    return score


def calculate_final_score(
    fuzzy_score: float,
    acronym_score: float,
    rule_score: float,
    vector_score: float = 0.0
) -> float:
    """
    Fuzzy + vector + acronym + rule score ile final skor üretir.
    Ağırlıklar config.py dosyasından alınır.
    """

    final_score = (
        FUZZY_WEIGHT * fuzzy_score +
        VECTOR_WEIGHT * vector_score +
        ACRONYM_WEIGHT * acronym_score +
        RULE_WEIGHT * rule_score
    )

    return round(final_score, 4)


def is_valid_match(
    fuzzy_score: float,
    acronym_score: float,
    rule_score: float,
    vector_score: float,
    candidate_source: str
) -> bool:
    """
    Eşleşmenin gerçekten dikkate değer olup olmadığını kontrol eder.
    """

    # Çok güçlü acronym varsa kabul edilebilir
    if acronym_score == 1.0 and (fuzzy_score >= 0.40 or vector_score >= 0.55):
        return True

    # Token index ile geldiyse daha esnek
    if candidate_source == "token_index":
        if fuzzy_score >= 0.55 and rule_score >= 0.3:
            return True

        if vector_score >= 0.70 and rule_score >= 0.3:
            return True

        if fuzzy_score >= 0.70:
            return True

    # Fuzzy fallback ile geldiyse daha sıkı
    if candidate_source == "fuzzy_fallback":
        if fuzzy_score >= 0.78 and rule_score >= 0.3:
            return True

        if vector_score >= 0.78 and rule_score >= 0.3:
            return True

    return False


def assign_risk_level(
    final_score: float,
    fuzzy_score: float,
    acronym_score: float,
    rule_score: float,
    vector_score: float,
    candidate_source: str
) -> str:
    """
    Final skora ve geçerlilik kontrolüne göre risk seviyesi atar.
    """

    valid_match = is_valid_match(
        fuzzy_score=fuzzy_score,
        acronym_score=acronym_score,
        rule_score=rule_score,
        vector_score=vector_score,
        candidate_source=candidate_source
    )

    if final_score >= HIGH_RISK_THRESHOLD:
        return "High Risk"
    elif final_score >= MEDIUM_RISK_THRESHOLD:
        return "Medium Risk"
    elif final_score >= LOW_RISK_THRESHOLD:
        return "Low Risk"
    else:
        return "No Match"


def build_reason(
    fuzzy_score: float,
    acronym_score: float,
    rule_score: float,
    vector_score: float,
    candidate_source: str
) -> str:
    """
    Skorların nedenini açıklamak için basit reason üretir.
    """

    reasons = []

    if candidate_source == "token_index":
        reasons.append("Aday şirket token index üzerinden bulundu.")

    if candidate_source == "fuzzy_fallback":
        reasons.append("Aday şirket fuzzy fallback ile bulundu.")

    if fuzzy_score >= 0.75:
        reasons.append("Yazımsal benzerlik yüksek bulundu.")
    elif fuzzy_score >= 0.55:
        reasons.append("Yazımsal benzerlik orta seviyede bulundu.")

    if vector_score >= 0.75:
        reasons.append("Vector similarity yüksek bulundu.")
    elif vector_score >= 0.60:
        reasons.append("Vector similarity orta seviyede bulundu.")

    if acronym_score == 1.0:
        reasons.append("Şirket acronym'i EFT açıklamasında bulundu.")

    if rule_score >= 0.5:
        reasons.append("Şirket adıyla ortak önemli tokenlar bulundu.")

    if not reasons:
        reasons.append("Güçlü eşleşme sinyali bulunamadı.")

    return " ".join(reasons)