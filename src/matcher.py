from rapidfuzz import fuzz
from text_utils import normalize_text, tokenize
from alias_utils import generate_acronym


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
    "import"
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


def calculate_rule_score(description: str, company_name: str) -> float:
    """
    Basit kural bazlı skor üretir.
    """

    desc_tokens = set(tokenize(description))
    company_tokens = set(tokenize(company_name))

    common_tokens = desc_tokens.intersection(company_tokens)

    if not common_tokens:
        return 0.0

    important_common_tokens = [
        token for token in common_tokens
        if token not in GENERAL_WORDS
    ]

    score = 0.0

    # Önemli ortak token varsa skor artır
    if important_common_tokens:
        score += 0.5

    # Birden fazla ortak token varsa skor artır
    if len(common_tokens) >= 2:
        score += 0.3

    # Eşleşme sadece genel kelimelerden oluşuyorsa cezalandır
    if common_tokens and not important_common_tokens:
        score -= 0.3

    # Skoru 0-1 aralığında tut
    score = max(0.0, min(1.0, score))

    return score


def calculate_final_score(
    fuzzy_score: float,
    acronym_score: float,
    rule_score: float
) -> float:
    """
    İlk MVP final skor formülü.
    Vector similarity sonraki fazda eklenecek.
    """

    final_score = (
        0.70 * fuzzy_score +
        0.20 * acronym_score +
        0.10 * rule_score
    )

    return round(final_score, 4)


def assign_risk_level(final_score: float) -> str:
    """
    Final skora göre risk seviyesi atar.
    """

    if final_score >= 0.85:
        return "High Risk"
    elif final_score >= 0.70:
        return "Medium Risk"
    elif final_score >= 0.55:
        return "Low Risk"
    else:
        return "No Match"


def build_reason(
    fuzzy_score: float,
    acronym_score: float,
    rule_score: float
) -> str:
    """
    Skorların nedenini açıklamak için basit reason üretir.
    """

    reasons = []

    if fuzzy_score >= 0.75:
        reasons.append("Yazımsal benzerlik yüksek bulundu.")

    if acronym_score == 1.0:
        reasons.append("Şirket acronym'i EFT açıklamasında bulundu.")

    if rule_score >= 0.5:
        reasons.append("Şirket adıyla ortak önemli tokenlar bulundu.")

    if not reasons:
        reasons.append("Güçlü eşleşme sinyali bulunamadı.")

    return " ".join(reasons)