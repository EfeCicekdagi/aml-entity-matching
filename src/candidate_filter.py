from rapidfuzz import fuzz

from text_utils import normalize_text, tokenize


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
    "to"
}


COMPANY_SUFFIX_WORDS = {
    "limited",
    "llc",
    "incorporated",
    "corporation",
    "company"
}


def get_important_tokens(text: str) -> set[str]:
    """
    Metinden genel kelimeleri çıkarıp önemli tokenları döndürür.
    """

    tokens = set(tokenize(text))

    important_tokens = {
        token for token in tokens
        if token not in GENERAL_WORDS and len(token) >= 2
    }

    return important_tokens


def has_acronym_match(description_tokens: set[str], alias: str) -> bool:
    """
    Alias acronym ise açıklamada birebir geçiyor mu kontrol eder.
    Örnek:
    alias = "nst"
    description = "payment nst ltd"
    """

    normalized_alias = normalize_text(alias)

    if len(normalized_alias) < 2:
        return False

    return normalized_alias in description_tokens


def has_token_overlap(description: str, alias: str) -> bool:
    """
    EFT açıklaması ile alias arasında önemli token ortaklığı var mı?
    """

    desc_important_tokens = get_important_tokens(description)
    alias_important_tokens = get_important_tokens(alias)

    common_tokens = desc_important_tokens.intersection(alias_important_tokens)

    return len(common_tokens) > 0


def has_suffix_signal(description: str, alias: str) -> bool:
    """
    Açıklama ve alias tarafında şirket eki var mı kontrol eder.
    Bu tek başına güçlü sinyal değildir, yardımcı sinyaldir.
    """

    desc_tokens = set(tokenize(description))
    alias_tokens = set(tokenize(alias))

    desc_suffixes = desc_tokens.intersection(COMPANY_SUFFIX_WORDS)
    alias_suffixes = alias_tokens.intersection(COMPANY_SUFFIX_WORDS)

    return bool(desc_suffixes and alias_suffixes)


def cheap_candidate_score(description: str, alias: str) -> float:
    """
    Ucuz aday skoru üretir.
    Bu final score değildir.
    Sadece aday seçmek için kullanılır.
    """

    description_tokens = set(tokenize(description))

    score = 0.0

    if has_token_overlap(description, alias):
        score += 0.5

    if has_acronym_match(description_tokens, alias):
        score += 0.4

    if has_suffix_signal(description, alias):
        score += 0.1

    return min(score, 1.0)


def find_candidate_aliases(
    description: str,
    alias_df,
    min_candidate_score: float = 0.4,
    max_candidates: int = 20,
    fuzzy_fallback_limit: int = 5
):
    """
    Bir EFT açıklaması için olası şirket alias adaylarını seçer.

    Önce ucuz token/acronym/suffix sinyalleriyle aday bulur.
    Eğer hiç aday bulunamazsa fuzzy fallback ile en yakın birkaç alias'ı getirir.
    """

    candidates = []

    for _, alias_row in alias_df.iterrows():
        alias = alias_row["alias"]

        candidate_score = cheap_candidate_score(description, alias)

        if candidate_score >= min_candidate_score:
            row_dict = alias_row.to_dict()
            row_dict["candidate_filter_score"] = round(candidate_score, 4)
            row_dict["candidate_source"] = "cheap_filter"
            candidates.append(row_dict)

    # En güçlü adayları tut
    candidates = sorted(
        candidates,
        key=lambda x: x["candidate_filter_score"],
        reverse=True
    )

    candidates = candidates[:max_candidates]

    # Eğer hiç aday çıkmazsa fuzzy fallback çalıştır
    if not candidates:
        normalized_description = normalize_text(description)

        fallback_candidates = []

        for _, alias_row in alias_df.iterrows():
            alias = alias_row["alias"]
            normalized_alias = normalize_text(alias)

            fuzzy_score = fuzz.partial_ratio(
                normalized_description,
                normalized_alias
            ) / 100

            fallback_candidates.append({
                **alias_row.to_dict(),
                "candidate_filter_score": round(fuzzy_score, 4),
                "candidate_source": "fuzzy_fallback"
            })

        fallback_candidates = sorted(
            fallback_candidates,
            key=lambda x: x["candidate_filter_score"],
            reverse=True
        )

        candidates = fallback_candidates[:fuzzy_fallback_limit]

    return candidates