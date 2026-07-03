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
    "to",
    "inc",
    "ltd",
    "co"
}


COMPANY_SUFFIX_WORDS = {
    "limited",
    "llc",
    "incorporated",
    "corporation",
    "company",
    "inc",
    "ltd",
    "co"
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


def build_alias_token_index(alias_df):
    """
    Alias kayıtlarından token index oluşturur.

    Örnek index:
    {
        "abc": {0, 3, 8},
        "north": {1},
        "star": {1, 7}
    }

    Buradaki sayılar alias_df index değerleridir.
    """

    token_index = {}

    for row_index, row in alias_df.iterrows():
        alias = row["alias"]

        important_tokens = get_important_tokens(alias)

        for token in important_tokens:
            if token not in token_index:
                token_index[token] = set()

            token_index[token].add(row_index)

    return token_index


def get_alias_rows_by_indices(alias_df, row_indices: set[int]):
    """
    alias_df içinden sadece verilen indexlere sahip satırları döndürür.
    """

    if not row_indices:
        return []

    rows = []

    for row_index in row_indices:
        row = alias_df.loc[row_index]
        rows.append(row.to_dict())

    return rows


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


def find_candidate_aliases_with_index(
    description: str,
    alias_df,
    token_index: dict,
    min_candidate_score: float = 0.4,
    max_candidates: int = 20,
    fuzzy_fallback_limit: int = 5
):
    """
    Bir EFT açıklaması için token index kullanarak olası şirket alias adaylarını seçer.

    Bu fonksiyon tüm alias_df üzerinde dönmez.
    Önce açıklamadaki önemli tokenlara bakar.
    Sonra sadece bu tokenlarla ilişkili alias kayıtlarını skorlar.
    """

    description_important_tokens = get_important_tokens(description)

    candidate_row_indices = set()

    for token in description_important_tokens:
        matched_indices = token_index.get(token, set())
        candidate_row_indices.update(matched_indices)

    candidate_rows = get_alias_rows_by_indices(alias_df, candidate_row_indices)

    candidates = []

    for alias_row in candidate_rows:
        alias = alias_row["alias"]

        candidate_score = cheap_candidate_score(description, alias)

        if candidate_score >= min_candidate_score:
            alias_row["candidate_filter_score"] = round(candidate_score, 4)
            alias_row["candidate_source"] = "token_index"
            candidates.append(alias_row)

    candidates = sorted(
        candidates,
        key=lambda x: x["candidate_filter_score"],
        reverse=True
    )

    candidates = candidates[:max_candidates]

    # Eğer token index hiç aday bulamazsa fuzzy fallback çalışır.
    # Bu fallback hâlâ tüm alias_df üzerinde döner ama sadece aday çıkmayan kayıtlar için devreye girer.
    if not candidates:
        normalized_description = normalize_text(description)

        fallback_candidates = []

        for _, alias_row in alias_df.iterrows():
            alias = alias_row["alias"]
            normalized_alias = normalize_text(alias)

            fuzzy_score = fuzz.WRatio(
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