from text_utils import normalize_text, remove_company_suffixes


STOPWORDS_FOR_ACRONYM = {
    "limited",
    "llc",
    "incorporated",
    "corporation",
    "company",
    "the",
    "and",
    "of"
}


def generate_acronym(company_name: str) -> str:
    """
    Şirket adından acronym üretir.
    Örnek:
    North Star Trading Limited -> nst
    """

    normalized_name = normalize_text(company_name)
    tokens = normalized_name.split()

    important_tokens = [
        token for token in tokens
        if token not in STOPWORDS_FOR_ACRONYM
    ]

    acronym = "".join(token[0] for token in important_tokens if token)

    return acronym


def generate_aliases(company_name: str) -> list[str]:
    """
    Şirket adı için basit alias varyasyonları üretir.
    """

    normalized_name = normalize_text(company_name)
    without_suffix = remove_company_suffixes(normalized_name)
    acronym = generate_acronym(company_name)

    aliases = {
        normalized_name,
        without_suffix,
        acronym
    }

    # Boş alias varsa çıkar
    aliases = {alias for alias in aliases if alias.strip()}

    return list(aliases)