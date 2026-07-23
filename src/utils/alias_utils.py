from itertools import product

from src.utils.text_utils import normalize_text, remove_company_suffixes


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


TOKEN_VARIATIONS = {
    "limited": ["limited", "ltd"],
    "corporation": ["corporation", "corp"],
    "incorporated": ["incorporated", "inc"],
    "company": ["company", "co"],
    "trading": ["trading", "trade", "trdng"],
    "logistics": ["logistics", "log"],
    "petroleum": ["petroleum", "petro"],
    "international": ["international", "intl"],
    "global": ["global", "glbl"],
    "export": ["export", "exp"],
    "import": ["import", "imp"],
    "energy": ["energy", "enrgy"],
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


def generate_token_variations(token: str) -> list[str]:
    """
    Bir token için varyasyonları üretir.
    Örnek:
    trading -> trading, trade, trdng
    limited -> limited, ltd
    """

    normalized_token = normalize_text(token)

    return TOKEN_VARIATIONS.get(normalized_token, [normalized_token])


def generate_abbreviated_aliases(company_name: str, max_alias_count: int = 50) -> list[str]:
    """
    Şirket adından kısaltmalı alias varyasyonları üretir.

    Örnek:
    ABC Trading Limited
    ->
    abc trading limited
    abc trading ltd
    abc trdng limited
    abc trdng ltd
    abc trade limited
    abc trade ltd
    """

    normalized_name = normalize_text(company_name)
    tokens = normalized_name.split()

    token_variation_lists = [
        generate_token_variations(token)
        for token in tokens
    ]

    aliases = set()

    for combination in product(*token_variation_lists):
        alias = " ".join(combination).strip()

        if alias:
            aliases.add(alias)

        if len(aliases) >= max_alias_count:
            break

    return list(aliases)


def generate_aliases(company_name: str) -> list[str]:
    """
    Şirket adı için alias varyasyonları üretir.
    """

    normalized_name = normalize_text(company_name)
    without_suffix = remove_company_suffixes(normalized_name)
    acronym = generate_acronym(company_name)
    abbreviated_aliases = generate_abbreviated_aliases(company_name)

    aliases = {
        normalized_name,
        without_suffix,
        acronym,
        *abbreviated_aliases
    }

    aliases = {alias for alias in aliases if alias.strip()}

    return list(aliases)
