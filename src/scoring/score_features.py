import difflib
from src.utils.text_utils import (
    normalize_text,
    remove_company_suffixes,
    get_normalized_core_name,
    is_consonant_match
)
from src.utils.alias_utils import generate_acronym

_RULE_STOPWORDS = {
    "services", "service", "group", "holding", "holdings",
    "international", "global", "enterprises", "enterprise",
    "solutions", "solution", "industries", "industry",
    "management", "investments", "investment",
    "trading", "trade", "export", "import",
    "logistics", "transport", "energy", "petroleum",
}


def _acronym_score(explanation: str, variant_name: str) -> float:
    """EFT açıklamasında şirket kısaltması arar. 1.0 veya 0.0 döner."""
    acronym = generate_acronym(variant_name)
    if acronym and len(acronym) >= 2 and acronym in explanation.split():
        return 1.0
    return 0.0


def _rule_score(explanation: str, variant_name: str) -> float:
    """Token overlap skoru. Generic kelimeler ve kısa tokenler hariç."""
    clean_variant = remove_company_suffixes(normalize_text(variant_name))
    variant_tokens = {
        t for t in clean_variant.split()
        if len(t) > 3 and t not in _RULE_STOPWORDS
    }
    if not variant_tokens:
        return 0.0
    exp_tokens = set(explanation.split())
    overlap = variant_tokens & exp_tokens
    return len(overlap) / len(variant_tokens)


def _exact_name_score(explanation: str, variant_name: str) -> float:
    """Variant adı EFT'de tam geçiyor mu? En az 2 anlamlı token gerektirir."""
    norm_variant = normalize_text(variant_name)
    tokens = [t for t in norm_variant.split() if len(t) > 3]
    if len(tokens) < 2:
        return 0.0
    exp_tokens = set(explanation.split())
    if all(t in exp_tokens for t in tokens):
        return 1.0
    return 0.0


def build_score_features(
    norm_exp: str,
    cand: dict,
    extracted_entity: str = None
) -> dict:
    """
    Candidate için kural tabanlı, fuzzy ve overlap özelliklerini hesaplar.
    """
    fuzzy_score  = cand.get("trgm_score", 0.0)
    vector_score = cand.get("vector_score", 0.0)
    fts_score    = cand.get("full_text_score", 0.0)
    norm_reranker = cand.get("normalized_reranker_score", 0.0)

    variant_name = cand.get("variant_name", "")
    norm_cand    = normalize_text(variant_name)
    core_query   = get_normalized_core_name(norm_exp)
    core_cand    = get_normalized_core_name(variant_name)

    exact_normalized_match = bool(norm_exp == norm_cand and norm_exp)
    exact_core_match       = bool(core_query == core_cand and core_query)
    legal_suffix_only_diff = exact_core_match and not exact_normalized_match
    query_is_contained     = bool(norm_exp in norm_cand and norm_exp)
    cand_is_contained      = bool(norm_cand in norm_exp and norm_cand)
    query_token_count      = len(norm_exp.split())

    scores_dict = {
        "fuzzy_score":    fuzzy_score,
        "vector_score":   vector_score,
        "acronym_score":  _acronym_score(norm_exp, variant_name),
        "rule_score":     max(
            _rule_score(norm_exp, variant_name),
            _exact_name_score(norm_exp, variant_name)
        ),
        "reranker_score": norm_reranker,
        "query_token_count": query_token_count,
        "exact_normalized_match": exact_normalized_match,
        "exact_core_match": exact_core_match,
        "legal_suffix_only_difference": legal_suffix_only_diff,
        "query_is_contained_in_candidate": query_is_contained,
        "candidate_is_contained_in_query": cand_is_contained,
        "consonant_match": is_consonant_match(core_query, core_cand),
        "_query_str":   norm_exp,
        "_variant_str": norm_cand,
    }

    if extracted_entity:
        fuzzy_ext = difflib.SequenceMatcher(
            None, extracted_entity.lower(), variant_name.lower()
        ).ratio()
        
        if core_cand:
            fuzzy_ext = max(
                fuzzy_ext,
                difflib.SequenceMatcher(None, extracted_entity.lower(), core_cand.lower()).ratio()
            )
        
        scores_dict["fuzzy_score"] = max(scores_dict["fuzzy_score"], fuzzy_ext)

        if is_consonant_match(extracted_entity, core_cand):
            scores_dict["consonant_match"] = True

        scores_dict["acronym_score"] = max(
            scores_dict["acronym_score"],
            _acronym_score(extracted_entity, variant_name)
        )
        scores_dict["rule_score"] = max(
            scores_dict["rule_score"],
            max(
                _rule_score(extracted_entity, variant_name),
                _exact_name_score(extracted_entity, variant_name)
            )
        )

    return scores_dict
