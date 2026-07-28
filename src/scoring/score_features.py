import re
import difflib
from src.utils.text_utils import (
    normalize_text,
    remove_company_suffixes,
    get_normalized_core_name,
    get_compact_core_name,
    is_consonant_match,
    compact_normalize,
    normalize_leetspeak,
    check_leetspeak_evasion,
    _AMBIGUOUS_SHORT_NAMES,
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

_TOKEN_MATCH_STOPWORDS = {
    "to", "for", "in", "by", "on", "at", "from", "with", "and", "the", "of", "or", "via",
    "ref", "inv", "no", "nr", "code", "payment", "odeme", "ödeme", "transfer", "trf",
    "fatura", "invoice", "settlement", "contract", "sözleşme", "sozlesme", "hesabina",
    "hesabına", "adina", "adına", "bedeli", "ucreti", "ücreti", "ltd", "pvt", "inc", "llc",
    "corp", "co", "a.s", "as", "sti", "şti", "ve", "sanayi", "ticaret", "limited", "sirketi",
    "şirketi", "anonim", "gmbh", "sa", "ag", "bv", "nv", "oy", "ab", "sendirian", "berhad", "tbk",
}


def _is_genuine_compact_match(compact_sim: float, core_c: str, n_exp: str) -> bool:
    """
    Compact fuzzy score'unun gerçek bir eşleşmeden mi yoksa boşluk kaldırmanın
    yan etkisinden (yanlış concat) mi oluştuğunu kontrol eder.

    Sorun:
        compact_normalize tüm boşlukları kaldırır. "MICRO SOFT" → "microsoft".
        Eğer "MICRO SOFT" Meksika tekstil firmasının adıysa, Microsoft'a yanlış
        1.0 compact fuzzy skoru üretilir.

    NOT: Bu guard yalnızca FUZZY SCORE ENRİCHMENT yolunda uygulanır.
    `exact_compact_match` için uygulanmaz; çünkü tam compact variant eşleşmesi
    zaten yanlış pozitifi doğal olarak engeller (açıklamada "corporation" yok →
    compact_matched_variant="microsoftcorporation" bulunamaz).

    Gerçek eşleşme kriterleri (birini sağlamak yeterli):
      1. core_cand metinde standalone token olarak geçiyor (\bcore_cand\b)
      2. core_cand, metindeki tek bir tokenin içinde substring olarak geçiyor
         (örn. "microsoftcorporation" tokenı içindeki "microsoft" gerçek evasion)

    Args:
        compact_sim: Compact fuzzy benzerlik skoru (0.0 - 1.0)
        core_c:      Adayın normalize edilmiş çekirdek adı (örn. "microsoft")
        n_exp:       Normalize edilmiş açıklama metni

    Returns:
        True  → Gerçek eşleşme, tam compact fuzzy skoru geçerli
        False → Yalnızca yanlış concat etkisi, partial credit (0.70) uygulanmalı
    """
    if compact_sim < 1.0:
        return True   # Kısmi eşleşmeler her zaman geçerli
    if not core_c:
        return False
    # Kriter 1: core_cand metinde standalone token (örn. "Microsoft Corporation odeme")
    if re.search(r'\b' + re.escape(core_c) + r'\b', n_exp):
        return True
    # Kriter 2: core_cand tek bir token içinde substring olarak geçiyor
    # (örn. "microsoftcorporation" → "microsoft" substring'i tek tokende mevcut)
    core_c_lower = core_c.casefold()
    for token in n_exp.split():
        clean_token = re.sub(r'[^\w]', '', token).casefold()
        if core_c_lower in clean_token and clean_token != core_c_lower:
            return True
    return False


def _is_unrelated_acronym_in_text(acronym: str, explanation: str, company_name: str) -> bool:
    norm_acc = normalize_text(acronym).strip()
    norm_exp = normalize_text(explanation).strip()
    norm_comp = normalize_text(company_name).strip()
    if not norm_acc or len(norm_acc) < 2:
        return False
    exp_words = norm_exp.split()
    comp_words = set(norm_comp.split())

    # 1. Kısaltmanın açıklamadaki kelimelerin baş harflerinden oluşup oluşmadığı (ör. BPS = Basınç Pompa Sistemi)
    for i in range(len(exp_words) - len(norm_acc) + 1):
        window = exp_words[i:i+len(norm_acc)]
        if ''.join(w[0] for w in window if w) == norm_acc:
            if not any(w in comp_words for w in window):
                return True

    # 2. Kısaltmanın proje/masraf/işlem/referans kodu olarak kullanılıp kullanılmadığı (ör. BPS proje kodu 2281)
    code_patterns = [
        r'\b' + re.escape(norm_acc) + r'\s+(?:proje|masraf|islem|sube|banka|referans|ref|fatura|takip)?\s*(?:kodu|kod|no|numarasi|id)\b',
        r'\b(?:proje|masraf|islem|sube|banka|referans|ref|fatura|takip)\s*(?:kodu|kod|no|numarasi|id)\s*[:=-]?\s*' + re.escape(norm_acc) + r'\b'
    ]
    for pat in code_patterns:
        if re.search(pat, norm_exp):
            return True
    return False


def _acronym_score(explanation: str, variant_name: str, original_company_name: str = None) -> float:
    """
    Basit kısaltma tespiti:
    1) Doğrudan bilinen kısaltma eşleşmesi (ör. THY, IBM, NST, ASELS, TUPRAS, BIM)
    2) Şirket adından oluşturulan baş harf kısaltmasının (acronym) açıklamada geçmesi
    3) Sistem tarafından türetilen kısaltmalı alias varyasyonlarının açıklamada geçmesi
    """
    if not explanation or not variant_name:
        return 0.0
    exp_tokens = set(normalize_text(explanation).split())
    norm_exp = normalize_text(explanation)

    names_to_check = [variant_name]
    if original_company_name and original_company_name != variant_name:
        names_to_check.append(original_company_name)

    for name in names_to_check:
        norm_name = normalize_text(name)
        if norm_name in _AMBIGUOUS_SHORT_NAMES:
            continue
        name_tokens = norm_name.split()

        # 1. Bilinen kısaltma veya adayın doğrudan kısa halinin eşleşmesi (ör. THY, IBM, NST, ASELS, TUPRAS, BIM)
        if norm_name and len(norm_name) >= 2:
            if len(name_tokens) == 1 and norm_name in exp_tokens:
                if not _is_unrelated_acronym_in_text(norm_name, explanation, name):
                    return 1.0
            elif len(name_tokens) > 1 and len(norm_name) <= 15:
                if re.search(r'\b' + re.escape(norm_name) + r'\b', norm_exp):
                    return 1.0

        # 2. Sistem tarafından üretilen baş harf kısaltması (acronym) kontrolü
        acronym = generate_acronym(name)
        if acronym and len(acronym) >= 2 and acronym not in _AMBIGUOUS_SHORT_NAMES and acronym in exp_tokens:
            if not _is_unrelated_acronym_in_text(acronym, explanation, name):
                return 1.0

        # 3. Sistem tarafından üretilen kısaltmalı alias varyasyonları (abbreviated aliases) kontrolü
        from src.utils.alias_utils import generate_abbreviated_aliases
        for abbr in generate_abbreviated_aliases(name, max_alias_count=15):
            if abbr and len(abbr) >= 2 and abbr not in _AMBIGUOUS_SHORT_NAMES and re.search(r'\b' + re.escape(abbr) + r'\b', norm_exp):
                if len(abbr.split()) == 1 and abbr not in exp_tokens:
                    continue
                if len(abbr.split()) == 1 and _is_unrelated_acronym_in_text(abbr, explanation, name):
                    continue
                return 1.0

    return 0.0


def _rule_score(norm_exp: str, variant_name: str) -> float:
    """
    Basit kural skoru: adayın tokenlerinin açıklamada bulunma oranı.
    Bitişik veya ayrı yazımlar için get_compact_core_name eşitliği kontrolü yapar.
    """
    v_tokens = [t for t in get_normalized_core_name(variant_name).split() if len(t) > 1 and t not in _RULE_STOPWORDS]
    if not v_tokens:
        v_tokens = [t for t in normalize_text(variant_name).split() if len(t) > 1 and t not in _RULE_STOPWORDS]
    if not v_tokens:
        return 0.0
    e_tokens = set(normalize_text(norm_exp).split())
    matched = sum(1 for t in v_tokens if t in e_tokens)
    score = matched / len(v_tokens)
    if score == 0.0:
        cq = get_compact_core_name(norm_exp)
        cc = get_compact_core_name(variant_name)
        if cc and len(cc) >= 4 and cq == cc:
            return 1.0
    return score


def _exact_name_score(norm_exp: str, variant_name: str) -> float:
    """EFT açıklaması adayın adını (tam veya core ad) tam olarak içeriyor mu?"""
    norm_cand  = normalize_text(variant_name)
    core_query = get_normalized_core_name(norm_exp)
    core_cand  = get_normalized_core_name(variant_name)

    if norm_cand and re.search(r'\b' + re.escape(norm_cand) + r'\b', norm_exp):
        return 1.0
    if core_cand and len(core_cand.replace(" ", "")) >= 4 and re.search(r'\b' + re.escape(core_cand) + r'\b', core_query):
        return 1.0
    cq = get_compact_core_name(norm_exp)
    cc = get_compact_core_name(variant_name)
    if cc and len(cc) >= 4 and cq == cc:
        return 1.0
    return 0.0


def _compute_token_fuzzy_score(query_core: str, cand_core: str) -> float:
    """
    EFT açıklaması ve aday şirketin öz adları (core names) arasındaki en yüksek kelime benzerliğini hesaplar.
    """
    if not query_core or not cand_core:
        return 0.0
        
    c_tokens = [w.casefold() for w in cand_core.split() if len(w) > 2 and w.casefold() not in _RULE_STOPWORDS]
    q_tokens = [w.casefold() for w in query_core.split() if len(w) > 2 and w.casefold() not in _RULE_STOPWORDS]
    
    if not c_tokens or not q_tokens:
        return difflib.SequenceMatcher(None, query_core.casefold(), cand_core.casefold()).ratio()
        
    total_max_sim = 0.0
    for ct in c_tokens:
        max_sim = max(difflib.SequenceMatcher(None, qt, ct).ratio() for qt in q_tokens)
        total_max_sim += max_sim
        
    return total_max_sim / len(c_tokens)


def _compute_compact_fuzzy_score(compact_query: str, compact_cand: str) -> float:
    """
    Bitişik veya ayrı yazılan kelimelerde (ve ek kelimeler içeren açıklamalarda) kayan pencere (sliding window)
    metoduyla en yüksek compact benzerlik skorunu hesaplar.
    """
    if not compact_query or not compact_cand:
        return 0.0
    if compact_cand in compact_query:
        return 1.0
    if len(compact_cand) < 4:
        return 0.0
        
    best_ratio = 0.0
    n = len(compact_cand)
    for w_len in range(max(3, n - 1), min(len(compact_query) + 1, n + 2)):
        for i in range(len(compact_query) - w_len + 1):
            sub = compact_query[i:i+w_len]
            ratio = difflib.SequenceMatcher(None, sub, compact_cand).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                if best_ratio == 1.0:
                    return 1.0
    return best_ratio


def build_score_features(
    norm_exp: str,
    cand: dict,
    extracted_entity: str = None,
    raw_explanation: str = None
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
    core_ext     = get_normalized_core_name(extracted_entity) if extracted_entity else ""
    
    compact_core_query = get_compact_core_name(raw_explanation or norm_exp)
    compact_core_cand  = get_compact_core_name(variant_name)

    compact_explanation = compact_normalize(raw_explanation) if raw_explanation else compact_normalize(norm_exp)
    compact_matched_variant = compact_normalize(variant_name)

    # ── exact_compact_match: compact substring eşleşmesi ───────────────────────────────────
    # compact_matched_variant (tam variant), compact_explanation içinde geçiyor mu?
    # NOT: _is_genuine_compact_match guard'a burada gerek yok.
    # "MICRO SOFT tekstil" → compact_matched_variant="microsoftcorporation" zaten
    # "microsofttekstilurunleritoptanalimi" içinde bulunmaz — naturally False.
    # Guard yalnızca aşağıdaki FUZZY ENRICHMENT yolunda gereklidir (core_cand substring'i).
    exact_compact_match = bool(compact_matched_variant and compact_matched_variant in compact_explanation)

    # Kural skoru eğer exact compact match ise 1.0 olmalı (fallback kural)
    base_rule_score = max(_rule_score(norm_exp, variant_name), _exact_name_score(norm_exp, variant_name))
    if exact_compact_match:
        base_rule_score = 1.0

    exact_normalized_match = bool(norm_exp == norm_cand and norm_exp)
    exact_core_match       = bool((core_query == core_cand and core_query) or (core_ext and core_ext == core_cand) or (compact_core_cand and len(compact_core_cand) >= 6 and compact_core_query == compact_core_cand))
    legal_suffix_only_diff = exact_core_match and not exact_normalized_match
    query_is_contained     = bool(norm_exp in norm_cand and norm_exp)
    cand_is_contained      = bool(norm_cand in norm_exp and norm_cand)
    query_token_count      = len(norm_exp.split())

    # ── Fuzzy score enrichment (yazım hatası, eksik harf, typo, bitişik/ayrı yazım) ────────
    if core_cand and core_query:
        core_fuzzy = _compute_token_fuzzy_score(core_query, core_cand)
        fuzzy_score = max(fuzzy_score, core_fuzzy)

    if compact_core_cand and len(compact_core_cand) >= 4:
        c_sim1 = _compute_compact_fuzzy_score(compact_core_query, compact_core_cand)
        c_sim2 = _compute_compact_fuzzy_score(compact_explanation, compact_core_cand)
        raw_compact_sim = max(c_sim1, c_sim2)
        # Guard: perfect compact sim (1.0) yalnızca boşluk kaldırmayla oluşuyorsa
        # (örn. "MICRO SOFT" → "microsoft") skor capped değere düşürülür.
        if raw_compact_sim >= 1.0 and not _is_genuine_compact_match(raw_compact_sim, core_cand, norm_exp):
            fuzzy_score = max(fuzzy_score, 0.70)  # Partial credit: high_fuzzy_boost tetiklenmez (threshold 0.82)
        else:
            fuzzy_score = max(fuzzy_score, raw_compact_sim)

    orig_name = cand.get("original_company_name", "")
    acronym_score_val = _acronym_score(norm_exp, variant_name, orig_name)

    # ── Leetspeak evasion detection & scoring enrichment ───────────────────────
    leet_evasion, leet_sim, leet_exp = check_leetspeak_evasion(raw_explanation or norm_exp, variant_name)
    if leet_evasion:
        fuzzy_score = max(fuzzy_score, leet_sim)
        base_rule_score = max(base_rule_score, _rule_score(leet_exp, variant_name), _exact_name_score(leet_exp, variant_name))
        if not exact_compact_match:
            leet_cand_compact = get_compact_core_name(normalize_leetspeak(variant_name))
            leet_query_compact = get_compact_core_name(normalize_leetspeak(raw_explanation or norm_exp))
            if leet_cand_compact and len(leet_cand_compact) >= 4 and leet_query_compact == leet_cand_compact:
                exact_compact_match = True
                base_rule_score = 1.0

    # ── Kısmi bilgi ve eksik bilgi tespiti (Substantial missing info) ──────────
    cand_tokens = [w for w in core_cand.split() if len(w) > 1 and w.casefold() not in _TOKEN_MATCH_STOPWORDS]
    query_tokens = [w for w in (core_ext or core_query).split() if len(w) > 1 and w.casefold() not in _TOKEN_MATCH_STOPWORDS]
    
    matched_tokens_count = 0
    for ct in cand_tokens:
        if any(qt == ct or (len(min(qt, ct, key=len)) >= 4 and (qt in ct or ct in qt)) or difflib.SequenceMatcher(None, qt, ct).ratio() >= 0.82 for qt in query_tokens):
            matched_tokens_count += 1
            
    missing_tokens_count = len(cand_tokens) - matched_tokens_count
    is_exact_like = exact_normalized_match or exact_core_match or exact_compact_match or (acronym_score_val >= 1.0) or (base_rule_score >= 1.0)
    
    substantial_missing_info = (
        not is_exact_like
        and len(cand_tokens) >= 2
        and matched_tokens_count >= 1
        and missing_tokens_count >= 1
        and (missing_tokens_count / len(cand_tokens)) >= 0.40
    )

    scores_dict = {
        "fuzzy_score":    fuzzy_score,
        "vector_score":   vector_score,
        "acronym_score":  acronym_score_val,
        "rule_score":     base_rule_score,
        "reranker_score": norm_reranker,
        "query_token_count": query_token_count,
        "exact_normalized_match": exact_normalized_match,
        "exact_core_match": exact_core_match,
        "legal_suffix_only_difference": legal_suffix_only_diff,
        "query_is_contained_in_candidate": query_is_contained,
        "candidate_is_contained_in_query": cand_is_contained,
        "consonant_match": is_consonant_match(core_query, core_cand),
        "exact_compact_match": exact_compact_match,
        "compact_explanation": compact_explanation,
        "compact_matched_variant": compact_matched_variant,
        "leetspeak_evasion_detected": leet_evasion,
        "substantial_missing_info": substantial_missing_info,
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
                difflib.SequenceMatcher(None, extracted_entity.lower(), core_cand.lower()).ratio(),
                _compute_token_fuzzy_score(extracted_entity.lower(), core_cand.lower())
            )
        
        scores_dict["fuzzy_score"] = max(scores_dict["fuzzy_score"], fuzzy_ext)

        if is_consonant_match(extracted_entity, core_cand):
            scores_dict["consonant_match"] = True

        scores_dict["acronym_score"] = max(
            scores_dict["acronym_score"],
            _acronym_score(extracted_entity, variant_name, orig_name)
        )
        scores_dict["rule_score"] = max(
            scores_dict["rule_score"],
            max(
                _rule_score(extracted_entity, variant_name),
                _exact_name_score(extracted_entity, variant_name)
            )
        )

        leet_ext_evasion, leet_ext_sim, leet_ext_norm = check_leetspeak_evasion(extracted_entity, variant_name)
        if leet_ext_evasion:
            scores_dict["leetspeak_evasion_detected"] = True
            scores_dict["fuzzy_score"] = max(scores_dict["fuzzy_score"], leet_ext_sim)
            scores_dict["rule_score"] = max(
                scores_dict["rule_score"],
                max(
                    _rule_score(leet_ext_norm, variant_name),
                    _exact_name_score(leet_ext_norm, variant_name)
                )
            )

    if scores_dict.get("acronym_score", 0.0) >= 1.0 or scores_dict.get("rule_score", 0.0) >= 1.0 or scores_dict.get("exact_core_match") or scores_dict.get("exact_normalized_match") or scores_dict.get("exact_compact_match"):
        scores_dict["substantial_missing_info"] = False

    return scores_dict
