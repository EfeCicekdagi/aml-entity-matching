import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.text_utils import normalize_text, get_normalized_core_name
from src.scoring.final_scorer import FinalScorer

class MockRepository:
    def get_connection(self): return None
    def release_connection(self, conn): pass


def _make_scores(query: str, candidate: str, fuzzy: float, vector: float, reranker: float, rule: float = 0.0):
    """Test için tutarlı skor sözlüğü üretir."""
    norm_query = normalize_text(query)
    norm_cand = normalize_text(candidate)
    core_query = get_normalized_core_name(norm_query)
    core_cand = get_normalized_core_name(norm_cand)
    exact_normalized_match = (norm_query == norm_cand and bool(norm_query))
    exact_core_match = (core_query == core_cand and bool(core_query))
    legal_suffix_only_difference = exact_core_match and not exact_normalized_match
    return {
        "fuzzy_score": fuzzy,
        "vector_score": vector,
        "acronym_score": 0.0,
        "rule_score": rule,
        "reranker_score": reranker,
        "query_token_count": len(norm_query.split()),
        "exact_normalized_match": exact_normalized_match,
        "exact_core_match": exact_core_match,
        "legal_suffix_only_difference": legal_suffix_only_difference,
        "consonant_match": False,
        "_query_str": norm_query,
        "_variant_str": norm_cand,
    }


def test_calculate_final_score_returns_three_values():
    """calculate_final_score 3 değer döndürmeli: (score, reason, reason_codes)."""
    scorer = FinalScorer(MockRepository())
    scores = _make_scores("Apple", "Apple Inc", 1.0, 0.9, 0.8)
    result = scorer.calculate_final_score(scores)
    assert isinstance(result, tuple), "Sonuç tuple olmalı"
    assert len(result) == 3, f"3 değer bekleniyor, {len(result)} geldi"
    final_score, match_reason, reason_codes = result
    assert isinstance(final_score, float), "final_score float olmalı"
    assert isinstance(match_reason, str), "match_reason str olmalı"
    assert isinstance(reason_codes, list), "reason_codes list olmalı"


def test_high_confidence_match_gives_high_score():
    """Yüksek fuzzy+reranker skoru → yüksek final skor."""
    scorer = FinalScorer(MockRepository())
    scorer.weights = {
        "fuzzy_weight": 0.20, "vector_weight": 0.20, "reranker_weight": 0.60,
        "acronym_weight": 0.0, "rule_weight": 0.0
    }
    scores = _make_scores("North Star Trading Limited", "North Star Trading Limited",
                          fuzzy=1.0, vector=0.95, reranker=0.90)
    final_score, match_reason, reason_codes = scorer.calculate_final_score(scores)
    assert final_score >= 0.85, f"Yüksek skorlu eşleşme için final_score >= 0.85 bekleniyor, got: {final_score}"
    assert reason_codes, "reason_codes boş olmamalı"


def test_low_confidence_mismatch_gives_low_score():
    """Farklı şirketler → düşük final skor."""
    scorer = FinalScorer(MockRepository())
    scorer.weights = {
        "fuzzy_weight": 0.20, "vector_weight": 0.20, "reranker_weight": 0.60,
        "acronym_weight": 0.0, "rule_weight": 0.0
    }
    scores = _make_scores("Apple", "Google LLC", fuzzy=0.20, vector=0.50, reranker=0.01)
    final_score, match_reason, reason_codes = scorer.calculate_final_score(scores)
    assert final_score < 0.62, f"Düşük skorlu uyuşmazlık için final_score < 0.62 bekleniyor, got: {final_score}"
    assert match_reason == "LOW_CONFIDENCE", f"Beklenen: LOW_CONFIDENCE, got: {match_reason}"


def test_reason_codes_not_empty():
    """Herhangi bir skorlamada reason_codes listesi boş olmamalı."""
    scorer = FinalScorer(MockRepository())
    scores = _make_scores("Oracle", "Oracle Corporation", fuzzy=0.8, vector=0.85, reranker=0.5)
    final_score, match_reason, reason_codes = scorer.calculate_final_score(scores)
    assert len(reason_codes) > 0, "reason_codes en az 1 eleman içermeli"


def test_short_single_token_name_no_exact_override():
    """
    Tek tokenli kısa isim (Apple) exact override almamalı.
    FinalScorer._is_safe_exact_override → False döner, skor 1.0'a çıkmamalı.
    """
    scorer = FinalScorer(MockRepository())
    scorer.weights = {
        "fuzzy_weight": 0.20, "vector_weight": 0.20, "reranker_weight": 0.60,
        "acronym_weight": 0.0, "rule_weight": 0.0
    }
    # Apple vs Apple Inc → exact_core_match = True ama tek token → override güvensiz
    scores = {
        "fuzzy_score": 1.0, "vector_score": 0.90, "acronym_score": 0.0,
        "rule_score": 0.0, "reranker_score": 0.30,
        "query_token_count": 1,
        "exact_normalized_match": False,
        "exact_core_match": True,
        "legal_suffix_only_difference": True,
        "consonant_match": False,
        "_query_str": "apple",
        "_variant_str": "apple inc",
    }
    final_score, match_reason, reason_codes = scorer.calculate_final_score(scores)
    # Kısa sorgu ağırlıkları devreye girer (query_token_count <= 2)
    # Exact override güvenli değil → 1.0 olmamalı
    assert final_score < 1.0, f"Tek token 'apple' için final_score 1.0 olmamalı, got: {final_score}"


def test_long_multitoken_exact_match_gets_high_score():
    """Çok tokenli uzun isim exact match → 0.95+ beklenir."""
    scorer = FinalScorer(MockRepository())
    scorer.weights = {
        "fuzzy_weight": 0.20, "vector_weight": 0.20, "reranker_weight": 0.60,
        "acronym_weight": 0.0, "rule_weight": 0.0
    }
    scores = {
        "fuzzy_score": 1.0, "vector_score": 0.95, "acronym_score": 0.0,
        "rule_score": 0.0, "reranker_score": 0.85,
        "query_token_count": 4,
        "exact_normalized_match": False,
        "exact_core_match": True,
        "legal_suffix_only_difference": True,
        "consonant_match": False,
        "_query_str": "north star trading limited",
        "_variant_str": "north star trading company limited",
    }
    final_score, match_reason, reason_codes = scorer.calculate_final_score(scores)
    assert final_score >= 0.85, f"Uzun isim legal suffix farkı için >= 0.85 bekleniyor, got: {final_score}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
