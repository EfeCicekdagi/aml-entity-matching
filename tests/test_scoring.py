import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.text_utils import normalize_text, get_normalized_core_name
from src.scoring.final_scorer import FinalScorer

class MockRepository:
    def get_connection(self): return None
    def release_connection(self, conn): pass

def test_scoring_cases():
    scorer = FinalScorer(MockRepository())
    
    # Overriding default weights just for consistent testing
    scorer.weights = {
        "fuzzy_weight": 0.30,
        "vector_weight": 0.20,
        "acronym_weight": 0.0,
        "rule_weight": 0.20,
        "reranker_weight": 0.30
    }
    
    cases = [
        ("Apple", "Apple Inc", "MATCH", "EXACT_CORE_MATCH"),
        ("Apple", "Apple Incorporated", "MATCH", "EXACT_CORE_MATCH"),
        ("APPLE INC.", "Apple Inc", "MATCH", "EXACT_NORMALIZED_MATCH"),
        ("Microsoft", "Microsoft Corporation", "MATCH", "EXACT_CORE_MATCH"),
        ("Google", "Google LLC", "MATCH", "EXACT_CORE_MATCH"),
        ("IBM", "IBM Corporation", "MATCH", "EXACT_CORE_MATCH"),
        ("Apple", "Google LLC", "NO_MATCH", "LOW_CONFIDENCE"),
        ("Apple", "Microsoft Corporation", "NO_MATCH", "LOW_CONFIDENCE"),
        ("Oracle", "Oracle Corporation", "MATCH", "EXACT_CORE_MATCH")
    ]
    
    for query, candidate, expected_risk, expected_reason in cases:
        norm_query = normalize_text(query)
        norm_cand = normalize_text(candidate)
        core_query = get_normalized_core_name(norm_query)
        core_cand = get_normalized_core_name(norm_cand)
        
        query_token_count = len(norm_query.split())
        exact_normalized_match = (norm_query == norm_cand and bool(norm_query))
        exact_core_match = (core_query == core_cand and bool(core_query))
        legal_suffix_only_difference = exact_core_match and not exact_normalized_match
        
        # Mock some reasonable ML scores
        if exact_core_match:
            fuzzy = 1.0
            vector = 0.90 # high vector similarity for same core
            reranker = 0.30 # reranker might fail on short ones
            rule = 1.0
        else:
            fuzzy = 0.20
            vector = 0.50 # e.g. apple vs google semantic vector might be medium, but let's say 0.50
            reranker = 0.01
            rule = 0.0

        scores_dict = {
            "fuzzy_score": fuzzy,
            "vector_score": vector,
            "acronym_score": 0.0,
            "rule_score": rule,
            "reranker_score": reranker,
            "query_token_count": query_token_count,
            "exact_normalized_match": exact_normalized_match,
            "exact_core_match": exact_core_match,
            "legal_suffix_only_difference": legal_suffix_only_difference
        }
        
        final_score, match_reason = scorer.calculate_final_score(scores_dict)
        risk_level = scorer.assign_risk_level(final_score)
        
        print(f"[{query} <-> {candidate}]")
        print(f"  norm_query: '{norm_query}', core_query: '{core_query}'")
        print(f"  norm_cand: '{norm_cand}', core_cand: '{core_cand}'")
        print(f"  final_score: {final_score:.2f}, risk: {risk_level}, reason: {match_reason}")
        print("-" * 40)
        
        assert match_reason == expected_reason, f"Failed on {query} ↔ {candidate}. Expected reason: {expected_reason}, got: {match_reason}"
        if expected_risk == "MATCH":
            assert final_score >= 0.85, f"Failed on {query} ↔ {candidate}. Expected MATCH (>=0.85), got: {final_score}"

if __name__ == "__main__":
    test_scoring_cases()
