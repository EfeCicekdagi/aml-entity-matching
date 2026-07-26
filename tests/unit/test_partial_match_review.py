"""
test_partial_match_review.py — Kısmi bilgi ile aday üretimi ve eksik bilgide analist incelemesi testleri.
"""

import pytest
from src.scoring.score_features import build_score_features
from src.scoring.final_scorer import FinalScorer
from src.scoring.reason_codes import ReasonCode


class TestPartialMatchReview:
    def test_partial_match_sets_substantial_missing_info(self):
        query = "North Star"
        candidate_name = "North Star Maritime Logistics Export Import Trading Company Private Limited"
        cand = {
            "variant_name": candidate_name,
            "trgm_score": 0.40,
            "original_company_name": candidate_name,
        }
        scores = build_score_features(query, cand, raw_explanation=query)
        assert scores["substantial_missing_info"] is True

    def test_partial_match_sent_to_analyst_review_instead_of_high_risk(self):
        query = "North Star"
        candidate_name = "North Star Maritime Logistics Export Import Trading Company Private Limited"
        cand = {
            "variant_name": candidate_name,
            "trgm_score": 0.40,
            "original_company_name": candidate_name,
        }
        scores = build_score_features(query, cand, raw_explanation=query)
        # Yapay olarak yüksek bir vektör/reranker skoru verelim ki normalde HIGH alert verecek olsun
        scores["reranker_score"] = 0.85
        scores["vector_score"] = 0.85

        scorer = FinalScorer(repository=None)
        final_score, match_reason, reason_codes = scorer.calculate_final_score(scores, alias_confidence=1.0)
        risk_level = scorer.assign_risk_level(final_score)

        assert 0.60 <= final_score <= 0.68, f"Beklenen orta risk bandı skoru, alınan: {final_score}"
        assert risk_level == "MEDIUM", f"Beklenen risk seviyesi MEDIUM (analist incelemesi), alınan: {risk_level}"
        assert match_reason == "PARTIAL_MATCH_REQUIRES_REVIEW"
        assert ReasonCode.PARTIAL_MATCH_REQUIRES_REVIEW in reason_codes

    def test_exact_or_acronym_not_flagged_as_missing_info(self):
        query = "North Star Trading Limited"
        candidate_name = "North Star Trading Limited"
        cand = {
            "variant_name": candidate_name,
            "trgm_score": 1.0,
            "original_company_name": candidate_name,
        }
        scores = build_score_features(query, cand, raw_explanation=query)
        assert scores["substantial_missing_info"] is False
