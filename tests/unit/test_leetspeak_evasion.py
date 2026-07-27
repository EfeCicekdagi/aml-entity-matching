"""
test_leetspeak_evasion.py — Leetspeak (harf yerine rakam/sembol kullanımı) gizleme 
girişimlerinin tespiti, normalizasyonu ve yüksek risk skoruna dönüştürülmesini test eder.
"""

import pytest
from src.utils.text_utils import normalize_leetspeak, check_leetspeak_evasion, get_leetspeak_compact_core_name, normalize_for_matching
from src.scoring.score_features import build_score_features
from src.scoring.final_scorer import FinalScorer, ReasonCode
from src.models.entity_extractor import EntityExtractor


class TestLeetspeakEvasion:

    @pytest.mark.parametrize("leet_text,expected_normal", [
        ("M!cr0s0ft C0rp0r4t!0n", "microsoft corporation"),
        ("4ppl3 Inc", "apple inc"),
        ("0r4cl3 Systems", "oracle systems"),
        ("F!nsp!r3 S0lut!0n5", "finspire solutions"),
        ("1nd!4f0r3ns!c", "indiaforensic"),
        ("N0rmal T3xt W!th L33t", "normal text with leet"),
    ])
    def test_normalize_leetspeak(self, leet_text, expected_normal):
        assert normalize_leetspeak(leet_text) == expected_normal

    def test_check_leetspeak_evasion_positive(self):
        query = "transfer to m1cr0s0ft c0rp0r4t!0n odeme"
        candidate = "Microsoft Corporation"
        evasion_detected, score, leet_query = check_leetspeak_evasion(query, candidate)
        assert evasion_detected is True
        assert score > 0.80
        assert "microsoft corporation" in leet_query

    def test_check_leetspeak_evasion_negative_normal_numbers(self):
        query = "payment for invoice 20240501"
        candidate = "Microsoft Corporation"
        evasion_detected, score, leet_query = check_leetspeak_evasion(query, candidate)
        assert evasion_detected is False

    @pytest.mark.parametrize("query,candidate_name", [
        ("odeme m1cr0s0ft c0rp0r4t10n lisans", "Microsoft Corporation"),
        ("payment to 4ppl3 inc contract", "Apple Inc"),
        ("transfer 0r4cl3 systems pvt ltd", "Oracle Systems Pvt Ltd"),
        ("invoice f!nsp!r3 s0lut!0n5", "Finspire Solutions Private Limited"),
    ])
    def test_build_score_features_sets_leetspeak_evasion(self, query, candidate_name):
        cand = {
            "variant_name": candidate_name,
            "trgm_score": 0.40,
            "original_company_name": candidate_name,
        }
        scores = build_score_features(query, cand, raw_explanation=query)
        assert scores["leetspeak_evasion_detected"] is True
        assert scores["fuzzy_score"] >= 0.85

    def test_final_scorer_boosts_to_high_risk_on_leetspeak(self):
        query = "transfer to m1cr0s0ft c0rp0r4t10n odeme"
        candidate_name = "Microsoft Corporation"
        cand = {
            "variant_name": candidate_name,
            "trgm_score": 0.40,
            "original_company_name": candidate_name,
        }
        scores = build_score_features(query, cand, raw_explanation=query)
        
        scorer = FinalScorer(repository=None)
        final_score, match_reason, reason_codes = scorer.calculate_final_score(scores, alias_confidence=1.0)
        
        assert final_score >= 0.88
        assert match_reason == "LEETSPEAK_EVASION_DETECTED"
        assert ReasonCode.LEETSPEAK_EVASION in reason_codes

    def test_entity_extractor_extracts_leetspeak(self):
        extractor = EntityExtractor()
        text = "Lütfen m1cr0s0ft c0rp0r4t10n hesabına havale yapın."
        candidates = [
            {"variant_name": "Microsoft Corporation", "original_company_name": "Microsoft Corporation"},
            {"variant_name": "Apple Inc", "original_company_name": "Apple Inc"}
        ]
        
        result = extractor.extract(text, candidates=candidates)
        assert result.extracted_entity == "Microsoft Corporation"
        assert result.entity_extraction_status == "EXTRACTED"

    def test_normalize_for_matching_unified(self):
        # 1. Kiril homoglyph testi ('о' Kiril karakteri)
        assert normalize_for_matching("Goоgle LLC") == "google llc"
        # 2. Leetspeak testi
        assert normalize_for_matching("1BM C0rp0rat10n") == "ibm corporation"
        # 3. Saf rakam koruması (referans numarası korunmalı)
        assert normalize_for_matching("REF-326121") == "ref-326121"
