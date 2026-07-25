"""
test_typo_fuzzy_matching.py — Küçük yazım hataları, eksik harfler ve harf yer değiştirmeleri
(typo / transposition / missing character) senaryoları için fuzzy matching testleri.
"""

import pytest
from src.scoring.score_features import _compute_token_fuzzy_score, build_score_features
from src.scoring.final_scorer import FinalScorer, ReasonCode
from src.models.entity_extractor import EntityExtractor, ExtractionResult


@pytest.fixture
def scorer():
    """DB olmaksızın varsayılan ağırlıklarla çalışan FinalScorer örneği."""
    return FinalScorer(repository=None)


@pytest.fixture
def extractor():
    """Varsayılan ayarlarla çalışan EntityExtractor örneği."""
    return EntityExtractor()


class TestTypoFuzzyMatching:
    """Kullanıcı tarafından belirtilen 8 yazım hatası senaryosunun testi."""

    @pytest.mark.parametrize("query, candidate, expected_min_score", [
        ("Microsft Corporation lisans bedeli", "Microsoft Corporation", 0.90),
        ("Microsofft Corp ödeme", "Microsoft Corporation", 0.90),
        ("Microsfot Corporation", "Microsoft Corporation", 0.88),
        ("Indaforensic Services Pvt Ltd", "Indiaforensic Services Pvt Ltd", 0.95),
        ("Indiaforensik Services", "Indiaforensic Services Pvt Ltd", 0.90),
        ("Finspaire Solutions Private Limited", "Finspire Solutions Private Limited", 0.90),
        ("Mehar Entertaiment Pvt Ltd", "Mehar Entertainment Pvt Ltd", 0.95),
        ("Amazn Technologies Inc", "Amazon Technologies Inc", 0.90),
    ])
    def test_token_fuzzy_score_high_for_typos(self, query, candidate, expected_min_score):
        """Token ve öz ad (core name) bazlı fuzzy benzerliğin yüksek çıktığını doğrular."""
        from src.utils.text_utils import get_normalized_core_name
        core_q = get_normalized_core_name(query)
        core_c = get_normalized_core_name(candidate)
        score = _compute_token_fuzzy_score(core_q, core_c)
        assert score >= expected_min_score, f"'{query}' ile '{candidate}' arası fuzzy skor ({score:.3f}) beklenen ({expected_min_score}) altında kaldı!"

    @pytest.mark.parametrize("query, variant_name, is_safe_expected", [
        ("Microsft Corporation lisans bedeli", "Microsoft Corporation", True),
        ("Indaforensic Services Pvt Ltd", "Indiaforensic Services Pvt Ltd", True),
        ("Finspaire Solutions Private Limited", "Finspire Solutions Private Limited", True),
        ("Mehar Entertaiment Pvt Ltd", "Mehar Entertainment Pvt Ltd", True),
        ("Amazn Technologies Inc", "Amazon Technologies Inc", False),  # 'amazon' tek kelime / genel ad koruması
    ])
    def test_fuzzy_boost_in_final_scorer_produces_high_risk(self, scorer, query, variant_name, is_safe_expected):
        """Fuzzy enrichment ve final scorer bütünselliğinde yazım hatalarının Yüksek Risk ürettiğini doğrular."""
        cand = {
            "variant_name": variant_name,
            "trgm_score": 0.75,  # Postgres trgm skoru (eksik harfte biraz düşebilir)
            "vector_score": 0.65,
            "normalized_reranker_score": 0.50
        }
        
        scores = build_score_features(query, cand)
        # Fuzzy enrichment sayesinde skor 0.88-0.98 aralığına yükselmiş olmalı
        assert scores["fuzzy_score"] >= 0.88
        
        final_score, match_reason, reason_codes = scorer.calculate_final_score(scores)
        
        # Yüksek risk eşiğini (>= 0.70) geçmeli
        assert final_score >= 0.70, f"'{query}' için nihai skor ({final_score:.3f}) yüksek risk eşiği 0.70'in altında!"
        assert ReasonCode.HIGH_FUZZY_SIMILARITY in reason_codes
        
        if is_safe_expected:
            assert final_score >= 0.88
            assert match_reason == "HIGH_FUZZY_MATCH"

    def test_entity_extractor_fuzzy_candidate_fallback(self, extractor):
        """Katman 3 (Candidate-supported) içinde yazım hatalarında FUZZY_CANDIDATE_MATCH çalıştığını doğrular."""
        text = "transfer to indaforensic services pvt ltd for annual maintenance"
        candidates = [
            {"variant_name": "Indiaforensic Services Pvt Ltd", "trgm_score": 0.85}
        ]
        
        res = extractor._extract_via_candidates(text, candidates)
        assert res is not None
        assert res.extracted_entity == "Indiaforensic Services Pvt Ltd"
        assert res.extraction_method == "FUZZY_CANDIDATE_MATCH"
        assert res.entity_extraction_status == "EXTRACTED"
