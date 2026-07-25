"""
test_concatenated_split_words.py — Kelimelerin bitişik veya ayrı yazılması 
(concatenated / split words / evasion / obfuscation) senaryolarının testi.
"""

import pytest
from src.utils.text_utils import get_compact_core_name, compact_normalize
from src.scoring.score_features import build_score_features
from src.scoring.final_scorer import FinalScorer, ReasonCode
from src.models.entity_extractor import EntityExtractor


@pytest.fixture
def scorer():
    """DB olmaksızın çalışan varsayılan FinalScorer örneği."""
    return FinalScorer(repository=None)


@pytest.fixture
def extractor():
    """EntityExtractor örneği."""
    return EntityExtractor()


class TestConcatenatedSplitWords:
    """Kelimelerin bitişik veya ayrı yazılması senaryolarının testleri."""

    @pytest.mark.parametrize("query, candidate_name, expected_core", [
        ("microsoftcorporation", "Microsoft Corporation", "microsoft"),
        ("micro soft corporation", "Microsoft Corporation", "microsoft"),
        ("indiaforensicservices", "Indiaforensic Services Pvt Ltd", "indiaforensic"),
        ("india forensic services", "Indiaforensic Services Pvt Ltd", "indiaforensic"),
        ("finspiresolutions", "Finspire Solutions Private Limited", "finspire"),
        ("fin spire solutions", "Finspire Solutions Private Limited", "finspire"),
        ("meharentertainment", "Mehar Entertainment Pvt Ltd", "mehar"),
        ("mehar enter tainment", "Mehar Entertainment Pvt Ltd", "mehar"),
        ("amazontechnologies", "Amazon Technologies Inc", "amazon"),
        ("ama zon technologies", "Amazon Technologies Inc", "amazon"),
    ])
    def test_compact_core_name_invariance(self, query, candidate_name, expected_core):
        """Bitişik veya ayrı yazımlarda saf alfanümerik compact core ismin değişmezliğini doğrular."""
        q_core = get_compact_core_name(query)
        c_core = get_compact_core_name(candidate_name)
        
        assert q_core == expected_core, f"'{query}' için compact core '{q_core}' çıkışı beklenen '{expected_core}' ile uyuşmadı!"
        assert c_core == expected_core, f"'{candidate_name}' için compact core '{c_core}' çıkışı beklenen '{expected_core}' ile uyuşmadı!"
        assert q_core == c_core, "Sorgu ve aday compact core değerleri tam eşit olmalı!"

    @pytest.mark.parametrize("query, variant_name, expect_exact_compact", [
        ("transfer to microsoftcorporation odeme", "Microsoft Corporation", True),
        ("micro soft corporation lisans bedeli", "Microsoft Corporation", True),
        ("indiaforensicservices pvt ltd contract", "Indiaforensic Services Pvt Ltd", True),
        ("india forensic services pvt ltd", "Indiaforensic Services Pvt Ltd", True),
        ("finspiresolutions private limited", "Finspire Solutions Private Limited", True),
        ("fin spire solutions private limited", "Finspire Solutions Private Limited", True),
    ])
    def test_build_score_features_and_final_scorer_high_risk(self, scorer, query, variant_name, expect_exact_compact):
        """Bitişik veya ayrı yazımların kural/fuzzy özelliklerini yükseltip Yüksek Risk ürettiğini doğrular."""
        cand = {
            "variant_name": variant_name,
            "trgm_score": 0.65,
            "vector_score": 0.70,
            "normalized_reranker_score": 0.50
        }
        
        scores = build_score_features(query, cand, raw_explanation=query)
        
        # Compact core eşleştiğinde fuzzy skoru 1.0 (veya çok yüksek) olmalı
        assert scores["fuzzy_score"] >= 0.95
        
        # Eğer yasal ekler de uyumluysa exact_compact_match tetiklenmeli
        if expect_exact_compact:
            assert scores["exact_compact_match"] is True
            assert scores["rule_score"] == 1.0
            
        final_score, match_reason, reason_codes = scorer.calculate_final_score(scores)
        
        # Nihai skor her koşulda Yüksek Risk threshold'unu (>= 0.70) geçmeli
        assert final_score >= 0.70, f"'{query}' için nihai skor ({final_score:.3f}) Yüksek Risk eşiğinin altında kaldı!"
        assert any(c in reason_codes for c in [
            ReasonCode.EXACT_COMPACT_MATCH,
            ReasonCode.HIGH_FUZZY_SIMILARITY,
            ReasonCode.EXACT_OFFICIAL_NAME
        ]), f"Beklenen risk reason code'larından hiçbiri alınmadı: {reason_codes}"

    def test_entity_extractor_extracts_concatenated_and_split(self, extractor):
        """Katman 3 Varlık Çıkarıcının bitişik ve ayrı yazılmış adayları bulabildiğini doğrular."""
        # 1. Bitişik yazım (concatenated)
        text1 = "invoice payment to indiaforensicservicespvtltd account"
        cands1 = [{"variant_name": "Indiaforensic Services Pvt Ltd", "trgm_score": 0.7}]
        res1 = extractor._extract_via_candidates(text1, cands1)
        assert res1 is not None
        assert res1.extracted_entity == "Indiaforensic Services Pvt Ltd"
        
        # 2. Ayrık yazım (split)
        text2 = "transfer due to fin spire solutions private limited"
        cands2 = [{"variant_name": "Finspire Solutions Private Limited", "trgm_score": 0.7}]
        res2 = extractor._extract_via_candidates(text2, cands2)
        assert res2 is not None
        assert res2.extracted_entity == "Finspire Solutions Private Limited"
