"""
test_acronym_abbreviation_matching.py — Bilinen ve sistem tarafından üretilen kısaltmaların (acronym & abbreviation)
özellik çıkarımı, varlık çıkarımı ve nihai skorlama katmanlarında doğru şekilde yakalandığını test eder.
"""

import pytest
from src.scoring.score_features import _acronym_score, build_score_features
from src.models.entity_extractor import EntityExtractor
from src.scoring.final_scorer import FinalScorer
from src.scoring.reason_codes import ReasonCode


class TestAcronymAbbreviationMatching:

    def test_acronym_score_with_system_generated_acronym(self):
        """Sistem tarafından üretilen baş harf kısaltmasının (öğr. North Star Trading Limited -> nst) yakalanması."""
        explanation = "payment for nst invoice 1234"
        variant_name = "North Star Trading Limited"
        score = _acronym_score(explanation, variant_name)
        assert score == 1.0, "Sistem tarafından üretilen nst kısaltması 1.0 skoru üretmelidir."

    def test_acronym_score_with_system_generated_abbreviated_alias(self):
        """Sistem tarafından üretilen kısaltmalı varyasyonların (öğr. North Star Trading Limited -> north star trdng ltd) yakalanması."""
        explanation = "payment for north star trdng ltd shipment"
        variant_name = "North Star Trading Limited"
        score = _acronym_score(explanation, variant_name)
        assert score == 1.0, "Kısaltmalı kelime varyasyonları (trdng ltd) 1.0 skoru üretmelidir."

    def test_acronym_score_with_known_abbreviation(self):
        """Veritabanındaki bilinen kısa kısaltmaların (öğr. THY, IBM, ASELS, BIM) yakalanması."""
        explanation = "bilet odeme thy istanbul genel mudurluk"
        variant_name = "THY"
        score = _acronym_score(explanation, variant_name)
        assert score == 1.0, "Bilinen THY kısaltması açıklamada kelime olarak geçtiğinde 1.0 skoru üretmelidir."

    def test_acronym_score_with_original_company_name_fallback(self):
        """Aday varyant adı farklı olsa bile original_company_name üzerinden kısaltma yakalanması."""
        explanation = "transfer to nst odeme"
        variant_name = "North Star Trading"
        orig_name = "North Star Trading Limited"
        score = _acronym_score(explanation, variant_name, orig_name)
        assert score == 1.0, "original_company_name üzerinden üretilen kısaltma yakalanmalıdır."

    def test_entity_extractor_extracts_via_system_acronym(self):
        """EntityExtractor'ın sistem kısaltması (nst) üzerinden doğru şirketi çıkarması."""
        extractor = EntityExtractor({"min_entity_length": 3})
        text = "transfer to nst odeme"
        candidates = [
            {"variant_name": "North Star Trading Limited", "original_company_name": "North Star Trading Limited"}
        ]
        result = extractor.extract(text, candidates)
        assert result.extracted_entity == "North Star Trading Limited"
        assert result.extraction_method == "ACRONYM_SUPPORTED"
        assert result.extraction_confidence == 0.75

    def test_entity_extractor_extracts_via_abbreviated_alias(self):
        """EntityExtractor'ın kısaltmalı alias (trdng ltd) üzerinden doğru şirketi çıkarması."""
        extractor = EntityExtractor({"min_entity_length": 3})
        text = "transfer to north star trdng ltd invoice"
        candidates = [
            {"variant_name": "North Star Trading Limited", "original_company_name": "North Star Trading Limited"}
        ]
        result = extractor.extract(text, candidates)
        assert result.extracted_entity == "North Star Trading Limited"
        assert result.extraction_method == "ACRONYM_SUPPORTED"

    def test_build_score_features_sets_acronym_score(self):
        """build_score_features fonksiyonunun acronym_score özelliğini doğru ataması."""
        explanation = "payment for nst invoice"
        cand = {
            "variant_name": "North Star Trading Limited",
            "original_company_name": "North Star Trading Limited",
            "trgm_score": 0.1,
            "vector_score": 0.2
        }
        features = build_score_features(explanation, cand)
        assert features["acronym_score"] == 1.0

    def test_final_scorer_boosts_high_risk_on_acronym_match(self):
        """FinalScorer'ın kısaltma eşleşmesi (acronym_score=1.0) olduğunda skoru yüksek riske yükseltmesi."""
        scorer = FinalScorer(repository=None)
        scores_dict = {
            "fuzzy_score": 0.15,
            "vector_score": 0.20,
            "acronym_score": 1.0,
            "rule_score": 0.0,
            "reranker_score": 0.10,
            "query_token_count": 4,
            "exact_normalized_match": False,
            "exact_core_match": False,
            "legal_suffix_only_difference": False,
            "_query_str": "payment for nst invoice",
            "_variant_str": "North Star Trading Limited"
        }
        final_score, match_reason, reason_codes = scorer.calculate_final_score(scores_dict, alias_confidence=1.0)
        assert final_score >= 0.88, f"Kısaltma eşleşmesinde final_score >= 0.88 olmalıdır, alınan: {final_score}"
        assert match_reason == "ACRONYM_MATCH"
        assert ReasonCode.ACRONYM_MATCH in reason_codes
