"""
test_word_order_invariance.py — Kelime sırası değişse bile aday üretimi, varlık çıkarımı ve skorlamanın
başarıyla çalıştığını doğrulayan birim testler.
"""

import unittest
from src.scoring.score_features import (
    _rule_score,
    _exact_name_score,
    _compute_token_fuzzy_score,
    build_score_features
)
from src.models.entity_extractor import EntityExtractor
from unittest.mock import MagicMock


class TestWordOrderInvariance(unittest.TestCase):
    """Kelime sırası bağımsızlığının (order invariance) test edilmesi."""

    def test_rule_score_is_order_invariant(self):
        """Kelimeler ters sıralansa bile kural skoru 1.0 olmalı."""
        score1 = _rule_score("corporation microsoft transfer odeme", "Microsoft Corporation")
        self.assertEqual(score1, 1.0)
        
        score2 = _rule_score("services indiaforensic pvt ltd contract", "Indiaforensic Services Pvt Ltd")
        self.assertEqual(score2, 1.0)
        
        score3 = _rule_score("limited private solutions finspire faturasi", "Finspire Solutions Private Limited")
        self.assertEqual(score3, 1.0)

    def test_token_fuzzy_score_is_order_invariant(self):
        """Token bazlı fuzzy benzerlik kelime sırasından etkilenmemeli."""
        sim1 = _compute_token_fuzzy_score("corporation microsoft", "Microsoft Corporation")
        self.assertEqual(sim1, 1.0)
        
        sim2 = _compute_token_fuzzy_score("services indiaforensic", "Indiaforensic Services")
        self.assertEqual(sim2, 1.0)

    def test_build_score_features_high_score_on_shuffled_words(self):
        """Kelime sırası değiştiğinde bile build_score_features yüksek kural ve fuzzy skoru üretmeli."""
        cand = {
            "variant_name": "Indiaforensic Services Pvt Ltd",
            "trgm_score": 0.30,
            "vector_score": 0.50,
            "full_text_score": 0.80
        }
        features = build_score_features(
            norm_exp="contract payment to services indiaforensic pvt ltd odeme",
            cand=cand,
            raw_explanation="contract payment to services indiaforensic pvt ltd odeme"
        )
        self.assertEqual(features["rule_score"], 1.0)
        self.assertEqual(features["fuzzy_score"], 1.0)

    def test_entity_extractor_extracts_shuffled_words_via_candidate(self):
        """EntityExtractor kelime sırası değiştiğinde de aday üzerinden varlığı çıkarabilmeli."""
        extractor = EntityExtractor()
        
        # Candidate listesinde doğru aday var ama EFT açıklamasında kelime sırası değişik
        cand_list = [
            {"company_id": 1, "variant_name": "Microsoft Corporation", "candidate_score": 0.8},
            {"company_id": 2, "variant_name": "Apple Inc", "candidate_score": 0.2}
        ]
        
        result = extractor.extract(
            text="transfer to corporation microsoft odeme",
            candidates=cand_list
        )
        
        self.assertEqual(result.extracted_entity, "Microsoft Corporation")
        self.assertEqual(result.extraction_method, "FUZZY_CANDIDATE_MATCH")


if __name__ == "__main__":
    unittest.main()
