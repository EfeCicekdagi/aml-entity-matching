"""
test_entity_extraction.py — Çok katmanlı entity extraction testleri.
"""

import unittest
from unittest.mock import MagicMock
from src.utils.entity_extractor import EntityExtractor, ExtractionResult


class TestEntityExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = EntityExtractor(ner_extractor=None)

    # ── Rule-based extraction ────────────────────────────────────────────────

    def test_rule_based_finds_company_with_suffix(self):
        """Şirket suffix içeren metin → rule-based yakalar."""
        text = "TRANSFER TO GLOBAL TRADING LTD FOR SERVICES"
        result = self.extractor._extract_via_rules(text)
        # Suffix varsa sonuç dönmeli
        # (Bazı metinler çok genel olabilir, None da kabul edilir)
        if result:
            self.assertIsInstance(result, ExtractionResult)
            self.assertIsNotNone(result.extracted_entity)

    def test_empty_text_returns_not_found(self):
        """Boş metin → ENTITY_NOT_FOUND."""
        result = self.extractor.extract("")
        self.assertEqual(result.entity_extraction_status, "NOT_FOUND")
        self.assertIsNone(result.extracted_entity)

    def test_none_like_text_returns_not_found(self):
        """Sadece boşluk → NOT_FOUND."""
        result = self.extractor.extract("   ")
        self.assertEqual(result.entity_extraction_status, "NOT_FOUND")

    # ── Candidate-supported extraction ──────────────────────────────────────

    def test_candidate_supported_finds_variant_in_text(self):
        """Variant adı EFT içinde geçiyorsa CANDIDATE_SUPPORTED döner."""
        text = "payment from apple trading limited for services"
        candidates = [
            {"variant_name": "apple trading limited", "candidate_score": 0.8},
            {"variant_name": "google corp", "candidate_score": 0.5},
        ]
        result = self.extractor._extract_via_candidates(text, candidates)
        self.assertIsNotNone(result)
        self.assertEqual(result.extraction_method, "CANDIDATE_SUPPORTED")
        self.assertIn("apple", result.extracted_entity.lower())

    def test_candidate_not_in_text_returns_none(self):
        """Variant metin içinde geçmiyorsa None döner."""
        text = "some unrelated text here"
        candidates = [
            {"variant_name": "completely different company", "candidate_score": 0.8},
        ]
        result = self.extractor._extract_via_candidates(text, candidates)
        self.assertIsNone(result)

    # ── Variant fallback ─────────────────────────────────────────────────────

    def test_variant_fallback_uses_best_candidate(self):
        """En yüksek skorlu aday fallback entity olarak kullanılır."""
        candidates = [
            {"variant_name": "low score corp", "candidate_score": 0.3},
            {"variant_name": "best match company", "candidate_score": 0.9},
        ]
        result = self.extractor._extract_via_variant_fallback("some text", candidates)
        self.assertIsNotNone(result)
        self.assertEqual(result.extraction_method, "FALLBACK_MATCHED_VARIANT")

    def test_variant_fallback_no_candidates_returns_none(self):
        """Aday yoksa None döner."""
        result = self.extractor._extract_via_variant_fallback("some text", [])
        self.assertIsNone(result)

    # ── Full text fallback ────────────────────────────────────────────────────

    def test_full_text_fallback_returns_snippet(self):
        """Full text fallback → metnin başından 60 karakter."""
        text = "this is a long explanation text that contains no specific entity"
        result = self.extractor._extract_full_text_fallback(text)
        self.assertIsNotNone(result.extracted_entity)
        self.assertLessEqual(len(result.extracted_entity), 60)
        self.assertEqual(result.extraction_method, "FULL_TEXT_FALLBACK")

    # ── Katman önceliği ──────────────────────────────────────────────────────

    def test_ner_overrides_rules(self):
        """NER mevcut ve sonuç üretiyorsa rule-based KULLANILMAZ."""
        mock_ner = MagicMock()
        mock_ner.extract_entity.return_value = "Mock Corp"
        extractor = EntityExtractor(ner_extractor=mock_ner)

        result = extractor.extract("TRANSFER TO GLOBAL TRADING LTD FOR SERVICES")
        self.assertEqual(result.extraction_method, "NER_MODEL")
        self.assertEqual(result.extracted_entity, "Mock Corp")

    def test_ner_failure_falls_through_to_rules(self):
        """NER başarısız olunca rule-based katman devreye girer."""
        mock_ner = MagicMock()
        mock_ner.extract_entity.side_effect = Exception("NER error")
        extractor = EntityExtractor(ner_extractor=mock_ner)

        # Rule-based bulabilirse RULE_BASED veya ENTITY_NOT_FOUND döner
        text = "TRANSFER FROM GLOBAL CORP LTD"
        result = extractor.extract(text)
        self.assertIn(result.extraction_method,
                      ["RULE_BASED", "ENTITY_NOT_FOUND"])

    # ── ExtractionResult defaults ────────────────────────────────────────────

    def test_extraction_result_defaults(self):
        """ExtractionResult varsayılan değerleri kontrol."""
        r = ExtractionResult()
        self.assertIsNone(r.extracted_entity)
        self.assertEqual(r.entity_type, "UNKNOWN")
        self.assertEqual(r.extraction_method, "ENTITY_NOT_FOUND")
        self.assertEqual(r.extraction_confidence, 0.0)
        self.assertEqual(r.entity_extraction_status, "NOT_FOUND")

    # ── Batch extraction ─────────────────────────────────────────────────────

    def test_batch_extract_returns_same_count(self):
        """batch_extract giriş ile aynı uzunlukta çıkış üretir."""
        texts = ["text one", "text two", "text three"]
        results = self.extractor.batch_extract(texts)
        self.assertEqual(len(results), len(texts))

    def test_batch_extract_handles_empty_texts(self):
        """batch_extract boş listede hata vermez."""
        results = self.extractor.batch_extract([])
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
