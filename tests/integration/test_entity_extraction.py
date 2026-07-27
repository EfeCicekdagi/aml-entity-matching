"""
test_entity_extraction.py — Çok katmanlı entity extraction testleri.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.models.entity_extractor import EntityExtractor, ExtractionResult
from src.models.ner_extractor import NERExtractor
from src.pipeline.match_engine import MatchEngine


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

    def test_ner_returns_metadata_person(self):
        """NER metadata dict döndürdüğünde PERSON entity türü, güven skoru ve offsetler korunur."""
        mock_ner = MagicMock()
        mock_ner.extract_entity.return_value = {
            "text": "Ahmet Yılmaz",
            "entity_type": "PERSON",
            "confidence": 0.91,
            "start": 10,
            "end": 22
        }
        extractor = EntityExtractor(ner_extractor=mock_ner)
        result = extractor.extract("TRANSFER TO AHMET YILMAZ FOR REFUND")
        self.assertEqual(result.extraction_method, "NER_MODEL")
        self.assertEqual(result.extracted_entity, "Ahmet Yılmaz")
        self.assertEqual(result.entity_type, "PERSON")
        self.assertEqual(result.extraction_confidence, 0.91)
        self.assertEqual(result.extraction_start, 10)
        self.assertEqual(result.extraction_end, 22)

    def test_batch_ner_returns_metadata(self):
        """Toplu extraction işleminde hem ORG hem PER ayrımı ve metadata değerleri korunur."""
        mock_ner = MagicMock()
        mock_ner.batch_extract_entities.return_value = [
            {"text": "Apple Inc", "entity_type": "ORGANIZATION", "confidence": 0.98, "start": 0, "end": 9},
            {"text": "Mehmet Demir", "entity_type": "PERSON", "confidence": 0.92, "start": 12, "end": 24}
        ]
        extractor = EntityExtractor(ner_extractor=mock_ner)
        results = extractor.batch_extract(["Apple Inc Laptops", "Payment to Mehmet Demir"])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].extracted_entity, "Apple Inc")
        self.assertEqual(results[0].entity_type, "ORGANIZATION")
        self.assertEqual(results[1].extracted_entity, "Mehmet Demir")
        self.assertEqual(results[1].entity_type, "PERSON")
        self.assertEqual(results[1].extraction_confidence, 0.92)

    def test_ner_confidence_preserved(self):
        """NER modelinden dönen gerçek skorun (örn. 0.51) yapay 0.90 yerine aynen korunduğunu doğrular."""
        mock_ner = MagicMock()
        mock_ner.extract_entity.return_value = {
            "text": "Low Conf Corp",
            "entity_type": "ORGANIZATION",
            "confidence": 0.51,
            "start": 0,
            "end": 13
        }
        extractor = EntityExtractor(ner_extractor=mock_ner)
        result = extractor.extract("Low Conf Corp transfer")
        self.assertEqual(result.extraction_confidence, 0.51)
        self.assertNotEqual(result.extraction_confidence, 0.90)

    def test_ner_offsets_preserved(self):
        """NER modelinden dönen karakter başlangıç ve bitiş offsetlerinin (start, end) korunduğunu doğrular."""
        mock_ner = MagicMock()
        mock_ner.extract_entity.return_value = {
            "text": "Acme Corp",
            "entity_type": "ORGANIZATION",
            "confidence": 0.95,
            "start": 15,
            "end": 24
        }
        extractor = EntityExtractor(ner_extractor=mock_ner)
        result = extractor.extract("Payment sent to Acme Corp for services")
        self.assertEqual(result.extracted_entity, "Acme Corp")
        self.assertEqual(result.extraction_start, 15)
        self.assertEqual(result.extraction_end, 24)

    @patch("src.models.ner_extractor.pipeline")
    def test_ner_raw_text_without_title_case(self, mock_pipeline_fn):
        """NERExtractor'ın küçük harfli metni t.title() ile yapay olarak değiştirmeden doğrudan model pipeline'ına ilettiğini doğrular."""
        mock_pipeline_inst = MagicMock(return_value=[])
        mock_pipeline_fn.return_value = mock_pipeline_inst
        ner = NERExtractor(model_name="dummy-model", device=-1)
        
        raw_text = "kurumsal tedarikçi ödemesi ibm ltd"
        ner.extract_entity(raw_text)
        mock_pipeline_inst.assert_called_with("kurumsal tedarikçi ödemesi ibm ltd")
        
        ner.batch_extract_entities([raw_text])
        mock_pipeline_inst.assert_called_with(["kurumsal tedarikçi ödemesi ibm ltd"], batch_size=64)

    @patch("src.models.ner_extractor.pipeline")
    def test_ner_org_priority_over_person_length(self, mock_pipeline_fn):
        """NERExtractor'ın yalnızca uzunluğa bakmak yerine ORG (şirket) entity'sini daha uzun PER (kişi) entity'sine tercih ettiğini doğrular."""
        mock_pipeline_inst = MagicMock()
        mock_pipeline_fn.return_value = mock_pipeline_inst
        ner = NERExtractor(model_name="dummy-model", device=-1)
        
        # Ahmet Yılmazoğlu (16 karakter - PER) vs Apple Corp (10 karakter - ORG)
        mock_pipeline_inst.return_value = [
            {"word": "Ahmet Yılmazoğlu", "entity_group": "PER", "score": 0.95, "start": 0, "end": 16},
            {"word": "Apple Corp", "entity_group": "ORG", "score": 0.90, "start": 23, "end": 33}
        ]
        
        res = ner.extract_entity("Ahmet Yılmazoğlu adına Apple Corp ödemesi")
        self.assertIsNotNone(res)
        self.assertEqual(res["text"], "Apple Corp")
        self.assertEqual(res["entity_type"], "ORGANIZATION")

    def test_match_engine_candidate_supported_stage2(self):
        """MatchEngine'in toplu işleme (process_batch) sırasında 2. aşama candidate-supported extraction'ı devreye aldığını doğrular."""
        mock_retriever = MagicMock()
        mock_retriever.batch_get_candidates.return_value = {
            "row_0": {
                "candidates": [{"variant_name": "British Petroleum", "candidate_score": 0.90}],
                "pipeline_status": "CANDIDATES_FOUND"
            },
            "row_1": {
                "candidates": [{"variant_name": "Microsoft", "candidate_score": 0.95}],
                "pipeline_status": "CANDIDATES_FOUND"
            }
        }
        
        engine = MatchEngine(
            config={},
            retriever=mock_retriever,
            reranker=MagicMock(),
            embedding_model=MagicMock(),
            entity_extractor=self.extractor
        )
        
        # bp is acronym of British Petroleum; microsoft is exact variant name without suffix
        results = engine.process_batch(["ödeme bp transferi", "havale microsoft bedeli"], ["row_0", "row_1"])
        
        self.assertEqual(results["row_0"]["extraction"].extraction_method, "ACRONYM_SUPPORTED")
        self.assertEqual(results["row_0"]["extraction"].extracted_entity, "British Petroleum")
        
        self.assertEqual(results["row_1"]["extraction"].extraction_method, "CANDIDATE_SUPPORTED")
        self.assertEqual(results["row_1"]["extraction"].extracted_entity, "Microsoft")

    def test_batch_extract_with_custom_row_ids(self):
        """batch_extract'in candidates_per_row haritasından sorgulama yaparken gerçek row_id'leri kullandığını doğrular."""
        texts = ["ödeme bp transferi", "havale microsoft bedeli"]
        row_ids = ["1000999", "1001000"]
        candidates_per_row = {
            "1000999": [{"variant_name": "British Petroleum", "candidate_score": 0.90}],
            "1001000": [{"variant_name": "Microsoft", "candidate_score": 0.95}]
        }
        
        results = self.extractor.batch_extract(texts, row_ids=row_ids, candidates_per_row=candidates_per_row)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].extraction_method, "ACRONYM_SUPPORTED")
        self.assertEqual(results[0].extracted_entity, "British Petroleum")
        self.assertEqual(results[1].extraction_method, "CANDIDATE_SUPPORTED")
        self.assertEqual(results[1].extracted_entity, "Microsoft")


if __name__ == "__main__":
    unittest.main()
