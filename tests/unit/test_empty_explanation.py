import unittest
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from src.pipeline.match_engine import MatchEngine
from src.pipeline.inference_service import AMLInferenceService
from src.scoring.final_scorer import FinalScorer
from src.scoring.reason_codes import ReasonCode


class TestEmptyExplanationGuard(unittest.TestCase):
    """Boş veya NaN açıklamaların arama engellemesi ve doğru statü vermesi testleri."""

    def setUp(self):
        self.mock_retriever = MagicMock()
        self.mock_reranker = MagicMock()
        self.mock_extractor = MagicMock()
        self.mock_embedding = MagicMock()

        # Engine kur
        self.engine = MatchEngine(
            config={},
            retriever=self.mock_retriever,
            reranker=self.mock_reranker,
            entity_extractor=self.mock_extractor,
            embedding_model=self.mock_embedding,
        )

        # Scorer kur
        mock_repo = MagicMock()
        mock_repo.get_connection.return_value = None
        self.scorer = FinalScorer(mock_repo)
        self.scorer.thresholds = []
        self.scorer.weights = {
            "fuzzy_weight": 0.0, "vector_weight": 0.30,
            "acronym_weight": 0.0, "rule_weight": 0.0, "reranker_weight": 0.70
        }

        # Inference service kur
        self.service = AMLInferenceService(
            config={"retrieval": {"reranker_top_k": 5}},
            retriever=self.mock_retriever,
            reranker=self.mock_reranker,
            entity_extractor=self.mock_extractor,
            embedding_model=self.mock_embedding,
            scorer=self.scorer,
            match_engine=self.engine
        )

    def test_match_engine_empty_or_nan_explanations(self):
        """None, NaN, boşluk ve 'nan' ifadlerinde veritabanı araması yapılmamalı ve EMPTY_EXPLANATION statüsü dönmeli."""
        raw_explanations = [None, np.nan, float("nan"), "", "   ", "nan", "NAN"]
        row_ids = [f"row_{i}" for i in range(len(raw_explanations))]

        # Mock retriever empty dönmeli
        self.mock_retriever.batch_get_candidates.return_value = {}

        results = self.engine.process_batch(raw_explanations, row_ids)

        # Arama (batch_get_candidates) çağrıldığında rows_for_batch boş olmalı
        self.mock_retriever.batch_get_candidates.assert_called_once_with([])

        for rid in row_ids:
            res = results[rid]
            self.assertEqual(res["pipeline_status"], "EMPTY_EXPLANATION")
            self.assertEqual(res["no_candidate_reason"], "INVALID_INPUT")
            self.assertEqual(res["candidates"], [])

    def test_inference_service_empty_explanation_decision_status(self):
        """AMLInferenceService boş girdi için decision_status='INVALID_INPUT' dönmeli."""
        raw_explanations = [None, "nan"]
        row_ids = ["row_1", "row_2"]
        eft_ids = [101, 102]

        self.mock_retriever.batch_get_candidates.return_value = {}

        batch_res = self.service.analyze_batch(
            raw_explanations=raw_explanations,
            row_ids=row_ids,
            run_id="test_run",
            eft_ids=eft_ids
        )

        self.assertEqual(len(batch_res), 2)
        for res in batch_res:
            self.assertTrue(res["no_candidate"])
            mr = res["match_results"][0]
            self.assertEqual(mr["pipeline_status"], "EMPTY_EXPLANATION")
            self.assertEqual(mr["decision_status"], "INVALID_INPUT")
            self.assertIn(ReasonCode.INVALID_INPUT.value, mr["reason_codes"])
            self.assertIn("Boş veya geçersiz EFT açıklaması", mr["human_explanation"])


if __name__ == "__main__":
    unittest.main()
