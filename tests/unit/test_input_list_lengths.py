import unittest
from unittest.mock import MagicMock
from src.pipeline.match_engine import MatchEngine
from src.pipeline.inference_service import AMLInferenceService


class TestInputListLengthsValidation(unittest.TestCase):
    """Girdi listesi uzunluklarının tutarlılık kontrolleri testleri."""

    def setUp(self):
        self.mock_retriever = MagicMock()
        self.mock_reranker = MagicMock()
        self.mock_extractor = MagicMock()
        self.mock_embedding = MagicMock()

        self.engine = MatchEngine(
            config={},
            retriever=self.mock_retriever,
            reranker=self.mock_reranker,
            entity_extractor=self.mock_extractor,
            embedding_model=self.mock_embedding,
        )

        mock_repo = MagicMock()
        mock_repo.get_connection.return_value = None
        self.service = AMLInferenceService(
            config={"retrieval": {"reranker_top_k": 5}},
            retriever=self.mock_retriever,
            reranker=self.mock_reranker,
            entity_extractor=self.mock_extractor,
            embedding_model=self.mock_embedding,
            match_engine=self.engine
        )

    def test_match_engine_process_batch_length_mismatch_raises_value_error(self):
        """MatchEngine.process_batch raw_explanations ve row_ids farklı boyutta ise ValueError fırlatmalı."""
        with self.assertRaisesRegex(ValueError, "raw_explanations and row_ids must have equal lengths"):
            self.engine.process_batch(["exp1", "exp2"], ["row_1"])

    def test_inference_service_analyze_batch_length_mismatch_raises_value_error(self):
        """AMLInferenceService.analyze_batch listelerden herhangi biri farklı boyutta ise ValueError fırlatmalı."""
        # eft_ids eksik
        with self.assertRaisesRegex(ValueError, "raw_explanations, row_ids and eft_ids must have equal lengths"):
            self.service.analyze_batch(
                raw_explanations=["exp1", "exp2"],
                row_ids=["row_1", "row_2"],
                run_id="test_run",
                eft_ids=[101]
            )

        # row_ids eksik
        with self.assertRaisesRegex(ValueError, "raw_explanations, row_ids and eft_ids must have equal lengths"):
            self.service.analyze_batch(
                raw_explanations=["exp1", "exp2"],
                row_ids=["row_1"],
                run_id="test_run",
                eft_ids=[101, 102]
            )


if __name__ == "__main__":
    unittest.main()
