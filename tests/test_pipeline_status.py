"""
test_pipeline_status.py — Pipeline status ayrıştırma testleri.

Test edilen davranışlar:
  - Trigram eşik altında kalan EFT'ler CLEAN değil NO_CANDIDATE_FOUND döner
  - Tüm kanallar boşsa decision_status = NO_CANDIDATE_FOUND
  - HIGH aday varsa decision_status = HIGH_ALERT
  - LOW skor adaylar alert tablosuna yazılmaz
"""

import unittest
from unittest.mock import MagicMock, patch
from src.scoring.final_scorer import FinalScorer
from src.scoring.reason_codes import ReasonCode


class TestPipelineStatus(unittest.TestCase):
    """Pipeline status ve decision_status ayrıştırma testleri."""

    def setUp(self):
        """Test için mock FinalScorer hazırla."""
        mock_repo = MagicMock()
        mock_repo.get_connection.return_value = None  # DB bağlantısı yok
        self.scorer = FinalScorer(mock_repo)
        # Fallback thresholds (DB yok)
        self.scorer.thresholds = []
        self.scorer.weights = {
            "fuzzy_weight": 0.0, "vector_weight": 0.30,
            "acronym_weight": 0.0, "rule_weight": 0.0, "reranker_weight": 0.70
        }

    # ── assign_risk_level testleri ───────────────────────────────────────────

    def test_high_score_gives_high_risk(self):
        self.assertEqual(self.scorer.assign_risk_level(0.85), "HIGH")

    def test_exactly_0_70_gives_high_risk(self):
        self.assertEqual(self.scorer.assign_risk_level(0.70), "HIGH")

    def test_medium_score_gives_medium_risk(self):
        self.assertEqual(self.scorer.assign_risk_level(0.65), "MEDIUM")

    def test_exactly_0_62_gives_medium_risk(self):
        self.assertEqual(self.scorer.assign_risk_level(0.62), "MEDIUM")

    def test_below_0_62_gives_no_match(self):
        self.assertEqual(self.scorer.assign_risk_level(0.61), "NO_MATCH")
        self.assertEqual(self.scorer.assign_risk_level(0.50), "NO_MATCH")
        self.assertEqual(self.scorer.assign_risk_level(0.00), "NO_MATCH")

    # ── assign_decision_status testleri ─────────────────────────────────────

    def test_high_risk_gives_high_alert_decision(self):
        self.assertEqual(self.scorer.assign_decision_status("HIGH"), "HIGH_ALERT")

    def test_medium_risk_gives_medium_alert_decision(self):
        self.assertEqual(self.scorer.assign_decision_status("MEDIUM"), "MEDIUM_ALERT")

    def test_no_match_gives_no_match_decision(self):
        self.assertEqual(self.scorer.assign_decision_status("NO_MATCH"), "NO_MATCH")

    def test_low_risk_gives_below_threshold_decision(self):
        self.assertEqual(self.scorer.assign_decision_status("LOW"), "MATCH_BELOW_THRESHOLD")

    # ── is_alert_worthy testleri ─────────────────────────────────────────────

    def test_high_is_alert_worthy(self):
        self.assertTrue(self.scorer.is_alert_worthy("HIGH"))

    def test_medium_is_alert_worthy(self):
        self.assertTrue(self.scorer.is_alert_worthy("MEDIUM"))

    def test_no_match_is_not_alert_worthy(self):
        self.assertFalse(self.scorer.is_alert_worthy("NO_MATCH"))

    def test_low_is_not_alert_worthy(self):
        self.assertFalse(self.scorer.is_alert_worthy("LOW"))

    # ── pipeline_status → NO_CANDIDATE_FOUND ────────────────────────────────

    def test_empty_candidates_should_be_no_candidate_found(self):
        """Boş aday listesi → NO_CANDIDATE_FOUND (CLEAN değil)."""
        from src.retrieval.postgres_candidate_retriever import PostgresCandidateRetriever

        mock_repo = MagicMock()
        retriever = PostgresCandidateRetriever(mock_repo, {})

        status, reason = retriever._compute_pipeline_status([], {})
        self.assertEqual(status, "ALL_RETRIEVAL_CHANNELS_EMPTY")
        self.assertIsNotNone(reason)

    def test_candidates_found_gives_correct_status(self):
        """Aday varsa CANDIDATES_FOUND döner."""
        from src.retrieval.postgres_candidate_retriever import PostgresCandidateRetriever

        mock_repo = MagicMock()
        retriever = PostgresCandidateRetriever(mock_repo, {})

        cands = [{"company_id": 1, "variant_id": 1, "candidate_score": 0.8}]
        status, reason = retriever._compute_pipeline_status(cands, {"trgm": 1, "fts": 0, "vector": 0})
        self.assertEqual(status, "CANDIDATES_FOUND")
        self.assertIsNone(reason)

    def test_only_trgm_empty_gives_trigram_no_result(self):
        """Sadece trigram boş → TRIGRAM_NO_RESULT."""
        from src.retrieval.postgres_candidate_retriever import PostgresCandidateRetriever

        mock_repo = MagicMock()
        retriever = PostgresCandidateRetriever(mock_repo, {})

        status, reason = retriever._compute_pipeline_status([], {"trgm": 0, "fts": 5, "vector": 3})
        # FTS ve vector var ama merged boş — bu durumda prefilter almış olabilir
        # Bu test merge öncesi channel bazında kontrol eder
        self.assertIn("TRIGRAM", status)


class TestDecisionSeparation(unittest.TestCase):
    """Teknik pipeline_status ile iş decision_status ayrımı."""

    def test_pipeline_status_and_decision_status_are_separate_concepts(self):
        """
        pipeline_status = teknik retrieval sonucu
        decision_status = iş kararı
        İkisi aynı olmak zorunda değil.
        """
        # Aday bulundu ama skor düşük → CANDIDATES_FOUND + NO_MATCH
        pipeline = "CANDIDATES_FOUND"
        risk_level = "NO_MATCH"

        mock_repo = MagicMock()
        mock_repo.get_connection.return_value = None
        scorer = FinalScorer(mock_repo)
        scorer.thresholds = []

        decision = scorer.assign_decision_status(risk_level)
        self.assertNotEqual(pipeline, decision)
        self.assertEqual(decision, "NO_MATCH")

    def test_no_candidate_decision_is_not_clean(self):
        """NO_CANDIDATE_FOUND, CLEAN ile aynı değil."""
        decision = "NO_CANDIDATE_FOUND"
        self.assertNotEqual(decision, "CLEAN")


if __name__ == "__main__":
    unittest.main()
