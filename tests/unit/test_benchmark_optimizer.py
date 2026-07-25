import pytest
from src.evaluation.weight_optimizer import WeightOptimizer
from src.evaluation.benchmark import BenchmarkRunner, TestCase
from src.scoring.final_scorer import FinalScorer
from src.scoring.score_features import build_score_features

class TestBenchmarkAndOptimizer:
    def test_weight_optimizer_grid_search_selects_best_weights(self):
        """
        Parametrelerin (ağırlıkların) kıyaslama yapılarak en iyisinin seçilmesini test eder.
        """
        opt = WeightOptimizer(repo=None)
        
        # Simüle edilmiş raw_scores_list
        # case 1: should_match=True, fuzzy yüksek, vector düşük, reranker yüksek
        # case 2: should_match=True, fuzzy yüksek, vector düşük, reranker düşük
        # case 3: should_match=False, fuzzy düşük, vector düşük, reranker düşük
        raw_scores_list = [
            {
                "eft_id": 1,
                "should_match": True,
                "raw_scores": {
                    "fuzzy_score": 0.95, "vector_score": 0.20, "reranker_score": 0.90,
                    "acronym_score": 0.0, "rule_score": 0.0,
                    "query_token_count": 3, "exact_normalized_match": False, "exact_core_match": False,
                    "legal_suffix_only_difference": False, "consonant_match": False,
                    "candidate_company_id": 101, "alias_confidence": 1.0,
                    "_query_str": "microsoft corp", "_variant_str": "microsoft corporation"
                },
                "best_cand_name": "Microsoft Corporation"
            },
            {
                "eft_id": 2,
                "should_match": True,
                "raw_scores": {
                    "fuzzy_score": 0.88, "vector_score": 0.10, "reranker_score": 0.85,
                    "acronym_score": 0.0, "rule_score": 0.0,
                    "query_token_count": 3, "exact_normalized_match": False, "exact_core_match": False,
                    "legal_suffix_only_difference": False, "consonant_match": False,
                    "candidate_company_id": 102, "alias_confidence": 1.0,
                    "_query_str": "apple inc odeme", "_variant_str": "apple inc"
                },
                "best_cand_name": "Apple Inc"
            },
            {
                "eft_id": 3,
                "should_match": False,
                "raw_scores": {
                    "fuzzy_score": 0.30, "vector_score": 0.20, "reranker_score": 0.10,
                    "acronym_score": 0.0, "rule_score": 0.0,
                    "query_token_count": 3, "exact_normalized_match": False, "exact_core_match": False,
                    "legal_suffix_only_difference": False, "consonant_match": False,
                    "candidate_company_id": 103, "alias_confidence": 1.0,
                    "_query_str": "random text", "_variant_str": "unrelated company"
                },
                "best_cand_name": "Unrelated Company"
            }
        ]
        
        # Ağırlık kombinasyonlarını ara:
        # Combo A: fuzzy=0.50, vector=0.10, reranker=0.40 -> toplam 1.00
        # Combo B: fuzzy=0.20, vector=0.60, reranker=0.20 -> toplam 1.00
        res = opt.grid_search(
            raw_scores_list,
            fuzzy_range=[0.20, 0.50],
            vector_range=[0.10, 0.60],
            reranker_range=[0.20, 0.40]
        )
        
        assert res is not None
        assert "best_combination" in res
        assert "all_results" in res
        assert len(res["all_results"]) >= 2
        
        best = res["best_combination"]
        assert best is not None
        # En iyi kombinasyonun F1 skoru 0'dan büyük olmalı ve en üst sırada olmalı
        assert best["f1"] >= res["all_results"][-1]["f1"]
        assert round(best["fuzzy_weight"] + best["vector_weight"] + best["reranker_weight"], 2) == 1.00

    def test_benchmark_runner_compute_metrics_and_eval_single(self):
        """
        Proje genelindeki değişikliklerin doğruluğunun ve metriklerin benchmark ile test edilmesi.
        """
        runner = BenchmarkRunner(retriever=None, scorer=None, repo=None)
        
        tc1 = TestCase(
            test_case_id="tc_001",
            eft_explanation="M!cr0s0ft C0rp0r4t!0n odeme",
            expected_company_id=10,
            expected_label="HIGH_ALERT",
            case_type="leetspeak",
            difficulty_level="high"
        )
        
        tc2 = TestCase(
            test_case_id="tc_002",
            eft_explanation="Indaforensic Services Pvt Ltd", # typo
            expected_company_id=20,
            expected_label="HIGH_ALERT",
            case_type="typo",
            difficulty_level="medium"
        )
        
        tc3 = TestCase(
            test_case_id="tc_003",
            eft_explanation="random unrelated transfer",
            expected_company_id=None,
            expected_label="NO_MATCH",
            case_type="normal",
            difficulty_level="easy"
        )
        
        rec1 = runner.evaluate_single(
            test_case=tc1,
            predicted_company_id=10,
            predicted_label="HIGH_ALERT",
            predicted_score=0.95,
            retrieved_company_ids=[10, 15, 20],
            reason_codes=["LEETSPEAK_EVASION"]
        )
        assert rec1.is_correct is True
        assert rec1.recall_at_1 is True
        
        rec2 = runner.evaluate_single(
            test_case=tc2,
            predicted_company_id=20,
            predicted_label="HIGH_ALERT",
            predicted_score=0.90,
            retrieved_company_ids=[15, 20, 25],
            reason_codes=["HIGH_FUZZY_SIMILARITY"]
        )
        assert rec2.is_correct is True
        assert rec2.recall_at_5 is True
        assert rec2.reciprocal_rank == 0.5 # 2. sırada
        
        rec3 = runner.evaluate_single(
            test_case=tc3,
            predicted_company_id=None,
            predicted_label="NO_MATCH",
            predicted_score=0.10,
            retrieved_company_ids=[30, 40],
            reason_codes=[]
        )
        assert rec3.is_correct is True
        assert rec3.is_true_negative is True
        
        metrics = runner.compute_metrics([rec1, rec2, rec3])
        assert metrics["total_cases"] == 3
        assert metrics["accuracy"] == 1.0  # %100 doğruluk
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_benchmark_with_new_features_integration(self):
        """
        Yeni eklenen özelliklerin (leetspeak, typo, partial match review) build_score_features,
        FinalScorer ve BenchmarkRunner üzerinden uçtan uca doğrulanması.
        """
        scorer = FinalScorer(repo=None)
        runner = BenchmarkRunner(retriever=None, scorer=scorer, repo=None)
        
        # 1. Leetspeak Test Case
        cand_ms = {"company_id": 500, "variant_name": "Microsoft Corporation", "trgm_score": 0.40, "vector_score": 0.50}
        exp_ms = "M!cr0s0ft C0rp0r4t!0n lisans bedeli"
        scores_ms = build_score_features(exp_ms, cand_ms, raw_explanation=exp_ms)
        final_score_ms, reason_ms, codes_ms = scorer.calculate_final_score(scores_ms)
        risk_ms = scorer.assign_risk_level(final_score_ms)
        label_ms = scorer.assign_decision_status(risk_ms)
        
        assert "LEETSPEAK_EVASION" in codes_ms
        assert label_ms == "HIGH_ALERT"
        
        tc_leetspeak = TestCase(
            test_case_id="e2e_001",
            eft_explanation=exp_ms,
            expected_company_id=500,
            expected_label="HIGH_ALERT",
            case_type="evasion",
            difficulty_level="high"
        )
        rec_leetspeak = runner.evaluate_single(
            test_case=tc_leetspeak,
            predicted_company_id=500,
            predicted_label=label_ms,
            predicted_score=final_score_ms,
            retrieved_company_ids=[500],
            reason_codes=codes_ms
        )
        assert rec_leetspeak.is_correct is True
        
        # 2. Substantial Missing Info (Partial Match Review) Test Case
        cand_partial = {"company_id": 501, "variant_name": "International Business Machines Corporation", "trgm_score": 0.65}
        exp_partial = "International transfer" # Büyük eksiklik var
        scores_partial = build_score_features(exp_partial, cand_partial, raw_explanation=exp_partial)
        final_score_p, reason_p, codes_p = scorer.calculate_final_score(scores_partial)
        risk_p = scorer.assign_risk_level(final_score_p)
        label_p = scorer.assign_decision_status(risk_p)
        
        assert "PARTIAL_MATCH_REQUIRES_REVIEW" in codes_p
        assert risk_p == "Medium"
        assert label_p in ("MEDIUM_ALERT", "REVIEW")
