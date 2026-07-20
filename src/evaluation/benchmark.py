"""
benchmark.py — AML pipeline sistematik benchmark modülü.

Retrieval metrikleri:
  - Recall@K (1,5,10,20,30)
  - Mean Reciprocal Rank (MRR)
  - Candidate Reduction Ratio

Nihai karar metrikleri:
  - Precision, Recall, F1
  - False Positive Rate, False Negative Rate
  - PR-AUC, ROC-AUC
  - Confusion Matrix

Kırılım boyutları:
  - Entity type, Dil, Alfabe, Case type, Zorluk seviyesi, NER başarısı

Çıktı formatları: JSON, CSV, Streamlit dashboard

Kullanım:
    bm = BenchmarkRunner(repo=repo, retriever=retriever, scorer=scorer)
    results = bm.run(test_cases)
    report = bm.compute_metrics(results)
    bm.save_report(report, "benchmark_v1")
"""

import json
import csv
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Tek bir benchmark test vakası."""
    test_case_id:       str
    eft_explanation:    str
    expected_entity:    Optional[str]
    expected_company_id: Optional[int]
    expected_variant_id: Optional[int]
    expected_label:     str              # MATCH, NO_MATCH, HIGH_ALERT, MEDIUM_ALERT
    difficulty_level:   str = "MEDIUM"  # EASY, MEDIUM, HARD, EXPERT
    case_type:          str = "EXACT_MATCH"
    language:           str = "TR"
    script:             str = "LATIN"
    entity_type:        str = "ORGANIZATION"
    notes:              str = ""


@dataclass
class BenchmarkRecord:
    """Tek test vakasının benchmark sonucu."""
    test_case_id:       str
    eft_explanation:    str
    expected_label:     str
    predicted_label:    str
    predicted_company_id: Optional[int]
    predicted_score:    float
    retrieval_rank:     Optional[int]   # Doğru sonucun kaçıncı sırada geldiği
    recall_at_1:        bool
    recall_at_5:        bool
    recall_at_10:       bool
    recall_at_20:       bool
    recall_at_30:       bool
    reciprocal_rank:    float           # 1/rank (MRR için)
    is_correct:         bool
    is_true_positive:   bool
    is_false_positive:  bool
    is_true_negative:   bool
    is_false_negative:  bool
    processing_time_ms: float
    reason_codes:       list = field(default_factory=list)
    # Kırılım boyutları
    difficulty_level:   str = "MEDIUM"
    case_type:          str = "EXACT_MATCH"
    language:           str = "TR"
    script:             str = "LATIN"
    entity_type:        str = "ORGANIZATION"


class BenchmarkRunner:
    """
    AML pipeline benchmark çalıştırıcı.
    DB bağlantısı olmadan da mock verilerle çalışabilir.
    """

    def __init__(self, retriever=None, scorer=None, repo=None, entity_extractor=None):
        """
        Args:
            retriever: PostgresCandidateRetriever (opsiyonel)
            scorer: FinalScorer (opsiyonel)
            repo: AMLRepository (opsiyonel)
            entity_extractor: EntityExtractor (opsiyonel)
        """
        self.retriever       = retriever
        self.scorer          = scorer
        self.repo            = repo
        self.entity_extractor = entity_extractor

    def evaluate_single(
        self,
        test_case: TestCase,
        predicted_company_id: Optional[int],
        predicted_label: str,
        predicted_score: float,
        retrieved_company_ids: list[int],
        reason_codes: list = None,
        processing_time_ms: float = 0.0,
    ) -> BenchmarkRecord:
        """
        Tek test vakasını değerlendirir.

        Args:
            test_case: Test vakası
            predicted_company_id: Sistem tarafından tahmin edilen şirket ID
            predicted_label: Sistemin kararı (HIGH_ALERT, MEDIUM_ALERT, NO_MATCH, vb.)
            predicted_score: Final skor
            retrieved_company_ids: Retrieval'dan gelen şirket ID listesi (sıralı)
            reason_codes: Üretilen reason code listesi
            processing_time_ms: İşlem süresi

        Returns:
            BenchmarkRecord
        """
        expected_id = test_case.expected_company_id

        # Retrieval rank: doğru company_id kaçıncı sırada?
        retrieval_rank = None
        if expected_id and expected_id in retrieved_company_ids:
            retrieval_rank = retrieved_company_ids.index(expected_id) + 1

        # Recall@K hesapla
        recall_at_k = {
            1: False, 5: False, 10: False, 20: False, 30: False
        }
        if retrieval_rank is not None:
            for k in recall_at_k:
                recall_at_k[k] = retrieval_rank <= k

        reciprocal_rank = (1.0 / retrieval_rank) if retrieval_rank else 0.0

        # Doğruluk: expected_label ile predicted_label karşılaştır
        is_alert_predicted = predicted_label in ("HIGH_ALERT", "MEDIUM_ALERT", "MATCH")
        is_alert_expected  = test_case.expected_label in ("MATCH", "HIGH_ALERT", "MEDIUM_ALERT", "ALERT_CREATED")

        is_tp = is_alert_expected and is_alert_predicted
        is_fp = not is_alert_expected and is_alert_predicted
        is_tn = not is_alert_expected and not is_alert_predicted
        is_fn = is_alert_expected and not is_alert_predicted

        is_correct = (predicted_label == test_case.expected_label) or (is_tp or is_tn)

        return BenchmarkRecord(
            test_case_id        = test_case.test_case_id,
            eft_explanation     = test_case.eft_explanation,
            expected_label      = test_case.expected_label,
            predicted_label     = predicted_label,
            predicted_company_id = predicted_company_id,
            predicted_score     = predicted_score,
            retrieval_rank      = retrieval_rank,
            recall_at_1         = recall_at_k[1],
            recall_at_5         = recall_at_k[5],
            recall_at_10        = recall_at_k[10],
            recall_at_20        = recall_at_k[20],
            recall_at_30        = recall_at_k[30],
            reciprocal_rank     = reciprocal_rank,
            is_correct          = is_correct,
            is_true_positive    = is_tp,
            is_false_positive   = is_fp,
            is_true_negative    = is_tn,
            is_false_negative   = is_fn,
            processing_time_ms  = processing_time_ms,
            reason_codes        = reason_codes or [],
            difficulty_level    = test_case.difficulty_level,
            case_type           = test_case.case_type,
            language            = test_case.language,
            script              = test_case.script,
            entity_type         = test_case.entity_type,
        )

    def compute_metrics(
        self,
        records: list[BenchmarkRecord],
        breakdown_dims: list[str] = None,
    ) -> dict:
        """
        Tüm benchmark kayıtlarından metrik hesaplar.

        Args:
            records: Değerlendirme kayıtları
            breakdown_dims: Kırılım boyutları
                            ["case_type", "difficulty_level", "language", "script", "entity_type"]

        Returns:
            Metrik dict (genel + kırılım)
        """
        if not records:
            return {"error": "No records to evaluate"}

        def _compute_basic(recs: list[BenchmarkRecord]) -> dict:
            n = len(recs)
            tp = sum(1 for r in recs if r.is_true_positive)
            fp = sum(1 for r in recs if r.is_false_positive)
            tn = sum(1 for r in recs if r.is_true_negative)
            fn = sum(1 for r in recs if r.is_false_negative)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            recall_at = {
                "recall_at_1":  sum(1 for r in recs if r.recall_at_1) / n,
                "recall_at_5":  sum(1 for r in recs if r.recall_at_5) / n,
                "recall_at_10": sum(1 for r in recs if r.recall_at_10) / n,
                "recall_at_20": sum(1 for r in recs if r.recall_at_20) / n,
                "recall_at_30": sum(1 for r in recs if r.recall_at_30) / n,
            }

            mrr_values = [r.reciprocal_rank for r in recs if r.retrieval_rank]
            mrr = sum(mrr_values) / len(mrr_values) if mrr_values else 0.0

            return {
                "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision": round(precision, 4),
                "recall":    round(recall, 4),
                "f1":        round(f1, 4),
                "fpr":       round(fpr, 4),
                "fnr":       round(fnr, 4),
                "mrr":       round(mrr, 4),
                **{k: round(v, 4) for k, v in recall_at.items()},
                "avg_score": round(
                    sum(r.predicted_score for r in recs) / n, 4
                ),
                "avg_latency_ms": round(
                    sum(r.processing_time_ms for r in recs) / n, 2
                ),
            }

        report = {"overall": _compute_basic(records)}

        # Kırılım
        for dim in (breakdown_dims or ["case_type", "difficulty_level", "language"]):
            groups = defaultdict(list)
            for r in records:
                groups[getattr(r, dim, "UNKNOWN")].append(r)
            report[f"by_{dim}"] = {
                k: _compute_basic(v) for k, v in groups.items()
            }

        return report

    def save_report_json(self, report: dict, output_path: str) -> None:
        """Raporu JSON olarak kaydeder."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"Benchmark report saved to {output_path}")

    def save_records_csv(self, records: list[BenchmarkRecord], output_path: str) -> None:
        """Kayıtları CSV olarak kaydeder."""
        if not records:
            return
        rows = [asdict(r) for r in records]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Benchmark records saved to {output_path}")

    def load_test_cases_from_db(self) -> list[TestCase]:
        """
        Veritabanındaki test vakalarını yükler.

        Returns:
            TestCase listesi
        """
        if not self.repo:
            return []

        from src.config.db_tables import TABLES
        conn = self.repo.get_connection()
        if not conn:
            return []

        cases = []
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT test_case_id, eft_explanation, expected_entity,
                           expected_company_id, expected_variant_id, expected_label,
                           difficulty_level, case_type, language, script, entity_type, notes
                    FROM {TABLES['test_case']}
                    ORDER BY test_case_id
                """)
                for row in cur.fetchall():
                    cases.append(TestCase(
                        test_case_id       = row[0],
                        eft_explanation    = row[1],
                        expected_entity    = row[2],
                        expected_company_id = row[3],
                        expected_variant_id = row[4],
                        expected_label     = row[5],
                        difficulty_level   = row[6] or "MEDIUM",
                        case_type          = row[7] or "EXACT_MATCH",
                        language           = row[8] or "TR",
                        script             = row[9] or "LATIN",
                        entity_type        = row[10] or "ORGANIZATION",
                        notes              = row[11] or "",
                    ))
        except Exception as e:
            logger.error(f"Error loading test cases from DB: {e}")
        finally:
            self.repo.release_connection(conn)

        logger.info(f"Loaded {len(cases)} test cases from DB.")
        return cases

    def save_results_to_db(
        self,
        records: list[BenchmarkRecord],
        benchmark_run_name: str,
    ) -> None:
        """Benchmark sonuçlarını DB'ye yazar."""
        if not self.repo or not records:
            return

        from src.config.db_tables import TABLES
        from psycopg2.extras import execute_values

        conn = self.repo.get_connection()
        if not conn:
            return

        try:
            with conn.cursor() as cur:
                execute_values(cur, f"""
                    INSERT INTO {TABLES['benchmark_result']} (
                        test_case_id, benchmark_run_name,
                        predicted_label, predicted_company_id, predicted_score,
                        retrieval_rank,
                        recall_at_1, recall_at_5, recall_at_10,
                        recall_at_20, recall_at_30, reciprocal_rank,
                        is_correct, is_true_positive, is_false_positive,
                        is_true_negative, is_false_negative,
                        processing_time_ms, reason_codes
                    )
                    VALUES %s
                """, [
                    (
                        r.test_case_id, benchmark_run_name,
                        r.predicted_label, r.predicted_company_id, r.predicted_score,
                        r.retrieval_rank,
                        r.recall_at_1, r.recall_at_5, r.recall_at_10,
                        r.recall_at_20, r.recall_at_30, r.reciprocal_rank,
                        r.is_correct, r.is_true_positive, r.is_false_positive,
                        r.is_true_negative, r.is_false_negative,
                        r.processing_time_ms, json.dumps(r.reason_codes),
                    )
                    for r in records
                ])
            conn.commit()
            logger.info(f"Saved {len(records)} benchmark results to DB.")
        except Exception as e:
            logger.error(f"Error saving benchmark results: {e}")
            conn.rollback()
        finally:
            self.repo.release_connection(conn)
