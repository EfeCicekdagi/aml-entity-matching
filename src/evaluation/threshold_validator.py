"""
threshold_validator.py — Threshold analiz ve öneri scripti.

Validation veri seti üzerinden farklı threshold değerlerinde:
  - Precision, Recall, F1
  - False Positive / False Negative sayısı
  - Alert volume, 1.000 işlem başına alert
metrikleri hesaplar ve öneri raporu oluşturur.

ÖNEMLİ: Bu script production config'i ASLA değiştirmez.
Sadece öneri raporu üretir.

Kullanım:
    python -m src.evaluation.threshold_validator \
        --scores outputs/scores.csv \
        --output outputs/threshold_report.json
"""

import json
import logging
import argparse
from typing import Optional
import sys, os

logger = logging.getLogger(__name__)


def analyze_thresholds(
    scores: list[dict],
    high_range: tuple = (0.55, 0.95),
    medium_range: tuple = (0.45, 0.85),
    step: float = 0.01,
) -> list[dict]:
    """
    Farklı threshold kombinasyonlarında metrikleri hesaplar.

    Args:
        scores: Her kayıt için {"score": float, "label": int (1=MATCH, 0=NO_MATCH)} içeren liste
        high_range: HIGH threshold arama aralığı (min, max)
        medium_range: MEDIUM threshold arama aralığı (min, max)
        step: Threshold adım büyüklüğü

    Returns:
        Her threshold kombinasyonu için metrik listesi
    """
    results = []
    n_total = len(scores)

    high_values   = [round(h, 2) for h in _frange(high_range[0], high_range[1], step)]
    medium_values = [round(m, 2) for m in _frange(medium_range[0], medium_range[1], step)]

    for high_t in high_values:
        for medium_t in medium_values:
            if medium_t >= high_t:
                continue  # MEDIUM her zaman HIGH'dan küçük olmalı

            tp = fp = tn = fn = 0
            alert_count = 0

            for rec in scores:
                score = rec["score"]
                label = rec.get("label", 0)
                # Tahmin
                if score >= high_t or score >= medium_t:
                    predicted = 1
                    alert_count += 1
                else:
                    predicted = 0

                if label == 1 and predicted == 1:
                    tp += 1
                elif label == 0 and predicted == 1:
                    fp += 1
                elif label == 0 and predicted == 0:
                    tn += 1
                elif label == 1 and predicted == 0:
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            results.append({
                "high_threshold":    high_t,
                "medium_threshold":  medium_t,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "precision":         round(precision, 4),
                "recall":            round(recall, 4),
                "f1":                round(f1, 4),
                "fpr":               round(fpr, 4),
                "fnr":               round(fnr, 4),
                "alert_count":       alert_count,
                "alerts_per_1k":     round(alert_count / n_total * 1000, 2) if n_total > 0 else 0,
            })

    return results


def find_optimal_threshold(
    results: list[dict],
    optimization_metric: str = "f1",
    max_fpr: float = 0.10,
) -> Optional[dict]:
    """
    Metrik bazlı en iyi threshold kombinasyonunu bulur.

    Args:
        results: analyze_thresholds() çıktısı
        optimization_metric: Optimize edilecek metrik ('f1', 'precision', 'recall')
        max_fpr: Maksimum izin verilen False Positive Rate

    Returns:
        En iyi threshold kombinasyonu
    """
    filtered = [r for r in results if r["fpr"] <= max_fpr]
    if not filtered:
        filtered = results  # FPR kısıtı karşılanamıyorsa tümünü değerlendir

    return max(filtered, key=lambda r: r[optimization_metric], default=None)


def generate_report(
    scores: list[dict],
    current_high: float = 0.70,
    current_medium: float = 0.62,
    validation_dataset: str = "unknown",
) -> dict:
    """
    Tam threshold validasyon raporu üretir.

    Args:
        scores: Skor ve etiket listesi
        current_high: Mevcut HIGH threshold
        current_medium: Mevcut MEDIUM threshold
        validation_dataset: Doğrulama veri seti adı

    Returns:
        Rapor dict
    """
    logger.info(f"Running threshold analysis on {len(scores)} samples...")

    all_results = analyze_thresholds(scores)

    # Mevcut threshold için metrikler
    current_metrics = next(
        (r for r in all_results
         if r["high_threshold"] == current_high
         and r["medium_threshold"] == current_medium),
        None
    )

    # En iyi F1
    best_f1     = find_optimal_threshold(all_results, "f1")
    best_recall = find_optimal_threshold(all_results, "recall", max_fpr=0.15)

    report = {
        "validation_dataset":   validation_dataset,
        "n_samples":            len(scores),
        "current_thresholds": {
            "high_threshold":   current_high,
            "medium_threshold": current_medium,
        },
        "current_metrics":      current_metrics,
        "recommendations": {
            "best_f1":     best_f1,
            "best_recall": best_recall,
        },
        "note": (
            "Bu rapor sadece öneri niteliğindedir. "
            "Production config'i otomatik olarak değiştirmez. "
            "Onaylanan threshold değerleri manuel olarak uygulanmalıdır."
        ),
        "all_results_count": len(all_results),
    }

    logger.info(
        f"Threshold analysis complete. "
        f"Current F1={current_metrics.get('f1', 'N/A') if current_metrics else 'N/A'}, "
        f"Best F1={best_f1.get('f1', 'N/A') if best_f1 else 'N/A'} "
        f"(high={best_f1.get('high_threshold') if best_f1 else 'N/A'}, "
        f"medium={best_f1.get('medium_threshold') if best_f1 else 'N/A'})"
    )

    return report


def _frange(start: float, stop: float, step: float):
    """Float range generator."""
    current = start
    while current <= stop + 1e-9:
        yield current
        current += step


def load_scores_from_csv(csv_path: str) -> list[dict]:
    """
    CSV dosyasından skor listesi yükler.
    CSV formatı: score,label (0 veya 1)

    Args:
        csv_path: CSV dosya yolu

    Returns:
        [{"score": float, "label": int}, ...] listesi
    """
    import csv
    scores = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append({
                "score": float(row.get("score", row.get("final_score", 0))),
                "label": int(row.get("label", row.get("expected_label", 0))),
            })
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Threshold Validator")
    parser.add_argument("--scores",  required=True, help="CSV: score,label")
    parser.add_argument("--output",  default="outputs/threshold_report.json")
    parser.add_argument("--high",    type=float, default=0.70)
    parser.add_argument("--medium",  type=float, default=0.62)
    parser.add_argument("--dataset", default="validation_v1")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    scores = load_scores_from_csv(args.scores)
    report = generate_report(
        scores,
        current_high    = args.high,
        current_medium  = args.medium,
        validation_dataset = args.dataset,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nRapor kaydedildi: {args.output}")
    if report.get("current_metrics"):
        cm = report["current_metrics"]
        print(f"Mevcut (high={args.high}, medium={args.medium}): "
              f"Precision={cm['precision']:.3f}, Recall={cm['recall']:.3f}, F1={cm['f1']:.3f}")
    if report.get("recommendations", {}).get("best_f1"):
        bf = report["recommendations"]["best_f1"]
        print(f"Önerilen (F1 bazlı): high={bf['high_threshold']}, medium={bf['medium_threshold']}, "
              f"F1={bf['f1']:.3f}")
