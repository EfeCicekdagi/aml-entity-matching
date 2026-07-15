"""
AML Evaluation Script
Tüm değerlendirme işlemleri PostgreSQL tablolarından yapılır.
CSV dosyalarına bağımlılık yoktur.
"""
import sys
import os
import psycopg2
from sklearn.metrics import precision_score, recall_score, f1_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config_loader import ConfigLoader


def get_db_connection():
    config_loader = ConfigLoader()
    db = config_loader.get_db_config()
    return psycopg2.connect(
        host=db['host'], port=db['port'],
        dbname=db['name'], user=db['user'], password=db['password']
    )


def evaluate(run_id: str = None):
    """
    PostgreSQL'deki aml_alert ve aml_ground_truth tablolarını kullanarak
    son (veya belirtilen) run'ı değerlendirir.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # En son başarılı run'ı bul (run_id verilmediyse)
    if not run_id:
        cur.execute("""
            SELECT run_id FROM aml_run_log
            WHERE status = 'SUCCESS'
            ORDER BY finished_at DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            print("Hata: Tamamlanmış başarılı bir run bulunamadı.")
            conn.close()
            return None, None, None, None
        run_id = row[0]

    print(f"Değerlendirilen Run: {run_id}")

    # aml_alert tablosundan bu run'ın sonuçlarını çek
    cur.execute("""
        SELECT
            a.eft_id::TEXT,
            a.company_id,
            a.risk_level
        FROM aml_alert a
        WHERE a.run_id = %s
    """, (run_id,))
    alert_rows = cur.fetchall()

    # Sonuçları dict'e çevir: eft_id -> (company_id, risk_level)
    results = {}
    for eft_id, company_id, risk_level in alert_rows:
        # EFT_00001 formatına normalize et
        if not eft_id.startswith("EFT_"):
            eft_id = f"EFT_{str(eft_id).zfill(5)}"
        results[eft_id] = (company_id, risk_level)

    # Ground truth tablosunu çek
    cur.execute("SELECT eft_id, true_company_id FROM aml_ground_truth")
    gt_rows = cur.fetchall()

    if not gt_rows:
        print("Hata: aml_ground_truth tablosu boş.")
        conn.close()
        return None, None, None, None

    # Değerlendirme
    y_true, y_pred = [], []
    exact_matches = 0
    total_known = 0

    for eft_id, true_company_id in gt_rows:
        is_match_true = 1 if true_company_id != -1 else 0
        if is_match_true:
            total_known += 1

        pred = results.get(eft_id, (None, "No Match"))
        pred_company_id, pred_risk = pred
        is_match_pred = 1 if (pred_company_id is not None and pred_risk not in ["No Match", "LOW"]) else 0

        y_true.append(is_match_true)
        y_pred.append(is_match_pred)

        if true_company_id != -1 and pred_company_id == true_company_id:
            exact_matches += 1

    precision  = precision_score(y_true, y_pred, zero_division=0)
    recall     = recall_score(y_true, y_pred, zero_division=0)
    f1         = f1_score(y_true, y_pred, zero_division=0)
    exact_acc  = exact_matches / total_known if total_known > 0 else 0

    # Alert dağılımı
    cur.execute("""
        SELECT risk_level, COUNT(*)
        FROM aml_alert WHERE run_id = %s
        GROUP BY risk_level ORDER BY COUNT(*) DESC
    """, (run_id,))
    risk_dist = cur.fetchall()

    # Log the metrics to the run_log table
    try:
        cur.execute("""
            UPDATE aml_run_log 
            SET precision_score = %s,
                recall_score = %s,
                f1_score = %s,
                exact_match_score = %s
            WHERE run_id = %s
        """, (float(precision), float(recall), float(f1), float(exact_acc), run_id))
        conn.commit()
    except Exception as e:
        print(f"Metrics update error: {e}")
        conn.rollback()

    conn.close()

    # Rapor
    print("-" * 55)
    print("EVALUATION RESULTS")
    print("-" * 55)
    print(f"Toplam Ground Truth EFT : {len(gt_rows)}")
    print(f"Toplam Bilinen Eşleşme  : {total_known}")
    print(f"Toplam Alert (bu run)   : {len(alert_rows)}")
    print("-" * 55)
    print(f"Risk Dağılımı:")
    for risk, cnt in risk_dist:
        print(f"  {risk:<12}: {cnt:,}")
    print("-" * 55)
    print(f"Precision               : {precision:.4f}")
    print(f"Recall                  : {recall:.4f}")
    print(f"F1 Score                : {f1:.4f}")
    print(f"Exact Match Accuracy    : {exact_acc:.4f}")
    print("-" * 55)

    return precision, recall, f1, exact_acc


if __name__ == "__main__":
    run_arg = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate(run_id=run_arg)
