import sys
import os
import warnings
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository
from src.config.db_tables import TABLES


def is_empty_val(val):
    if val is None or pd.isna(val):
        return True
    s = str(val).strip()
    return s == "" or s == "None"


def is_missing_expected(val):
    if is_empty_val(val):
        return True
    return str(val).strip().casefold() == "unknown"


def normalize_company_name(value):
    if is_empty_val(value):
        return ""
    return " ".join(str(value).casefold().split())


def categorize_fn(row):
    retrieved = row.get("retrieved_company")
    if is_empty_val(retrieved):
        return "NO_CANDIDATE"
    
    expected = row.get("expected_company")
    if is_missing_expected(expected):
        return "MISSING_EXPECTED_COMPANY"
        
    if normalize_company_name(expected) != normalize_company_name(retrieved):
        return "WRONG_CANDIDATE"
        
    pred_label = row.get("predicted_label")
    final_score = row.get("final_score")
    if (pd.notna(pred_label) and str(pred_label).strip().upper() == "NO_MATCH") or (pd.isna(pred_label) and pd.notna(final_score)):
        return "BELOW_THRESHOLD"
        
    return "UNKNOWN"


def analyze_false_negatives():
    config_loader = ConfigLoader()
    db_cfg = config_loader.get_db_config()
    repo = AMLRepository(
        host=db_cfg.get("host"), port=db_cfg.get("port"),
        dbname=db_cfg.get("name"), user=db_cfg.get("user"), password=db_cfg.get("password")
    )
    conn = repo.get_connection()
    if not conn:
        print("Veritabanı bağlantısı kurulamadı.")
        return

    query = f"""
    SELECT 
        d.expected_company,
        d.retrieved_company,
        d.fuzzy_score,
        d.vector_score,
        d.reranker_score,
        d.final_score,
        d.predicted_label,
        d.reason_code,
        e.explanation,
        e.eft_id
    FROM {TABLES.get("decision_analysis", "aml_experiment.decision_analysis")} d
    LEFT JOIN {TABLES.get("eft_input", "bronze_eft_raw")} e ON e.eft_id::varchar = d.eft_id::varchar
    WHERE d.error_type = 'FN'
    """
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Sorgu çalıştırılırken hata oluştu: {e}")
        return
    finally:
        repo.release_connection(conn)

    # 1. Toplam FN sayısı
    print(f"Toplam False Negative (FN) Sayısı: {len(df)}")
    if len(df) == 0:
        print("İncelenecek False Negative kaydı bulunamadı.")
        return

    # Eksik olabilecek alanları güvenli varsayılanlarla doldur
    for col in ["expected_company", "retrieved_company", "reason_code", "predicted_label", "explanation"]:
        if col not in df.columns:
            df[col] = None

    # 9. Hassas veri ve uzun çıktı kontrolü (maks 200 karakter)
    df["explanation_preview"] = df["explanation"].fillna("").astype(str).str.slice(0, 200)
    df["fn_category"] = df.apply(categorize_fn, axis=1)

    # 2. FN kategori dağılımı
    print("\n--- FN Kategori Dağılımı ---")
    cat_counts = df["fn_category"].value_counts()
    for cat, count in cat_counts.items():
        print(f"  {cat}: {count} ({count/len(df):.1%})")

    # 3. Reason code dağılımı
    print("\n--- Reason Code Dağılımı ---")
    if "reason_code" in df.columns and not df["reason_code"].isnull().all() and (df["reason_code"] != "").any():
        reason_counts = (
            df["reason_code"]
            .fillna("UNKNOWN")
            .replace("", "UNKNOWN")
            .value_counts()
        )
        for code, count in reason_counts.items():
            print(f"  {code}: {count} ({count/len(df):.1%})")
    else:
        print("  Sorgu sonucunda 'reason_code' kolonu bulunmuyor veya tümü boş.")

    # 4. Retrieval analizi
    print("\n--- Retrieval Analizi ---")
    no_cand_count = (df["fn_category"] == "NO_CANDIDATE").sum()
    print(f"Retrieval aşamasında hiç aday bulunamayan kayıt sayısı: {no_cand_count}")
    
    if df["expected_company"].apply(is_missing_expected).all():
        print("Beklenen şirket bilgisi bulunmadığı için yanlış aday analizi yapılamadı.")
    else:
        wrong_cand_count = (df["fn_category"] == "WRONG_CANDIDATE").sum()
        print(f"Aday bulunmuş ancak beklenen şirketten farklı olan kayıt sayısı: {wrong_cand_count}")

    # 5. Skor istatistikleri
    print("\n--- Skor İstatistikleri ---")
    candidates_df = df[~df["retrieved_company"].apply(is_empty_val)].copy()
    if len(candidates_df) == 0:
        print("  Aday bulunan kayıt olmadığı için skor istatistiği hesaplanamadı.")
    else:
        score_cols = ["fuzzy_score", "vector_score", "reranker_score", "final_score"]
        for col in score_cols:
            if col not in candidates_df.columns:
                print(f"  {col}: Kolon bulunamadı.")
                continue
            s = pd.to_numeric(candidates_df[col], errors="coerce").dropna()
            if len(s) == 0:
                print(f"  {col}: N/A (Bütün değerler null)")
            else:
                print(f"  {col}: ortalama={s.mean():.4f}, medyan={s.median():.4f}, minimum={s.min():.4f}, maksimum={s.max():.4f}")

    # 6. Threshold’a en yakın 5 FN
    print("\n--- Threshold’a En Yakın 5 False Negative ---")
    cols_to_show = [
        "eft_id", "explanation_preview", "expected_company", 
        "retrieved_company", "fuzzy_score", "vector_score", 
        "reranker_score", "final_score", "reason_code"
    ]
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    
    if "final_score" in df.columns and not df["final_score"].isnull().all():
        df["_final_score_num"] = pd.to_numeric(df["final_score"], errors="coerce")
        top5_close = df.sort_values("_final_score_num", ascending=False).head(5)
        print(top5_close[cols_to_show].to_string(index=False))
    else:
        print("  final_score bilgisi bulunmadığı veya null olduğu için sıralama yapılamadı.")

    # 7. İlk 5 genel FN örneği
    print("\n--- İlk 5 Genel FN Örneği ---")
    print(df[cols_to_show].head(5).to_string(index=False))


if __name__ == "__main__":
    analyze_false_negatives()
