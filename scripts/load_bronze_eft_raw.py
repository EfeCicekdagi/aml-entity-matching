import sys
import os
import logging
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from psycopg2.extras import execute_values

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository
from src.config.db_tables import TABLES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "aml_eft_challenge_dataset_1440_berke_final.csv")


def check_or_create_table(cur, target_table: str):
    if "." in target_table:
        schema, table = target_table.split(".", 1)
    else:
        schema, table = "public", target_table
    
    cur.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_schema = %s AND table_name = %s
    """, (schema, table))
    existing_cols = {row[0].lower() for row in cur.fetchall()}
    
    expected_cols = {
        "eft_id", "transaction_date", "amount", "explanation", 
        "sender_account_id", "receiver_account_id", "batch_id", "source_system"
    }
    
    if not existing_cols:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {target_table} (
                eft_id BIGINT PRIMARY KEY,
                transaction_date DATE,
                amount NUMERIC,
                explanation TEXT,
                sender_account_id TEXT,
                receiver_account_id TEXT,
                batch_id TEXT,
                source_system TEXT
            );
        """)
        logger.info(f"Tablo mevcut değildi, DDL ile oluşturuldu: {target_table}")
    else:
        if not expected_cols.issubset(existing_cols):
            missing = expected_cols - existing_cols
            raise ValueError(f"Hedef tablo {target_table} mevcut ancak kolon yapısı uyumsuz! Eksik kolonlar: {missing}")


def load_data(
    csv_path: str = DEFAULT_CSV_PATH,
    replace_existing: bool = False,
) -> bool:
    if not os.path.exists(csv_path):
        logger.error(f"Error: {csv_path} not found.")
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"CSV dosyası okunamadı: {e}")
        return False

    csv_total_count = len(df)
    logger.info(f"CSV okundu: {csv_path} ({csv_total_count:,} kayıt)")

    # 1. Kolon isimlerini normalize et
    df.columns = [c.lower().strip() for c in df.columns]

    # 2. Alias kolonları map et (eksik explanation kontrolü)
    if "explanation" not in df.columns:
        if "description" in df.columns:
            df.rename(columns={"description": "explanation"}, inplace=True)
        elif "text" in df.columns:
            df.rename(columns={"text": "explanation"}, inplace=True)
        else:
            logger.error("CSV içinde explanation, description veya text kolonu bulunamadı.")
            raise ValueError("CSV içinde explanation, description veya text kolonu bulunamadı.")

    # 3. Eksik kolonları tamamla
    missing_cols = []
    if "eft_id" not in df.columns:
        missing_cols.append("eft_id")
        df["eft_id"] = range(1, len(df) + 1)
    if "transaction_date" not in df.columns:
        missing_cols.append("transaction_date")
        df["transaction_date"] = pd.Timestamp.now().floor("s")
    if "amount" not in df.columns:
        missing_cols.append("amount")
        df["amount"] = 0.0
    if "sender_account_id" not in df.columns:
        missing_cols.append("sender_account_id")
        df["sender_account_id"] = "TEST_SENDER"
    if "receiver_account_id" not in df.columns:
        missing_cols.append("receiver_account_id")
        df["receiver_account_id"] = "TEST_RECEIVER"
    if "source_system" not in df.columns:
        missing_cols.append("source_system")
        df["source_system"] = "TEST_CSV"
    if "batch_id" not in df.columns:
        missing_cols.append("batch_id")
        df["batch_id"] = f"TEST-BATCH-{datetime.now():%Y%m%d%H%M%S}"

    if missing_cols:
        logger.warning(f"UYARI: CSV içinde bazı kolonlar eksik olduğundan test defaultları atandı. Eksik kolonlar: {missing_cols}")
        logger.warning("Bu yükleme yalnızca test ve demo amaçlı çalışmaktadır.")

    current_batch_id = str(df["batch_id"].iloc[0]) if len(df) > 0 and pd.notna(df["batch_id"].iloc[0]) else "TEST_BATCH_UNKNOWN"

    # 4. Gerekli kolonları seç
    required_cols = [
        "eft_id", "transaction_date", "amount", "explanation", 
        "sender_account_id", "receiver_account_id", "batch_id", "source_system"
    ]
    df = df[required_cols].copy()

    # 5. Veri doğrulaması yap
    # eft_id doğrulaması
    df["eft_id"] = pd.to_numeric(df["eft_id"], errors="coerce")
    if df["eft_id"].isna().any():
        raise ValueError("CSV içinde null veya sayısal olmayan (geçersiz) eft_id değerleri bulundu.")
    if (df["eft_id"] <= 0).any():
        raise ValueError("CSV içinde pozitif olmayan (<= 0) eft_id değerleri bulundu.")
    dups = df[df["eft_id"].duplicated()]["eft_id"].unique()
    if len(dups) > 0:
        dup_sample = list(dups[:10])
        raise ValueError(f"CSV içinde tekrar eden eft_id bulundu: {dup_sample}")
    df["eft_id"] = df["eft_id"].astype(int)

    # Tarih doğrulaması
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    invalid_dates_mask = df["transaction_date"].isna()
    if invalid_dates_mask.any():
        invalid_count = invalid_dates_mask.sum()
        sample_ids = df.loc[invalid_dates_mask, "eft_id"].head(5).tolist()
        logger.error(f"Geçersiz tarih bilgisine sahip {invalid_count} kayıt bulundu! Örnek eft_id: {sample_ids}")
        raise ValueError(f"CSV içinde geçersiz transaction_date bulunan {invalid_count} kayıt var.")

    # Tutar doğrulaması
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    if df["amount"].isna().any() or np.isinf(df["amount"]).any():
        raise ValueError("CSV içinde NaN veya sonsuz (infinite) tutar (amount) değerleri bulundu.")
    if (df["amount"] < 0).any():
        neg_sample = df.loc[df["amount"] < 0, "eft_id"].head(5).tolist()
        raise ValueError(f"CSV içinde negatif tutar (amount) değerleri bulundu! Örnek eft_id: {neg_sample}")

    # Açıklama doğrulaması
    null_exp_count = df["explanation"].isna().sum()
    if null_exp_count > 0:
        raise ValueError(f"CSV içinde {null_exp_count} kayıtta null (eksik) explanation bilgisi bulunuyor.")
    empty_exp_mask = df["explanation"].astype(str).str.strip() == ""
    empty_exp_count = empty_exp_mask.sum()
    if empty_exp_count > 0:
        sample_ids = df.loc[empty_exp_mask, "eft_id"].head(5).tolist()
        raise ValueError(f"CSV içinde tamamen boş veya sadece boşluktan oluşan {empty_exp_count} explanation kaydı bulundu! Örnek eft_id: {sample_ids}")

    validated_count = len(df)
    error_count = 0  # Başarılı doğrulamada hata yok

    # 6. Veri tiplerini dönüştür (PostgreSQL için temiz Python tipleri)
    df["transaction_date_str"] = df["transaction_date"].dt.strftime("%Y-%m-%d")
    records = [
        (
            int(row["eft_id"]),
            str(row["transaction_date_str"]),
            float(row["amount"]),
            str(row["explanation"]).strip(),
            str(row["sender_account_id"]).strip() if pd.notna(row["sender_account_id"]) else None,
            str(row["receiver_account_id"]).strip() if pd.notna(row["receiver_account_id"]) else None,
            str(row["batch_id"]).strip() if pd.notna(row["batch_id"]) else None,
            str(row["source_system"]).strip() if pd.notna(row["source_system"]) else None,
        )
        for _, row in df.iterrows()
    ]

    # 7. DB'ye yaz
    config_loader = ConfigLoader()
    db_cfg = config_loader.get_db_config()
    repo = AMLRepository(
        host=db_cfg.get("host"), port=db_cfg.get("port"),
        dbname=db_cfg.get("name"), user=db_cfg.get("user"), password=db_cfg.get("password")
    )
    conn = repo.get_connection()
    if not conn:
        logger.error("Database connection failed.")
        return False

    target_table = TABLES.get("bronze_eft_raw", "public.bronze_eft_raw")

    try:
        with conn.cursor() as cur:
            check_or_create_table(cur, target_table)

            cur.execute(f"SELECT COUNT(*) FROM {target_table};")
            count_before = cur.fetchone()[0]

            if replace_existing:
                logger.warning(f"UYARI: {target_table} tablosundaki mevcut veriler silinecek.")
                cur.execute(f"TRUNCATE TABLE {target_table};")
                count_before = 0
                query = f"""
                    INSERT INTO {target_table} (
                        eft_id, transaction_date, amount, explanation, 
                        sender_account_id, receiver_account_id, batch_id, source_system
                    ) VALUES %s
                """
            else:
                query = f"""
                    INSERT INTO {target_table} (
                        eft_id, transaction_date, amount, explanation, 
                        sender_account_id, receiver_account_id, batch_id, source_system
                    ) VALUES %s
                    ON CONFLICT (eft_id) DO NOTHING
                """

            execute_values(cur, query, records, page_size=1000)

            cur.execute(f"SELECT COUNT(*) FROM {target_table};")
            count_after = cur.fetchone()[0]
            db_added = count_after - count_before
            dup_skipped = len(records) - db_added if not replace_existing else 0
            if dup_skipped < 0:
                dup_skipped = 0

        conn.commit()

        logger.info("\n=== YÜKLEME RAPORU ===")
        logger.info(f"Target table: {target_table}")
        logger.info(f"Batch ID: {current_batch_id}")
        logger.info(f"Replace mode: {replace_existing}")
        logger.info(f"CSV toplam kayıt: {csv_total_count:,}")
        logger.info(f"Doğrulanan kayıt: {validated_count:,}")
        logger.info(f"DB’ye eklenen kayıt: {db_added:,}")
        logger.info(f"Duplicate nedeniyle atlanan: {dup_skipped:,}")
        logger.info(f"Hatalı kayıt: {error_count:,}")

        print(f"Loaded {db_added:,} new records into {target_table} successfully (skipped {dup_skipped:,} duplicates).")
        return True

    except Exception as e:
        conn.rollback()
        logger.error(f"Veritabanına yazılırken hata oluştu: {e}")
        return False
    finally:
        repo.release_connection(conn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EFT ham verisi yükleme betiği.")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Yüklenecek CSV dosyasının yolu.")
    parser.add_argument("--replace", action="store_true", help="Mevcut verileri silip yenisiyle değiştirir.")
    args = parser.parse_args()

    success = load_data(csv_path=args.csv, replace_existing=args.replace)
    if not success:
        sys.exit(1)
    sys.exit(0)

