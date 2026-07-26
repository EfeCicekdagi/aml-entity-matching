import sys
import os
import logging
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository
from src.config.db_tables import TABLES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clear_results(
    dry_run: bool = False,
    include_run_logs: bool = False,
    include_audit_history: bool = False,
):
    if dry_run:
        logger.info("[DRY-RUN] Önceki çalışmalara ait işlem sonuçları ve önbellek (cache) kontrol ediliyor...")
    else:
        logger.info("Önceki çalışmalara ait işlem sonuçları ve önbellek (cache) temizleniyor...")

    config_loader = ConfigLoader()
    db_config = config_loader.get_db_config()

    repo = AMLRepository(
        host=db_config.get("host"),
        port=db_config.get("port"),
        dbname=db_config.get("name"),
        user=db_config.get("user"),
        password=db_config.get("password")
    )
    
    conn = repo.get_connection()
    if not conn:
        logger.error("Veritabanı bağlantısı kurulamadı.")
        return

    # 2. Temizlenecek tabloları mantıksal sıraya koy (önce child/result, sonra parent/run)
    all_possible_tables = [
        ("alert_export", False, False),
        ("alert_history", False, True),  # audit history
        ("alert", False, False),
        ("match_result", False, False),
        ("candidate_match", False, False),
        ("scoring_result", False, False),
        ("reranker_cache", False, False),
        ("performance_log", False, True), # audit history
        ("quality_check", False, True),   # audit history
        ("run_log", True, False),         # run logs
    ]

    tables_to_clear = []
    for key, needs_run_log, needs_audit in all_possible_tables:
        if needs_run_log and not include_run_logs:
            continue
        if needs_audit and not include_audit_history:
            continue
        table_name = TABLES.get(key)
        if table_name is not None:
            tables_to_clear.append(table_name)

    # 3. Varsayılan temizliği yalnızca sonuç tablolarıyla sınırla (korumalı tablolar)
    forbidden_prefixes = ("aml_source.", "aml_config.", "aml_eval.", "aml_experiment.")
    forbidden_tables = {
        "aml_stage.company_variant", "aml_stage.company_detail",
        "aml_stage.person_detail", "aml_stage.vessel_detail",
        "aml_ml.company_embedding"
    }

    safe_tables_to_clear = []
    for t in tables_to_clear:
        if any(t.startswith(p) for p in forbidden_prefixes) or t in forbidden_tables:
            logger.warning(f"Güvenlik uyarısı: {t} korumalı bir tablodur ve temizlenemez.")
            continue
        safe_tables_to_clear.append(t)

    if include_run_logs and TABLES.get("run_log") in safe_tables_to_clear:
        logger.warning("UYARI: run_log temizleniyor; geçmiş pipeline run kayıtları silinecek.")

    success_count = 0
    skipped_count = 0
    error_count = 0
    total_deleted = 0
    error_tables = []

    try:
        with conn.cursor() as cur:
            for table in safe_tables_to_clear:
                cur.execute("SAVEPOINT clear_table")
                try:
                    # 4. Eksik tablo kontrolü ekle
                    cur.execute("SELECT to_regclass(%s)", (table,))
                    exists = cur.fetchone()[0] is not None
                    if not exists:
                        print(f"SKIPPED: {table} bulunamadı.")
                        skipped_count += 1
                        cur.execute("RELEASE SAVEPOINT clear_table")
                        continue

                    # 8. Temizlik öncesi kayıt sayılarını göster
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cur.fetchone()[0]
                    print(f"{table} temizleniyor... {count:,} kayıt")

                    if not dry_run:
                        # 1. CASCADE kullanımını kaldır
                        cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY;")
                        print(f"OK: {table} temizlendi.")
                    else:
                        print(f"[DRY-RUN] OK: {table} temizlenecek.")

                    success_count += 1
                    total_deleted += count
                    cur.execute("RELEASE SAVEPOINT clear_table")

                except Exception as e:
                    cur.execute("ROLLBACK TO SAVEPOINT clear_table")
                    cur.execute("RELEASE SAVEPOINT clear_table")
                    error_count += 1
                    error_tables.append(table)
                    err_str = str(e).casefold()
                    if "foreign key" in err_str or getattr(e, "pgcode", "") in ("0A000", "23503") or "referenced in a foreign key" in err_str:
                        print("Tablo foreign key bağımlılığı nedeniyle temizlenemedi.\nCASCADE güvenlik nedeniyle otomatik uygulanmadı.")
                    else:
                        print(f"HATA: {table} temizlenirken hata oluştu: {e}")

        if not dry_run:
            conn.commit()
            logger.info("✅ Önceki işlem kayıtları ve önbellek başarıyla temizlendi. Sistem yeni analizler için hazır.")
        else:
            conn.rollback()
            logger.info("ℹ️ Dry-run tamamlandı; veritabanında hiçbir değişiklik yapılmadı.")

    except Exception as e:
        logger.error(f"Tablolar temizlenirken beklenmeyen hata oluştu: {e}")
        conn.rollback()
    finally:
        repo.release_connection(conn)

    # 10. Sonuç özeti ekle
    print("\n=== TEMİZLİK ÖZETİ ===")
    print(f"Başarılı: {success_count}")
    print(f"Atlanan: {skipped_count}")
    print(f"Hatalı: {error_count}")
    print(f"Silinen toplam kayıt: {total_deleted:,}")

    if error_tables:
        print(f"Hatalı tablolar: {', '.join(error_tables)}")
        print("Temizlik kısmen tamamlandı. Sistem yeni run için tamamen temiz olmayabilir.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML sonuç ve önbellek temizleme betiği.")
    parser.add_argument("--dry-run", action="store_true", help="Değişiklik yapmadan hangi tabloların temizleneceğini gösterir.")
    parser.add_argument("--confirm", action="store_true", help="Gerçek veri silme işlemini onaylar.")
    parser.add_argument("--include-run-logs", action="store_true", help="run_log tablosunu da temizler.")
    parser.add_argument("--include-audit-history", action="store_true", help="alert_history gibi denetim geçmişi tablolarını temizler.")
    args = parser.parse_args()

    if not args.confirm and not args.dry_run:
        clear_results(
            dry_run=True,
            include_run_logs=args.include_run_logs,
            include_audit_history=args.include_audit_history
        )
        print("\nVeri silme uygulanmadı. Gerçek temizlik için --confirm kullanın.")
    elif args.dry_run:
        clear_results(
            dry_run=True,
            include_run_logs=args.include_run_logs,
            include_audit_history=args.include_audit_history
        )
    else:
        clear_results(
            dry_run=False,
            include_run_logs=args.include_run_logs,
            include_audit_history=args.include_audit_history
        )

