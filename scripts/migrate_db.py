"""
migrate_db.py — AML veritabanı migration çalıştırıcı.

Kullanım:
    python scripts/migrate_db.py              → tüm yeni migration'ları çalıştır
    python scripts/migrate_db.py --dry-run   → sadece hangi dosyaların çalışacağını göster
    python scripts/migrate_db.py --from 13   → 13 numaralı ve sonrası migration'ları çalıştır
"""

import sys
import os
import logging
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Çalıştırılacak migration dosyaları (sıralı)
MIGRATIONS = [
    "01_create_extensions_and_schemas.sql",
    "02_create_source_views.sql",
    "03_create_stage_tables.sql",
    "04_create_ml_tables.sql",
    "05_create_core_tables.sql",
    "06_create_config_tables.sql",
    "07_create_audit_tables.sql",
    "08_create_indexes.sql",
    "10_add_match_reason_to_alerts.sql",
    "11_update_alert_structure.sql",
    "12_create_match_result_table.sql",
    "13_extend_alert_table.sql",
    "14_extend_run_log_table.sql",
    "15_extend_company_variant_table.sql",
    "16_create_alert_history_table.sql",
    "17_create_benchmark_tables.sql",
    "18_create_model_governance_table.sql",
    "19_create_auxiliary_entity_fields.sql",
    "20_create_alert_export_table.sql",
    "21_create_experiment_tables.sql",
    "22_add_compact_match_fields.sql",
]


def migrate(from_number: int = 1, dry_run: bool = False):
    config_loader = ConfigLoader()
    db_config = config_loader.get_db_config()

    repo = AMLRepository(
        host=db_config.get("host"),
        port=db_config.get("port"),
        dbname=db_config.get("name"),
        user=db_config.get("user"),
        password=db_config.get("password")
    )

    sql_dir = os.path.join(os.path.dirname(__file__), '..', 'sql')

    # from_number filtresine göre migration'ları seç
    selected = [f for f in MIGRATIONS if int(f.split("_")[0]) >= from_number]

    if not selected:
        logger.info("Çalıştırılacak migration yok.")
        return

    print("\n" + "="*60)
    print(f"  AML Migration {'(DRY RUN)' if dry_run else 'Başlatılıyor'}")
    print(f"  Toplam {len(selected)} migration çalıştırılacak")
    print("="*60)

    results = []
    for i, migration_file in enumerate(selected, 1):
        script_path = os.path.join(sql_dir, migration_file)

        if not os.path.exists(script_path):
            logger.warning(f"  [{i}/{len(selected)}] [!] DOSYA YOK: {migration_file}")
            results.append((migration_file, "SKIPPED"))
            continue

        if dry_run:
            print(f"  [{i}/{len(selected)}] [DRY-RUN] {migration_file}")
            results.append((migration_file, "DRY_RUN"))
            continue

        try:
            print(f"  [{i}/{len(selected)}] [...] Calistiriliyor: {migration_file}")
            repo.execute_script(script_path)
            print(f"  [{i}/{len(selected)}] [OK] BASARILI: {migration_file}")
            results.append((migration_file, "OK"))
        except Exception as e:
            print(f"  [{i}/{len(selected)}] [FAIL] HATA: {migration_file}")
            print(f"        {e}")
            results.append((migration_file, f"FAILED: {e}"))

    print("\n" + "="*60)
    print("  OZET:")
    for fname, status in results:
        icon = "[OK]" if status == "OK" else ("[DRY]" if status == "DRY_RUN" else ("[SKIP]" if status == "SKIPPED" else "[FAIL]"))
        print(f"  {icon} {fname} -> {status}")
    print("="*60 + "\n")

    failed = [r for r in results if r[1].startswith("FAILED")]
    if failed:
        logger.error(f"{len(failed)} migration başarısız oldu!")
        sys.exit(1)
    else:
        if not dry_run:
            logger.info("Tüm migration'lar başarıyla tamamlandı.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML DB Migration")
    parser.add_argument("--dry-run", action="store_true",
                        help="Sadece hangi migration'ların çalışacağını göster, uygulamaz")
    parser.add_argument("--from", dest="from_number", type=int, default=1,
                        help="Bu numara ve sonrasından başla (default: 1)")
    args = parser.parse_args()

    migrate(from_number=args.from_number, dry_run=args.dry_run)
