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
import hashlib
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SQL_DIR = os.path.join(PROJECT_ROOT, "sql")

sys.path.append(PROJECT_ROOT)
from src.config.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository
from src.config.db_tables import TABLES

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


def validate_migration_list(migrations: list[str]) -> list[str]:
    """
    Migration listesini başlangıçta doğrular.
    Hata varsa ValueError fırlatır.
    """
    nums = []
    seen_files = set()
    seen_nums = set()

    for fname in migrations:
        if not fname.endswith(".sql"):
            raise ValueError(f"Migration dosyası '.sql' uzantılı olmalıdır: {fname}")

        parts = fname.split("_")
        if not (parts[0].isdigit() and len(parts[0]) >= 2):
            raise ValueError(f"Dosya adı iki haneli veya sayısal prefix ile başlamalıdır: {fname}")

        if fname in seen_files:
            raise ValueError(f"Aynı migration dosyası listelendiğinde tekrar edemez: {fname}")
        seen_files.add(fname)

        num = int(parts[0])
        if num in seen_nums:
            raise ValueError(f"Migration numarası tekrar ediyor: {num:02d} ({fname})")
        seen_nums.add(num)
        nums.append(num)

    for i in range(1, len(nums)):
        if nums[i] <= nums[i-1]:
            raise ValueError(f"Migration listesi artan sırada değil: {nums[i-1]:02d} sonrasında {nums[i]:02d} geliyor.")
        if nums[i] - nums[i-1] > 1:
            logger.warning(f"UYARI: Migration numarası boşluğu bulundu: {nums[i-1]:02d} -> {nums[i]:02d}")

    return migrations


def calculate_sha256(file_path: str) -> str:
    """Dosyanın SHA-256 özetini hesaplar."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def bootstrap_history_table(repo: AMLRepository) -> None:
    """schema_migration history tablosunu güvenli şekilde oluşturur."""
    conn = repo.get_connection()
    if not conn:
        raise RuntimeError("Veritabanı bağlantısı kurulamadı (History tablosu bootstrap hatası).")
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLES['schema_migration']} (
                    migration_name TEXT PRIMARY KEY,
                    migration_number INTEGER NOT NULL,
                    checksum_sha256 TEXT NOT NULL,
                    applied_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    execution_time_ms NUMERIC,
                    status TEXT NOT NULL,
                    error_message TEXT
                );
            """)
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"History tablosu oluşturulamadı: {e}", exc_info=True)
        raise
    finally:
        repo.release_connection(conn)


def get_applied_migrations(repo: AMLRepository) -> dict[str, dict]:
    """Daha önce başarıyla uygulanmış migration'ları döndürür."""
    conn = repo.get_connection()
    if not conn:
        return {}
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT migration_name, checksum_sha256, status
                FROM {TABLES['schema_migration']}
                WHERE status = 'SUCCESS'
            """)
            rows = cur.fetchall()
            return {row[0]: {"checksum_sha256": row[1], "status": row[2]} for row in rows}
    except Exception:
        conn.rollback()
        return {}
    finally:
        repo.release_connection(conn)


def execute_migration_with_history(
    repo: AMLRepository,
    script_path: str,
    migration_name: str,
    migration_number: int,
    checksum: str
) -> float:
    """Tek bir migration dosyasını ve history kaydını aynı transaction içinde atomik çalıştırır."""
    conn = repo.get_connection()
    if not conn:
        raise RuntimeError("Veritabanı bağlantısı kurulamadı.")
    start = time.perf_counter()
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            sql = f.read()
        with conn.cursor() as cur:
            cur.execute(sql)
            elapsed_ms = (time.perf_counter() - start) * 1000
            cur.execute(f"""
                INSERT INTO {TABLES['schema_migration']} (
                    migration_name, migration_number, checksum_sha256,
                    applied_at, execution_time_ms, status, error_message
                )
                VALUES (%s, %s, %s, NOW(), %s, 'SUCCESS', NULL)
                ON CONFLICT (migration_name) DO UPDATE SET
                    checksum_sha256 = EXCLUDED.checksum_sha256,
                    applied_at = EXCLUDED.applied_at,
                    execution_time_ms = EXCLUDED.execution_time_ms,
                    status = EXCLUDED.status,
                    error_message = EXCLUDED.error_message;
            """, (migration_name, migration_number, checksum, round(elapsed_ms, 2)))
        conn.commit()
        return elapsed_ms
    except Exception:
        conn.rollback()
        raise
    finally:
        repo.release_connection(conn)


def record_failed_migration(
    repo: AMLRepository,
    migration_name: str,
    migration_number: int,
    checksum: str,
    elapsed_ms: float,
    error_text: str
) -> None:
    """Başarısız migration'ın hata durumunu ayrı bir işlemle history tablosuna kaydeder."""
    conn = repo.get_connection()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {TABLES['schema_migration']} (
                    migration_name, migration_number, checksum_sha256,
                    applied_at, execution_time_ms, status, error_message
                )
                VALUES (%s, %s, %s, NOW(), %s, 'FAILED', %s)
                ON CONFLICT (migration_name) DO UPDATE SET
                    checksum_sha256 = EXCLUDED.checksum_sha256,
                    applied_at = EXCLUDED.applied_at,
                    execution_time_ms = EXCLUDED.execution_time_ms,
                    status = EXCLUDED.status,
                    error_message = EXCLUDED.error_message;
            """, (migration_name, migration_number, checksum, round(elapsed_ms, 2), error_text[:1000]))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Başarısız migration kaydı history tablosuna yazılamadı: {e}")
    finally:
        repo.release_connection(conn)


def migrate(from_number: int = 1, dry_run: bool = False):
    try:
        validate_migration_list(MIGRATIONS)
    except ValueError as e:
        logger.error(f"Migration listesi doğrulama hatası: {e}")
        sys.exit(1)

    config_loader = ConfigLoader()
    db_config = config_loader.get_db_config()

    repo = AMLRepository(
        host=db_config.get("host"),
        port=db_config.get("port"),
        dbname=db_config.get("name"),
        user=db_config.get("user"),
        password=db_config.get("password")
    )

    if not dry_run:
        try:
            bootstrap_history_table(repo)
        except Exception as e:
            logger.error(f"History tablosu hazırlanamadı: {e}")
            sys.exit(1)

    applied_migrations = get_applied_migrations(repo)

    # --from kullanımı öncesindeki dosyaların durumunu kontrol et
    if from_number > 1:
        prior_migrations = [f for f in MIGRATIONS if int(f.split("_")[0]) < from_number]
        missing_priors = [f for f in prior_migrations if f not in applied_migrations]
        if missing_priors:
            logger.warning(f"UYARI: {from_number} numaralı migration’dan başlanıyor ancak önceki migration’ların tamamı uygulanmış görünmüyor.")

    # from_number filtresine göre migration'ları seç
    selected = [f for f in MIGRATIONS if int(f.split("_")[0]) >= from_number]

    if not selected:
        logger.info("Çalıştırılacak migration yok.")
        return

    print("\n" + "="*60)
    print(f"  AML Migration {'(DRY RUN)' if dry_run else 'Başlatılıyor'}")
    print(f"  Toplam {len(selected)} migration çalıştırılacak")
    print("="*60)

    if dry_run:
        print("\n--- DRY-RUN DOSYA İNCELEMESİ ---")
        for fname in MIGRATIONS:
            num_str = fname.split("_")[0]
            fpath = os.path.join(SQL_DIR, fname)
            exists = os.path.exists(fpath)
            chk = calculate_sha256(fpath) if exists else "N/A"
            is_selected = fname in selected
            print(f"  Numara: {num_str} | Dosya: {fname} | Mevcut: {'Evet' if exists else 'HAYIR'} | Checksum: {chk[:8]}... | Seçildi: {'Evet' if is_selected else 'Hayır'}")
        print("--------------------------------\n")

    results = []
    for i, migration_file in enumerate(selected, 1):
        script_path = os.path.join(SQL_DIR, migration_file)
        num = int(migration_file.split("_")[0])

        if not os.path.exists(script_path):
            logger.error(f"  [{i}/{len(selected)}] [FAIL] EKSİK DOSYA: {migration_file}")
            results.append((migration_file, "FAILED_MISSING_FILE"))
            if not dry_run:
                for rem in selected[i:]:
                    results.append((rem, "NOT_RUN"))
                break
            continue

        checksum = calculate_sha256(script_path)

        if migration_file in applied_migrations:
            db_checksum = applied_migrations[migration_file]["checksum_sha256"]
            if db_checksum != checksum:
                logger.error(f"  [{i}/{len(selected)}] [FAIL] CHECKSUM MISMATCH: {migration_file} (Dosya sonradan değiştirilmiş!)")
                results.append((migration_file, "CHECKSUM_MISMATCH"))
                if not dry_run:
                    for rem in selected[i:]:
                        results.append((rem, "NOT_RUN"))
                    break
                continue
            else:
                print(f"  [{i}/{len(selected)}] [SKIP] ALREADY_APPLIED: {migration_file}")
                results.append((migration_file, "ALREADY_APPLIED"))
                continue

        if dry_run:
            print(f"  [{i}/{len(selected)}] [DRY-RUN] {migration_file}")
            results.append((migration_file, "DRY_RUN"))
            continue

        start_time = time.perf_counter()
        try:
            print(f"  [{i}/{len(selected)}] [...] Calistiriliyor: {migration_file}")
            elapsed_ms = execute_migration_with_history(repo, script_path, migration_file, num, checksum)
            print(f"  [{i}/{len(selected)}] [OK] {migration_file} — {elapsed_ms:.1f} ms")
            results.append((migration_file, "SUCCESS"))
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            error_text = str(e)[:1000]
            logger.error(f"Error executing migration {migration_file}: {e}", exc_info=True)
            record_failed_migration(repo, migration_file, num, checksum, elapsed_ms, error_text)
            print(f"  [{i}/{len(selected)}] [FAIL] HATA: {migration_file} — {elapsed_ms:.1f} ms")
            print(f"        {error_text}")
            results.append((migration_file, f"FAILED: {error_text}"))
            for rem in selected[i:]:
                results.append((rem, "NOT_RUN"))
            break

    print("\n" + "="*60)
    print("  OZET:")
    for fname, status in results:
        if status == "SUCCESS":
            icon = "[OK]"
        elif status == "ALREADY_APPLIED":
            icon = "[SKIP]"
        elif status == "DRY_RUN":
            icon = "[DRY]"
        elif status == "NOT_RUN":
            icon = "[---]"
        else:
            icon = "[FAIL]"
        print(f"  {icon} {fname} -> {status}")
    print("="*60 + "\n")

    failed = [r for r in results if r[1] in ("FAILED_MISSING_FILE", "CHECKSUM_MISMATCH") or r[1].startswith("FAILED")]
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

