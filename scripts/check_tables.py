import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config.config_loader import ConfigLoader
from src.config.db_tables import TABLES
from src.repository.aml_repository import AMLRepository


def check_column_exists(cur, full_table_name, col_name):
    if "." in full_table_name:
        schema, table = full_table_name.split(".", 1)
    else:
        schema, table = "public", full_table_name
    cur.execute("""
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = %s AND table_name = %s AND column_name = %s
    """, (schema, table, col_name))
    return cur.fetchone() is not None


def get_table_columns(cur, full_table_name):
    if "." in full_table_name:
        schema, table = full_table_name.split(".", 1)
    else:
        schema, table = "public", full_table_name
    cur.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_schema = %s AND table_name = %s
    """, (schema, table))
    return {row[0] for row in cur.fetchall()}


def check_tables():
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

    cur = None
    try:
        cur = conn.cursor()

        # 1. TABLO VE VIEW LİSTESİ
        print('=== TABLO VE VIEW LİSTESİ ===')
        cur.execute("""
            SELECT table_schema, table_name, table_type 
            FROM information_schema.tables 
            WHERE table_schema IN ('aml_source', 'aml_stage', 'aml_ml', 'aml_core', 'aml_config', 'aml_audit', 'aml_eval', 'aml_experiment', 'public') 
            ORDER BY table_schema, table_name
        """)
        tables = cur.fetchall()
        
        found_tables_set = set()
        table_status_map = {}
        success_count = 0
        error_count = 0

        for schema, t_name, t_type_raw in tables:
            full_table = f"{schema}.{t_name}" if schema != 'public' else t_name
            full_table_sql = f'"{schema}"."{t_name}"'
            found_tables_set.add(f"{schema}.{t_name}")
            
            t_type = "VIEW" if "VIEW" in str(t_type_raw).upper() else "BASE TABLE"
            type_label = t_type

            cur.execute("SAVEPOINT table_count_check")
            try:
                cur.execute(f"SELECT COUNT(*) FROM {full_table_sql}")
                count = cur.fetchone()[0]
                cur.execute("RELEASE SAVEPOINT table_count_check")
                print(f"  {full_table} [{type_label}]: {count:,} satır")
                table_status_map[f"{schema}.{t_name}"] = f"OK ({count:,})" if count > 0 else "EMPTY"
                success_count += 1
            except Exception as e:
                try:
                    cur.execute("ROLLBACK TO SAVEPOINT table_count_check")
                    cur.execute("RELEASE SAVEPOINT table_count_check")
                except Exception:
                    conn.rollback()
                print(f"  {full_table} [{type_label}]: HATA - {e}")
                table_status_map[f"{schema}.{t_name}"] = f"HATA - {e}"
                error_count += 1

        # 2. KRİTİK TABLO DURUMU
        print('\n=== KRİTİK TABLO DURUMU ===')
        critical_tables = [
            TABLES["eft_input"],
            TABLES["company_variant"],
            TABLES["company_embedding"],
            TABLES["match_result"],
            TABLES["alert"],
            TABLES["run_log"],
        ]
        for crit_tab in critical_tables:
            status_str = table_status_map.get(crit_tab, "MISSING")
            print(f"  {crit_tab}: {status_str}")

        # 3. SON RUN ALERT DAĞILIMI
        print(f'\n=== SON RUN ALERT DAĞILIMI ({TABLES["alert"]}) ===')
        if TABLES["alert"] not in found_tables_set:
            print("  Alert tablosu bulunamadı.")
        else:
            try:
                has_run_id = check_column_exists(cur, TABLES["alert"], "run_id")
                if not has_run_id:
                    print("  Alert tablosunda run_id kolonu bulunamadığı için son run filtresi uygulanamadı.")
                    print("  Tüm alert dağılımı gösteriliyor.")
                    cur.execute(f"SELECT risk_level, COUNT(*) FROM {TABLES['alert']} GROUP BY risk_level ORDER BY COUNT(*) DESC")
                    for row in cur.fetchall():
                        print(f"  {row[0]}: {row[1]:,}")
                else:
                    cur.execute(f"SELECT run_id FROM {TABLES['run_log']} ORDER BY started_at DESC LIMIT 1")
                    res_run = cur.fetchone()
                    if not res_run:
                        print("  Henüz çalıştırılmış pipeline run kaydı bulunamadı.")
                    else:
                        latest_run_id = res_run[0]
                        print(f"  Son Run ID: {latest_run_id}")
                        cur.execute(f"SELECT risk_level, COUNT(*) FROM {TABLES['alert']} WHERE run_id = %s GROUP BY risk_level ORDER BY COUNT(*) DESC", (latest_run_id,))
                        rows = cur.fetchall()
                        if not rows:
                            print("  Bu run için alert bulunamadı.")
                        for row in rows:
                            print(f"  {row[0]}: {row[1]:,}")
            except Exception as e:
                print(f"  HATA: {e}")
                conn.rollback()

        # 4. SON 5 RUN LOGU
        print(f'\n=== SON 5 RUN LOGU ({TABLES["run_log"]}) ===')
        if TABLES["run_log"] not in found_tables_set:
            print("  Run log tablosu bulunamadı.")
        else:
            try:
                existing_cols = get_table_columns(cur, TABLES["run_log"])
                if not {"run_id", "status", "started_at"}.issubset(existing_cols):
                    print("  Run log tablosunda temel kolonlar (run_id, status, started_at) bulunmuyor.")
                else:
                    desired_cols = [
                        "run_id", "status", "started_at",
                        "total_duration_s", "input_count", "prescreen_skipped_count",
                        "alert_count", "embedding_model_name", "reranker_model_name",
                        "completed_at", "error_message"
                    ]
                    cols_to_query = [c for c in desired_cols if c in existing_cols]
                    cols_sql = ", ".join(f'"{c}"' for c in cols_to_query)
                    cur.execute(f"SELECT {cols_sql} FROM {TABLES['run_log']} ORDER BY started_at DESC LIMIT 5")
                    rows = cur.fetchall()
                    if not rows:
                        print("  Kayıt bulunamadı.")
                    for row in rows:
                        row_dict = dict(zip(cols_to_query, row))
                        r_id = row_dict.get("run_id", "N/A")
                        stat = row_dict.get("status", "N/A")
                        dur = row_dict.get("total_duration_s")
                        dur_str = f"{dur}s" if dur is not None else "N/A"
                        inp = row_dict.get("input_count", "N/A")
                        skip = row_dict.get("prescreen_skipped_count", "N/A")
                        alerts = row_dict.get("alert_count", "N/A")
                        emb = row_dict.get("embedding_model_name")
                        rer = row_dict.get("reranker_model_name")
                        err = row_dict.get("error_message")
                        
                        print(f"  {r_id} | {stat} | {dur_str} | In:{inp} | Skip:{skip} | Alerts:{alerts}")
                        if emb or rer:
                            print(f"    -> Emb: {emb or 'N/A'} | Rerank: {rer or 'N/A'}")
                        if str(stat).strip().upper() == "FAILED" and err:
                            err_str = str(err)[:200]
                            print(f"    -> Error: {err_str}")
            except Exception as e:
                print(f"  HATA: {e}")
                conn.rollback()

        # 5. GENEL KONTROL ÖZETİ
        print('\n=== GENEL KONTROL ÖZETİ ===')
        all_expected_tables = set(TABLES.values())
        missing_count = sum(1 for t in all_expected_tables if t not in found_tables_set)
        total_objects = success_count + error_count + missing_count
        
        print(f"Toplam nesne: {total_objects}")
        print(f"Başarılı sayılan: {success_count}")
        print(f"Eksik: {missing_count}")
        print(f"Hatalı: {error_count}")

    except Exception as e:
        print(f"Kontrol sırasında beklenmeyen hata: {e}")
    finally:
        if cur is not None:
            cur.close()
        repo.release_connection(conn)


if __name__ == "__main__":
    check_tables()

