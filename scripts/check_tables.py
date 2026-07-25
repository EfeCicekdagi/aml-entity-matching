import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config.config_loader import ConfigLoader
from src.config.db_tables import TABLES
from src.repository.aml_repository import AMLRepository

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
    cur = conn.cursor()

    cur.execute("""
        SELECT table_schema, table_name FROM information_schema.tables 
        WHERE table_schema IN ('aml_source', 'aml_stage', 'aml_ml', 'aml_core', 'aml_config', 'aml_audit', 'aml_eval', 'aml_experiment', 'public') 
        ORDER BY table_schema, table_name
    """)
    tables = cur.fetchall()
    print('=== TABLOLAR ===')
    for schema, t in tables:
        full_table = f"{schema}.{t}" if schema != 'public' else t
        try:
            cur.execute(f'SELECT COUNT(*) FROM {full_table}')
            count = cur.fetchone()[0]
            print(f'  {full_table}: {count} satir')
        except Exception as e:
            print(f'  {full_table}: HATA - {e}')
            conn.rollback()

    # Son run'un alertleri
    print(f'\n=== SON RUN ALERTLERI ({TABLES["alert"]}) ===')
    try:
        cur.execute(f"""
            SELECT risk_level, COUNT(*) 
            FROM {TABLES['alert']} 
            GROUP BY risk_level
            ORDER BY COUNT(*) DESC
        """)
        for row in cur.fetchall():
            print(f'  {row[0]}: {row[1]}')
    except Exception as e:
        print(f"  HATA: {e}")
        conn.rollback()

    print(f'\n=== RUN LOGLARI (Son 5 Run - {TABLES["run_log"]}) ===')
    try:
        cur.execute(f"""
            SELECT 
                run_id, 
                status, 
                total_duration_s,
                input_count, 
                prescreen_skipped_count,
                alert_count, 
                embedding_model_name,
                reranker_model_name,
                started_at 
            FROM {TABLES['run_log']} 
            ORDER BY started_at DESC 
            LIMIT 5
        """)
        for row in cur.fetchall():
            r_id, stat, dur, inp, skip, alerts, emb, rer, start = row
            dur_str = f"{dur}s" if dur is not None else "?"
            skip_str = f"{skip}" if skip is not None else "?"
            
            print(f"  {r_id} | {stat} | {dur_str} | In:{inp} | Skip:{skip_str} | Alerts:{alerts}")
            if emb or rer:
                print(f"    -> Emb: {emb} | Rerank: {rer}")
    except Exception as e:
        print(f"  HATA: {e}")
        conn.rollback()

    cur.close()
    repo.release_connection(conn)

if __name__ == "__main__":
    check_tables()
