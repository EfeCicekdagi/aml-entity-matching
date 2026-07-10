from src.repository.aml_repository import AMLRepository
from src.utils.config_loader import ConfigLoader
import pandas as pd

def main():
    c = ConfigLoader().get_db_config()
    r = AMLRepository(c['host'], c['port'], c['name'], c['user'], c['password'])
    conn = r.get_connection()

    run_id = 'RUN-5F35F502'
    
    log = pd.read_sql(f"SELECT * FROM aml_run_log WHERE run_id = '{run_id}'", conn)
    print("--- RUN LOG ---")
    if not log.empty:
        for k, v in log.iloc[0].items():
            print(f"{k}: {v}")
    else:
        print("No run log found.")

    alerts = pd.read_sql(f"""
        SELECT a.eft_id, v.original_company_name, a.risk_level, a.extracted_entity 
        FROM aml_alert a 
        JOIN silver_company_variant v ON a.variant_id = v.variant_id 
        WHERE a.run_id = '{run_id}' AND a.extracted_entity IS NOT NULL 
        LIMIT 10
    """, conn)
    print("\n--- SAMPLE ALERTS (WITH NER) ---")
    print(alerts.to_string())

    alerts_stats = pd.read_sql(f"""
        SELECT risk_level, count(*) as count 
        FROM aml_alert 
        WHERE run_id = '{run_id}' 
        GROUP BY risk_level
    """, conn)
    print("\n--- ALERT STATS ---")
    print(alerts_stats.to_string())

    conn.close()

if __name__ == "__main__":
    main()
