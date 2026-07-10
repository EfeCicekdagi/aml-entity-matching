import sys
import os
import pandas as pd
import psycopg2.extras
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.repository.aml_repository import AMLRepository
from src.utils.config_loader import ConfigLoader

def load_excel_to_db():
    print("Loading excel file into memory (data/bank.xlsx)...")
    df = pd.read_excel('data/bank.xlsx')
    
    print(f"Loaded {len(df)} rows. Transforming data...")
    # Clean Account No
    df['Account No'] = df['Account No'].astype(str).str.replace("'", "")
    
    # Calculate amount: Withdrawal if > 0, else Deposit
    df['WITHDRAWAL AMT'] = pd.to_numeric(df['WITHDRAWAL AMT'], errors='coerce').fillna(0)
    df['DEPOSIT AMT'] = pd.to_numeric(df['DEPOSIT AMT'], errors='coerce').fillna(0)
    df['amount'] = df[['WITHDRAWAL AMT', 'DEPOSIT AMT']].max(axis=1)
    
    # Fill explanation NaNs
    df['TRANSACTION DETAILS'] = df['TRANSACTION DETAILS'].fillna("").astype(str)
    
    c = ConfigLoader().get_db_config()
    repo = AMLRepository(c['host'], c['port'], c['name'], c['user'], c['password'])
    conn = repo.get_connection()
    
    print("Emptying old records from bronze_eft_raw...")
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE bronze_eft_raw;")
        # Also clean up old run data just to start fresh
        cur.execute("TRUNCATE TABLE aml_alert;")
        cur.execute("TRUNCATE TABLE aml_candidate_match;")
        cur.execute("TRUNCATE TABLE aml_run_log;")
        conn.commit()

    print("Inserting data into PostgreSQL...")
    batch_size = 10000
    total_inserted = 0
    
    insert_query = """
        INSERT INTO bronze_eft_raw 
        (eft_id, transaction_date, amount, sender_account_id, explanation, source_system, batch_id)
        VALUES %s
    """
    
    dates = df['DATE'].tolist()
    amounts = df['amount'].tolist()
    accounts = df['Account No'].tolist()
    details = df['TRANSACTION DETAILS'].tolist()
    
    with conn.cursor() as cur:
        rows = []
        for i in range(len(df)):
            try:
                t_date = dates[i].date() if pd.notnull(dates[i]) else datetime.now().date()
            except:
                t_date = datetime.now().date()
                
            rows.append((
                i + 1,
                t_date,
                float(amounts[i]),
                str(accounts[i]),
                str(details[i]),
                "bank.xlsx",
                "BATCH-INITIAL"
            ))
            
            if len(rows) == batch_size:
                psycopg2.extras.execute_values(cur, insert_query, rows)
                conn.commit()
                total_inserted += len(rows)
                print(f"Inserted {total_inserted}/{len(df)} rows...")
                rows = []
                
        if rows:
            psycopg2.extras.execute_values(cur, insert_query, rows)
            conn.commit()
            total_inserted += len(rows)
            print(f"Inserted {total_inserted}/{len(df)} rows...")
            
    print("Database population completed successfully!")
    repo.release_connection(conn)

if __name__ == '__main__':
    load_excel_to_db()
