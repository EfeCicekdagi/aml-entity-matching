import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository

def load_data():
    csv_path = "data/aml_eft_challenge_dataset_1100.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]

    if 'explanation' not in df.columns:
        if 'description' in df.columns:
            df.rename(columns={'description': 'explanation'}, inplace=True)
        elif 'text' in df.columns:
            df.rename(columns={'text': 'explanation'}, inplace=True)

    if 'eft_id' not in df.columns:
        df['eft_id'] = range(1, len(df) + 1)
    if 'transaction_date' not in df.columns:
        df['transaction_date'] = pd.Timestamp.now()
    if 'amount' not in df.columns:
        df['amount'] = 1000.0
    if 'sender_account_id' not in df.columns:
        df['sender_account_id'] = 'TEST_SENDER'
    if 'receiver_account_id' not in df.columns:
        df['receiver_account_id'] = 'TEST_RECEIVER'
    if 'source_system' not in df.columns:
        df['source_system'] = 'TEST_CSV'
    if 'batch_id' not in df.columns:
        df['batch_id'] = 'TEST_BATCH_1'

    required_cols = ['eft_id', 'transaction_date', 'amount', 'explanation', 'sender_account_id', 'receiver_account_id', 'batch_id', 'source_system']
    df = df[required_cols]

    # Convert to matching types
    df['transaction_date'] = df['transaction_date'].astype(str)
    df['eft_id'] = df['eft_id'].astype(int)
    df['amount'] = df['amount'].astype(float)
    df['explanation'] = df['explanation'].astype(str)

    config_loader = ConfigLoader()
    db_cfg = config_loader.get_db_config()
    repo = AMLRepository(
        host=db_cfg["host"], port=db_cfg["port"],
        dbname=db_cfg["name"], user=db_cfg["user"], password=db_cfg["password"]
    )
    conn = repo.get_connection()

    try:
        with conn.cursor() as cur:
            # Ensure the table exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bronze_eft_raw (
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
            
            # Truncate before insert
            cur.execute("TRUNCATE TABLE bronze_eft_raw;")
            
            records = [tuple(x) for x in df.to_numpy()]
            
            query = """
                INSERT INTO bronze_eft_raw (
                    eft_id, transaction_date, amount, explanation, 
                    sender_account_id, receiver_account_id, batch_id, source_system
                ) VALUES %s
            """
            execute_values(cur, query, records)
            conn.commit()
            print(f"Loaded {len(records)} records into bronze_eft_raw successfully.")
    except Exception as e:
        print(f"Database error: {e}")
        conn.rollback()
    finally:
        repo.release_connection(conn)

if __name__ == "__main__":
    load_data()
