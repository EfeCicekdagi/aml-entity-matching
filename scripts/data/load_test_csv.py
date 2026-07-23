import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.config.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run(csv_path: str):
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        sys.exit(1)

    logger.info(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Clean and prepare columns. Assuming standard columns exist in CSV.
    # If any are missing, fill them with defaults.
    required_cols = ['eft_id', 'transaction_date', 'amount', 'sender_account_id', 'receiver_account_id', 'explanation', 'source_system', 'batch_id']
    
    # Auto-mapping logic
    # Lowercase column names for easier matching
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Check if we have the minimal required 'explanation'
    if 'explanation' not in df.columns:
        if 'description' in df.columns:
            df.rename(columns={'description': 'explanation'}, inplace=True)
        elif 'text' in df.columns:
            df.rename(columns={'text': 'explanation'}, inplace=True)
        else:
            logger.error("Could not find 'explanation' column in CSV!")
            sys.exit(1)

    # Fill defaults for missing columns
    if 'eft_id' not in df.columns:
        df['eft_id'] = range(1, len(df) + 1)
    if 'transaction_date' not in df.columns:
        df['transaction_date'] = datetime.now()
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

    # Ensure only required columns are kept
    df = df[required_cols]

    # Connect to DB
    config_loader = ConfigLoader()
    db_cfg = config_loader.get_db_config()
    repo = AMLRepository(
        host=db_cfg["host"], port=db_cfg["port"],
        dbname=db_cfg["name"], user=db_cfg["user"], password=db_cfg["password"]
    )
    conn = repo.get_connection()

    logger.info("Creating test table aml_source.test_eft_input...")
    with conn.cursor() as cur:
        # Drop if exists
        cur.execute("DROP TABLE IF EXISTS aml_source.test_eft_input CASCADE;")
        
        # Create table matching bronze_eft_raw
        cur.execute("""
            CREATE TABLE aml_source.test_eft_input (
                eft_id BIGINT PRIMARY KEY,
                transaction_date TIMESTAMP,
                amount NUMERIC(15,2),
                sender_account_id TEXT,
                receiver_account_id TEXT,
                explanation TEXT,
                source_system TEXT,
                batch_id TEXT
            );
        """)
        
        # Insert data
        records = [tuple(x) for x in df.to_numpy()]
        logger.info(f"Inserting {len(records)} records...")
        
        execute_values(
            cur,
            "INSERT INTO aml_source.test_eft_input (eft_id, transaction_date, amount, sender_account_id, receiver_account_id, explanation, source_system, batch_id) VALUES %s",
            records
        )
        conn.commit()

    repo.release_connection(conn)
    logger.info("Test data successfully loaded to aml_source.test_eft_input!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_test_csv.py <path_to_csv>")
        sys.exit(1)
    
    run(sys.argv[1])
