import sys
import os
import random
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.repository.aml_repository import AMLRepository
from src.utils.config_loader import ConfigLoader

def seed_database():
    c = ConfigLoader().get_db_config()
    repo = AMLRepository(c['host'], c['port'], c['name'], c['user'], c['password'])
    conn = repo.get_connection()

    explanations = [
        "TR TO Indiaforensic SERVICES IN",
        "TRF FRM ABC TRADING LTD STI INV 1002",
        "PAYMENT FOR North Star Trading LOGISTICS",
        "SALARY FOR JOHN DOE",
        "RENT PAYMENT FOR HQ",
        "INV 99912 GLOBAL ENTERPRISES LLC",
        "trf frm indiaforensic services",
        "SWIFT TO PACIFIC HOLDINGS CO",
        "INVOICE PAYMENT xyz solutions",
        "FEE FOR TECH INNOVATIONS INC"
    ]
    
    try:
        with conn.cursor() as cur:
            # Generate 50 dummy EFT records
            for i in range(1, 51):
                eft_id = 1000 + i
                amount = round(random.uniform(100.0, 10000.0), 2)
                exp = random.choice(explanations)
                
                cur.execute("""
                    INSERT INTO bronze_eft_raw 
                    (eft_id, transaction_date, amount, sender_account_id, receiver_account_id, explanation, source_system, batch_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (eft_id) DO NOTHING
                """, (
                    eft_id, 
                    datetime.now().date(), 
                    amount, 
                    f"ACC-SND-{i}", 
                    f"ACC-RCV-{i}", 
                    exp, 
                    "TEST_SYSTEM", 
                    "BATCH-TEST-1"
                ))
            
            conn.commit()
            print("Successfully inserted 50 test records into bronze_eft_raw.")
    except Exception as e:
        print(f"Failed to seed data: {e}")
    finally:
        repo.release_connection(conn)

if __name__ == '__main__':
    seed_database()
