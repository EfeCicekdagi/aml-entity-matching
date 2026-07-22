import psycopg2
from src.ui.dashboard import get_repo

def run():
    repo = get_repo()
    conn = repo.get_connection()
    if not conn:
        print("Could not connect to DB")
        return
        
    try:
        with conn.cursor() as cur:
            # Check if company table exists
            cur.execute("SELECT MAX(company_id) FROM aml_stage.company_variant")
            max_id = cur.fetchone()[0] or 0
            new_id = max_id + 1
            
            # Insert into variant table
            insert_query = """
                INSERT INTO aml_stage.company_variant 
                (company_id, original_company_name, variant_name, normalized_variant_name, variant_type, is_active, source_name)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cur.execute(insert_query, (
                new_id,
                'Berke Plastik Sanayi',
                'Berke Plastik Sanayi',
                'berke plastik sanayi',
                'ORIGINAL',
                True,
                'MANUAL_INSERT'
            ))
            conn.commit()
            print("Successfully inserted 'Berke Plastik Sanayi' into aml_stage.company_variant.")
            
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
    finally:
        repo.release_connection(conn)

if __name__ == "__main__":
    run()
