import pandas as pd
from src.ui.dashboard import get_repo

def run():
    repo = get_repo()
    conn = repo.get_connection()
    if not conn:
        return
        
    try:
        query = "SELECT * FROM aml_stage.company_variant LIMIT 1"
        df = pd.read_sql(query, conn)
        print("company_variant schema:")
        print(df.columns)
        
    finally:
        repo.release_connection(conn)

if __name__ == "__main__":
    run()
