import sys
import os

# Add src to python path for easier imports if running from root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository

def main():
    config_loader = ConfigLoader()
    db_config = config_loader.get_db_config()
    repo = AMLRepository(
        host=db_config.get("host"),
        port=db_config.get("port"),
        dbname=db_config.get("name"),
        user=db_config.get("user"),
        password=db_config.get("password")
    )
    conn = repo.get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE aml_config.scoring_weight SET fuzzy_weight=0.20, vector_weight=0.20, reranker_weight=0.60 WHERE config_version='scoring_v2_reranker'")

    conn.commit()
    cur.close()
    repo.release_connection(conn)
    print("Database updated.")

if __name__ == "__main__":
    main()
