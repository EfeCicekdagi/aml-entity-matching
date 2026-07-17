import sys
import os
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository
from src.config.db_tables import TABLES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def clear_results():
    logger.info("Eski sonuçlar ve önbellek (cache) temizleniyor...")
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
    if not conn:
        logger.error("Veritabanı bağlantısı kurulamadı.")
        return

    tables_to_clear = [
        TABLES['alert'],
        TABLES['run_log'],
        TABLES['reranker_cache']
    ]

    try:
        with conn.cursor() as cur:
            for table in tables_to_clear:
                logger.info(f"{table} tablosu temizleniyor...")
                cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;")
        conn.commit()
        logger.info("✅ Tüm eski sonuçlar ve cache başarıyla temizlendi! Artık temiz bir şekilde çalıştırabilirsiniz.")
    except Exception as e:
        logger.error(f"Tablolar temizlenirken hata oluştu: {e}")
        conn.rollback()
    finally:
        repo.release_connection(conn)

if __name__ == "__main__":
    clear_results()
