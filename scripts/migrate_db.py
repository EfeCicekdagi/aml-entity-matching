import sys
import os
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def migrate():
    config_loader = ConfigLoader()
    db_config = config_loader.get_db_config()

    repo = AMLRepository(
        host=db_config.get("host"),
        port=db_config.get("port"),
        dbname=db_config.get("name"),
        user=db_config.get("user"),
        password=db_config.get("password")
    )
    
    script_path = os.path.join(os.path.dirname(__file__), '..', 'sql', '12_update_alert_structure.sql')
    repo.execute_script(script_path)
    logger.info("Migration completed.")

if __name__ == "__main__":
    migrate()
