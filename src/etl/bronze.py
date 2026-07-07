import logging
import pandas as pd
import sys
from pathlib import Path

# Add src to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import EFT_FILE_PATH, COMPANY_FILE_PATH

logger = logging.getLogger(__name__)

def ingest_raw_data() -> dict:
    """
    Bronze Layer: Ingests raw data from external sources.
    Currently reads from the CSV files defined in config.py.
    """
    logger.info("--- [BRONZE LAYER] Starting Raw Data Ingestion ---")
    
    try:
        # Read the actual CSV files instead of mock data
        bank_data = pd.read_csv(EFT_FILE_PATH)
        sanction_data = pd.read_csv(COMPANY_FILE_PATH)
        
        logger.info(f"Ingested {len(bank_data)} bank/EFT records and {len(sanction_data)} company/sanction records.")
    except Exception as e:
        logger.error(f"Error reading raw data: {e}")
        bank_data = pd.DataFrame()
        sanction_data = pd.DataFrame()
        
    return {
        "bank_data_raw": bank_data,
        "sanction_data_raw": sanction_data
    }
