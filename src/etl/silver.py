import logging
import pandas as pd
import sys
from pathlib import Path

# Add src to path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from text_utils import normalize_text

logger = logging.getLogger(__name__)

def process_to_silver(raw_data: dict) -> dict:
    """
    Silver Layer: Cleanses and standardizes the raw Bronze data.
    Applies text cleaning, alias expansion, and format standardizations.
    """
    logger.info("--- [SILVER LAYER] Starting Data Cleansing ---")
    
    bank_df = raw_data.get("bank_data_raw", pd.DataFrame())
    sanction_df = raw_data.get("sanction_data_raw", pd.DataFrame())
    
    # Cleaning EFT descriptions
    if not bank_df.empty and 'description' in bank_df.columns:
        bank_df['description_clean'] = bank_df['description'].apply(lambda x: normalize_text(x) if pd.notnull(x) else x)
        
    # Cleaning company/sanction names
    if not sanction_df.empty and 'company_name' in sanction_df.columns:
        sanction_df['company_name_clean'] = sanction_df['company_name'].apply(lambda x: normalize_text(x) if pd.notnull(x) else x)
        
    # TODO: Add alias expansion and other complex logic here later.
    
    logger.info("Data cleansing completed. Silver data ready.")
    return {
        "bank_data_silver": bank_df,
        "sanction_data_silver": sanction_df
    }
