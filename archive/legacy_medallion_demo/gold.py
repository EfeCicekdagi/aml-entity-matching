import logging
import pandas as pd
import sys
from pathlib import Path

# Add src to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

def run_entity_matching(silver_data: dict) -> pd.DataFrame:
    """
    Gold Layer: Runs the AI/ML matching algorithms on cleansed data.
    Outputs business-ready matched entities.
    """
    logger.info("--- [GOLD LAYER] Starting Entity Matching ---")
    
    bank_df = silver_data.get("bank_data_silver", pd.DataFrame())
    sanction_df = silver_data.get("sanction_data_silver", pd.DataFrame())
    
    # TODO: Integrate vector_utils, matcher, and candidate_filter here.
    # For now, simulate a match output.
    
    matches = []
    if not bank_df.empty and not sanction_df.empty:
        # Dummy matching logic for skeleton
        # Handle cases where eft_id or company_id might be missing
        b_id = bank_df.iloc[0].get("eft_id", "UNKNOWN_EFT")
        s_id = sanction_df.iloc[0].get("company_id", "UNKNOWN_COMPANY")
        
        matches.append({
            "bank_entity_id": b_id,
            "sanction_entity_id": s_id,
            "match_score": 0.95,
            "decision": "REVIEW"
        })
        
    results_df = pd.DataFrame(matches)
    logger.info(f"Generated {len(results_df)} potential matches.")
    
    return results_df
