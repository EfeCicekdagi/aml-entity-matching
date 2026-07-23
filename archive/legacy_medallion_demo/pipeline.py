import logging
from bronze import ingest_raw_data
from silver import process_to_silver
from gold import run_entity_matching

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_etl_pipeline():
    logger.info("Starting Medallion ETL Pipeline...")
    
    # 1. Bronze Layer (Ingestion)
    raw_data = ingest_raw_data()
    
    # 2. Silver Layer (Cleansing)
    cleansed_data = process_to_silver(raw_data)
    
    # 3. Gold Layer (Matching & Business Logic)
    final_matches = run_entity_matching(cleansed_data)
    
    logger.info("Pipeline completed successfully.")
    
    print("\n=== FINAL MATCH RESULTS (GOLD LAYER) ===")
    print(final_matches.to_markdown(index=False) if not final_matches.empty else "No matches found.")
    
if __name__ == "__main__":
    run_etl_pipeline()
