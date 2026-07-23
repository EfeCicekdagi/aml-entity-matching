import logging
import uuid
import sys
import os
import argparse

# Add src to python path for easier imports if running from root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.app.container import ApplicationContainer
from src.pipeline.batch_processor import BatchProcessor
from src.config.db_tables import TABLES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run AML Entity Matching Pipeline")
    parser.add_argument("--input-table", type=str, default="eft_input",
                        help="The input table logical name (e.g., eft_input). Mapped via TABLES.")
    args = parser.parse_args()

    logger.info("Starting AML Entity Matching Pipeline...")
    
    # Setup PII Masking if enabled (will be implemented in masking.py)
    # from src.utils.masking import setup_secure_logging
    # setup_secure_logging()
    
    container = ApplicationContainer()
    
    if container.config.get("security", {}).get("enable_pii_masking_in_logs", False):
        try:
            from src.utils.masking import setup_secure_logging
            setup_secure_logging()
        except ImportError:
            logger.warning("Masking module not found. PII masking is skipped.")

    logger.info("Initializing Pipeline Components...")
    container.init_resources()
    
    # Verify DB connection
    conn = container.repo.get_connection()
    if not conn:
        logger.error("Failed to connect to the database. Exiting.")
        sys.exit(1)
    container.repo.release_connection(conn)

    processor = BatchProcessor(
        repository=container.repo,
        config=container.config,
        inference_service=container.inference_service
    )

    run_id   = f"RUN-{uuid.uuid4().hex[:8].upper()}"
    batch_id = f"BATCH-{uuid.uuid4().hex[:8].upper()}"
    
    logger.info(f"Generated RUN_ID: {run_id}")
    table_name = TABLES.get(args.input_table, args.input_table)
    logger.info(f"Using Input Table: {table_name}")
    
    try:
        processor.process_db_table_in_chunks(
            run_id=run_id, 
            batch_id=batch_id, 
            table_name=table_name
        )
        logger.info(f"Pipeline completed successfully for RUN_ID: {run_id}.")
    except Exception as e:
        logger.error(f"Pipeline FAILED for RUN_ID: {run_id}. Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
