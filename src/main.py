import logging
import uuid
import sys
import os

# Add src to python path for easier imports if running from root
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository
from src.retrieval.postgres_candidate_retriever import PostgresCandidateRetriever
from src.reranker.reranker import Reranker
from src.scoring.final_scorer import FinalScorer
from src.etl.batch_processor import BatchProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting AML Entity Matching Pipeline...")
    
    # 1. Load Configuration
    config_loader = ConfigLoader()
    db_config = config_loader.get_db_config()
    retrieval_config = config_loader.get_retrieval_config()
    reranker_config = config_loader.get_reranker_config()
    scoring_config = config_loader.get_scoring_config()

    # 2. Initialize Repository
    logger.info("Initializing Repository...")
    repo = AMLRepository(
        host=db_config.get("host"),
        port=db_config.get("port"),
        dbname=db_config.get("name"),
        user=db_config.get("user"),
        password=db_config.get("password")
    )
    
    # Verify DB connection
    conn = repo.get_connection()
    if not conn:
        logger.error("Failed to connect to the database. Exiting.")
        sys.exit(1)
    conn.close()

    # 3. Initialize Pipeline Components
    logger.info("Initializing Pipeline Components...")
    retriever = PostgresCandidateRetriever(repo, retrieval_config)
    reranker = Reranker(repo, reranker_config)
    
    scorer = FinalScorer(
        repo, 
        config_version=scoring_config.get("scoring_config_version", "scoring_v2_reranker"),
        threshold_version=scoring_config.get("threshold_config_version", "threshold_v2_reranker")
    )
    
    processor = BatchProcessor(
        repository=repo,
        config=config_loader.config,
        retriever=retriever,
        reranker=reranker,
        scorer=scorer
    )

    # 4. Orchestrate the Run
    run_id   = f"RUN-{uuid.uuid4().hex[:8].upper()}"
    batch_id = f"BATCH-{uuid.uuid4().hex[:8].upper()}"
    
    logger.info(f"Generated RUN_ID: {run_id}")
    
    # 5. Process Table
    # Reads from bronze_eft_raw in chunks and inserts alerts
    processor.process_db_table_in_chunks(
        run_id=run_id, 
        batch_id=batch_id, 
        table_name="bronze_eft_raw",
        chunk_size=10000
    )
    
    logger.info(f"Pipeline completed for RUN_ID: {run_id}. Check aml_run_log and aml_alert tables for results.")

if __name__ == "__main__":
    main()