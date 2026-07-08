import pandas as pd
import logging
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, repository, config, retriever, reranker, scorer):
        self.repo = repository
        self.config = config
        self.retriever = retriever
        self.reranker = reranker
        self.scorer = scorer
        
        self.batch_size = self.config.get("embedding", {}).get("batch_size", 128)
        self.embedding_model_name = self.config.get("embedding", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model = None

    def _load_embedding_model(self):
        if not self.embedding_model:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model loaded.")

    def process_file_in_chunks(self, file_path: str, run_id: str, batch_id: str, chunk_size: int = 10000):
        """Reads a CSV file in chunks and processes each chunk."""
        self._load_embedding_model()
        
        metrics = {
            "input_row_count": 0,
            "processed_row_count": 0,
            "candidate_count": 0,
            "alert_count": 0
        }
        
        self.repo.start_run_log(run_id, batch_id, pipeline_name="AML_Production_Pipeline")
        
        try:
            # First pass: count lines if possible, or just use pandas chunks
            logger.info(f"Starting batch processing for {file_path}")
            
            chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
            for chunk_idx, chunk in enumerate(chunk_iter):
                logger.info(f"Processing chunk {chunk_idx+1} ({len(chunk)} rows)...")
                metrics["input_row_count"] += len(chunk)
                
                # 1. Prepare data (clean text, etc. - skipping DB insert for brevity, but you'd normally insert to silver_eft_clean)
                chunk['normalized_explanation'] = chunk['description'].astype(str).str.lower() # very basic norm
                
                # 2. Get Embeddings
                explanations = chunk['normalized_explanation'].tolist()
                embeddings = self.embedding_model.encode(explanations, batch_size=self.batch_size, show_progress_bar=False)
                
                # 3. For each row, retrieve, score, and alert
                for row_idx, row in chunk.iterrows():
                    eft_id = row.get("eft_id", row_idx)  # Fallback to index if no ID
                    norm_exp = row["normalized_explanation"]
                    emb = embeddings[row_idx - chunk.index[0]].tolist()
                    
                    # Candidate Retrieval
                    candidates = self.retriever.get_merged_candidates(norm_exp, emb)
                    metrics["candidate_count"] += len(candidates)
                    
                    if not candidates:
                        metrics["processed_row_count"] += 1
                        continue
                        
                    # Reranking
                    candidates = self.reranker.score_candidates(norm_exp, candidates)
                    
                    # Final Scoring & Alert Generation
                    for cand in candidates:
                        # Assuming fuzzy, vector, acronym, rule are calculated somewhere.
                        # For now, we will simulate them based on candidate_score.
                        # In reality, you'd call your Matcher logic here.
                        
                        fuzzy_score = cand["candidate_score"] if cand["source"] in ["pg_trgm", "combined"] else 0.0
                        vector_score = cand["candidate_score"] if cand["source"] in ["pgvector", "combined"] else 0.0
                        acronym_score = 0.0 # Would be calculated by alias_utils
                        rule_score = 0.0 # Would be calculated by matcher
                        
                        scores_dict = {
                            "fuzzy_score": fuzzy_score,
                            "vector_score": vector_score,
                            "acronym_score": acronym_score,
                            "rule_score": rule_score,
                            "reranker_score": cand["reranker_score"]
                        }
                        
                        final_score = self.scorer.calculate_final_score(scores_dict)
                        risk_level = self.scorer.assign_risk_level(final_score)
                        
                        # DB Insert for scoring result
                        # self.repo.insert_scoring_result(...)
                        
                        if risk_level in ["HIGH", "MEDIUM"]:
                            metrics["alert_count"] += 1
                            # self.repo.insert_alert(...)
                            
                    metrics["processed_row_count"] += 1
                    
            self.repo.finish_run_log(run_id, metrics)
            logger.info("Batch processing finished successfully.")
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            self.repo.fail_run_log(run_id, str(e))
