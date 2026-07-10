import sys
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository
from src.retrieval.postgres_candidate_retriever import PostgresCandidateRetriever
from src.reranker.reranker import Reranker
from src.scoring.final_scorer import FinalScorer
from src.etl.batch_processor import BatchProcessor
from sentence_transformers import SentenceTransformer
from src.utils.ner_extractor import NERExtractor

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

# Global instances
components = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing AML Models and Repository...")
    cfg = ConfigLoader()
    repo = AMLRepository(
        cfg.get_db_config()["host"],
        cfg.get_db_config()["port"],
        cfg.get_db_config()["name"],
        cfg.get_db_config()["user"],
        cfg.get_db_config()["password"]
    )
    
    components["repo"] = repo
    components["retriever"] = PostgresCandidateRetriever(repo, cfg.get_retrieval_config())
    components["reranker"] = Reranker(repo, cfg.get_reranker_config())
    components["scorer"] = FinalScorer(repo)
    
    # Load NER if enabled
    ner_cfg = cfg.get_ner_config()
    if ner_cfg.get("enabled", True):
        components["ner"] = NERExtractor(model_name=ner_cfg.get("model_name"), device=-1) # CPU by default for stability in dev
    else:
        components["ner"] = None
        
    # Load Embedding
    emb_cfg = cfg.get_embedding_config()
    logger.info(f"Loading embedding model {emb_cfg['model_name']}...")
    components["embedding"] = SentenceTransformer(emb_cfg["model_name"])
    
    logger.info("AML API is ready to accept requests.")
    yield
    logger.info("Shutting down API...")

app = FastAPI(title="AML Entity Matching API", lifespan=lifespan)

class TransactionInput(BaseModel):
    transaction_id: str
    explanation: str

class CandidateOutput(BaseModel):
    variant_name: str
    original_company_name: str
    retrieval_score: float
    reranker_score: float

class ScoringResult(BaseModel):
    transaction_id: str
    risk_level: str
    final_score: float
    extracted_entity: Optional[str]
    candidates: List[CandidateOutput]

@app.post("/api/v1/score_transaction", response_model=ScoringResult)
async def score_transaction(data: TransactionInput):
    explanation = data.explanation.strip()
    
    # 1. Extract Entity
    extracted = None
    if components["ner"]:
        extracted = components["ner"].extract_entity(explanation)
        
    # Search query is always the full explanation for robust vector search
    search_query = explanation
    
    # 2. Embed
    embedding = components["embedding"].encode([search_query])[0].tolist()
    
    # 3. Retrieve
    batch_rows = [{
        "row_id": data.transaction_id,
        "normalized_explanation": search_query,
        "embedding": embedding,
        "extracted_entity": extracted
    }]
    candidates_dict = components["retriever"].batch_get_candidates(batch_rows)
    candidates = candidates_dict.get(data.transaction_id, [])
    
    if not candidates:
        return ScoringResult(
            transaction_id=data.transaction_id,
            risk_level="NO_MATCH",
            final_score=0.0,
            extracted_entity=extracted,
            candidates=[]
        )
        
    # 4. Rerank
    candidates = components["reranker"].score_candidates(search_query, candidates)
    
    # 5. Score
    best_candidate = None
    best_score = -1.0
    
    output_candidates = []
    
    for c in candidates:
        fuzzy_score = c["candidate_score"] if c["source"] in ["pg_trgm", "combined"] else 0.0
        vec_score = c["candidate_score"] if c["source"] in ["pgvector", "combined"] else 0.0
        
        scores_dict = {
            "fuzzy_score": fuzzy_score,
            "vector_score": vec_score,
            "reranker_score": c.get("reranker_score", 0.0),
            "acronym_score": 0.0,
            "rule_score": 0.0
        }
        
        final_score = components["scorer"].calculate_final_score(scores_dict)
        c["final_score"] = final_score
        
        output_candidates.append(CandidateOutput(
            variant_name=c["variant_name"],
            original_company_name=c["original_company_name"],
            retrieval_score=c["candidate_score"],
            reranker_score=c.get("reranker_score", 0.0)
        ))
        
        if final_score > best_score:
            best_score = final_score
            best_candidate = c
            
    risk_level = components["scorer"].assign_risk_level(best_score)
    
    return ScoringResult(
        transaction_id=data.transaction_id,
        risk_level=risk_level,
        final_score=best_score,
        extracted_entity=extracted,
        candidates=sorted(output_candidates, key=lambda x: x.reranker_score, reverse=True)[:5]
    )

def run_batch_pipeline():
    from src.main import main as batch_main
    batch_main()

@app.post("/api/v1/batch_scan")
async def batch_scan(background_tasks: BackgroundTasks):
    """Triggers the massive asynchronous batch processing pipeline"""
    background_tasks.add_task(run_batch_pipeline)
    return {"message": "Batch scan started in background. Check AML Run Logs for progress."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
