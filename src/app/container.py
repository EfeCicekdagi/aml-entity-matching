import logging
import torch

from src.config.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository
from src.retrieval.postgres_candidate_retriever import PostgresCandidateRetriever
from src.models.reranker import Reranker
from src.scoring.final_scorer import FinalScorer
from src.models.calibration import CalibrationWrapper
from src.pipeline.inference_service import AMLInferenceService

logger = logging.getLogger(__name__)

class ApplicationContainer:
    def __init__(self, config_loader: ConfigLoader = None):
        if config_loader is None:
            self.config_loader = ConfigLoader()
        else:
            self.config_loader = config_loader
            
        self.config = self.config_loader.config
        self.db_config = self.config_loader.get_db_config()
        self.retrieval_config = self.config_loader.get_retrieval_config()
        self.reranker_config = self.config_loader.get_reranker_config()
        self.scoring_config = self.config_loader.get_scoring_config()
        
        self.repo = None
        self.retriever = None
        self.reranker = None
        self.scorer = None
        self.calibration = None
        self.entity_extractor = None
        self.embedding_model = None
        self.inference_service = None
        
    def init_resources(self):
        # 1. Repository
        self.repo = AMLRepository(
            host=self.db_config.get("host"),
            port=self.db_config.get("port"),
            dbname=self.db_config.get("name"),
            user=self.db_config.get("user"),
            password=self.db_config.get("password")
        )
        
        # 2. Retriever
        self.retriever = PostgresCandidateRetriever(self.repo, self.retrieval_config)
        
        # 3. Scorer & Calibration
        self.scorer = FinalScorer(
            self.repo, 
            config_version=self.scoring_config.get("scoring_config_version", "scoring_v2_reranker"),
            threshold_version=self.scoring_config.get("threshold_config_version", "threshold_v2_reranker")
        )
        if self.config.get("scoring", {}).get("enable_isotonic_calibration"):
            self.calibration = CalibrationWrapper(self.repo, "bge-m3-isotonic-v1")
            
        # 4. Entity Extractor
        ner_enabled = self.config.get("ner", {}).get("enabled", False)
        if ner_enabled:
            try:
                from src.models.ner_extractor import NERExtractor
                from src.models.entity_extractor import EntityExtractor
                ner_model = self.config.get("ner", {}).get("model_name", "savasy/bert-base-turkish-ner-cased")
                ner_dev = self.config.get("ner", {}).get("device", "auto")
                logger.info(f"Loading NER model: {ner_model}...")
                ner_device_id = 0 if (ner_dev == "cuda" or (ner_dev == "auto" and torch.cuda.is_available())) else -1
                ner_ext = NERExtractor(model_name=ner_model, device=ner_device_id)
                self.entity_extractor = EntityExtractor(
                    ner_extractor=ner_ext,
                    config=self.config.get("entity_extraction")
                )
            except Exception as e:
                logger.error(f"Failed to load NER model: {e}")
                
        # 5. Embedding Model
        emb_name = self.config.get("embedding", {}).get("model_name", "BAAI/bge-m3")
        emb_dev_cfg = self.config.get("embedding", {}).get("device", "auto")
        emb_device = "cuda" if (emb_dev_cfg in ("auto", None) and torch.cuda.is_available()) else (
            emb_dev_cfg if emb_dev_cfg not in ("auto", None) else "cpu"
        )
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(emb_name, device=emb_device)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            
        # 6. Reranker
        self.reranker = Reranker(self.repo, self.reranker_config)
        
        # 7. Inference Service
        self.inference_service = AMLInferenceService(
            config=self.config,
            retriever=self.retriever,
            reranker=self.reranker,
            entity_extractor=self.entity_extractor,
            embedding_model=self.embedding_model,
            scorer=self.scorer,
            calibration=self.calibration
        )
