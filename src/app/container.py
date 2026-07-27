import logging
import torch

from src.config.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository
from src.retrieval.postgres_candidate_retriever import PostgresCandidateRetriever
from src.models.reranker import Reranker
from src.scoring.final_scorer import FinalScorer
from src.models.calibration import CalibrationWrapper
from src.pipeline.inference_service import AMLInferenceService
from src.pipeline.match_engine import MatchEngine

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
            password=self.db_config.get("password"),
            sslmode=self.db_config.get("sslmode", self.config.get("security", {}).get("db_sslmode", "prefer")),
            enable_audit_trail=self.db_config.get("enable_audit_trail", self.config.get("security", {}).get("enable_audit_trail", True)),
            append_only_history=self.db_config.get("append_only_history", self.config.get("security", {}).get("append_only_history", True)),
        )
        
        # 2. Retriever
        self.retriever = PostgresCandidateRetriever(self.repo, self.retrieval_config)
        
        # 3. Scorer & Calibration
        self.scorer = FinalScorer(
            self.repo, 
            config_version=self.scoring_config.get("scoring_config_version", "scoring_v2_reranker"),
            threshold_version=self.scoring_config.get("threshold_config_version", "threshold_v2_reranker")
        )
        calibration_config = self.config.get("calibration", {})
        if calibration_config.get("enabled", False):
            self.calibration = CalibrationWrapper(
                calibration_model_path=calibration_config.get("model_path"),
                calibration_version=calibration_config.get("version"),
                calibration_method=calibration_config.get("method"),
            )
            
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
                logger.warning(
                    f"Failed to load NER model: {e}. Graceful degradation applied: Pipeline will continue without entity extraction."
                )
                self.entity_extractor = None
        else:
            logger.info("NER model is optional and disabled in config. Pipeline will continue without entity extraction.")
            self.entity_extractor = None
                
        # 5. Embedding Model
        emb_enabled = self.config.get("embedding", {}).get("enabled", True)
        if emb_enabled:
            emb_name = self.config.get("embedding", {}).get("model_name", "BAAI/bge-m3")
            emb_dev_cfg = self.config.get("embedding", {}).get("device", "auto")
            emb_device = "cuda" if (emb_dev_cfg in ("auto", None) and torch.cuda.is_available()) else (
                emb_dev_cfg if emb_dev_cfg not in ("auto", None) else "cpu"
            )
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(emb_name, device=emb_device)

                # Vektör boyutu denetimi (DB kolonu ile Model boyutu eşleşiyor mu?)
                try:
                    model_dim = self.embedding_model.get_sentence_embedding_dimension()
                except Exception:
                    model_dim = self.config.get("embedding", {}).get("dimension", 1024)

                if self.retriever and not self.retriever.validate_vector_dimension(model_dim):
                    logger.warning(
                        f"Vector dimension mismatch between model ({model_dim}) and DB column. Graceful degradation applied: Vector retrieval channel will be disabled."
                    )
                    self.retriever.disable_vector_retrieval(reason=f"Vector dimension mismatch (model dim={model_dim})")
            except Exception as e:
                logger.warning(
                    f"Failed to load embedding model ({emb_name}): {e}. Graceful degradation applied: Pipeline will continue without embedding model and vector retrieval channel will be disabled."
                )
                self.embedding_model = None
                if self.retriever:
                    self.retriever.disable_vector_retrieval(reason="Embedding model failed to load")
        else:
            logger.info("Embedding model is optional and disabled in config. Pipeline will continue without embedding model and vector retrieval channel will be disabled.")
            self.embedding_model = None
            if self.retriever:
                self.retriever.disable_vector_retrieval(reason="Embedding model disabled in config")
            
        # 6. Reranker
        if self.reranker_config.get("enabled", True):
            self.reranker = Reranker(
                self.repo,
                self.reranker_config
            )
        
        # 7. Match Engine
        self.match_engine = MatchEngine(
            config=self.config,
            retriever=self.retriever,
            reranker=self.reranker,
            entity_extractor=self.entity_extractor,
            embedding_model=self.embedding_model
        )
        
        # 8. Inference Service
        self.inference_service = AMLInferenceService(
            config=self.config,
            scorer=self.scorer,
            calibration=self.calibration,
            match_engine=self.match_engine
        )
