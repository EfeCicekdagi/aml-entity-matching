import yaml
import os

class ConfigLoader:
    def __init__(self, config_path="config/aml_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_db_config(self):
        return self.config.get("database", {})

    def get_embedding_config(self):
        return self.config.get("embedding", {})

    def get_retrieval_config(self):
        return self.config.get("retrieval", {})

    def get_reranker_config(self):
        return self.config.get("reranker", {})

    def get_ner_config(self):
        return self.config.get("ner", {})

    def get_scoring_config(self):
        return self.config.get("scoring", {})
