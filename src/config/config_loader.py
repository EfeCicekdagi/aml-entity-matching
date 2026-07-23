import yaml
import os

class ConfigLoader:
    def __init__(self, config_path="config/aml_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._override_with_env()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _override_with_env(self):
        """Override database and specific config keys using environment variables."""
        db_cfg = self.config.get("database", {})
        db_cfg["host"] = os.getenv("AML_DB_HOST", db_cfg.get("host"))
        db_cfg["port"] = int(os.getenv("AML_DB_PORT", db_cfg.get("port", 5432)))
        db_cfg["name"] = os.getenv("AML_DB_NAME", db_cfg.get("name"))
        db_cfg["user"] = os.getenv("AML_DB_USER", db_cfg.get("user"))
        db_cfg["password"] = os.getenv("AML_DB_PASSWORD", db_cfg.get("password"))

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

