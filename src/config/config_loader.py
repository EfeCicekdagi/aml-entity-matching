import yaml
import os

class ConfigLoader:
    def __init__(self, config_path="config/aml_config.yaml"):
        self.config_path = config_path
        self._load_env_file()
        self.config = self._load_config()
        self._override_with_env()
        self._validate_config()

    def _load_env_file(self):
        """Basit bir .env dosyası okuyucusu (.env ortam değişkenlerini yükler)."""
        env_paths = [
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.path.dirname(self.config_path), "..", ".env"),
        ]
        for path in env_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and "=" in line:
                                k, v = line.split("=", 1)
                                k, v = k.strip(), v.strip().strip("'").strip('"')
                                if k not in os.environ:
                                    os.environ[k] = v
                except Exception:
                    pass
                break

    def _resolve_env_str(self, val):
        """YAML içindeki ${VAR} veya ${VAR:-default} sözdizimini çözümler."""
        if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
            inner = val[2:-1].strip()
            if ":-" in inner:
                var_name, default_val = inner.split(":-", 1)
                return os.getenv(var_name.strip(), default_val.strip())
            else:
                return os.getenv(inner, "")
        return val

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _override_with_env(self):
        """Override database and specific config keys using environment variables."""
        db_cfg = self.config.get("database", {})
        db_cfg["host"] = os.getenv("AML_DB_HOST", self._resolve_env_str(db_cfg.get("host")))
        db_cfg["port"] = int(os.getenv("AML_DB_PORT", self._resolve_env_str(db_cfg.get("port", 5432))))
        db_cfg["name"] = os.getenv("AML_DB_NAME", self._resolve_env_str(db_cfg.get("name")))
        db_cfg["user"] = os.getenv("AML_DB_USER", self._resolve_env_str(db_cfg.get("user")))
        db_cfg["password"] = os.getenv("AML_DB_PASSWORD", self._resolve_env_str(db_cfg.get("password")))

        sec_cfg = self.config.get("security", {})
        sec_cfg["db_sslmode"] = os.getenv("AML_DB_SSLMODE", self._resolve_env_str(sec_cfg.get("db_sslmode", "prefer")))
        if os.getenv("AML_ENABLE_AUDIT_TRAIL") is not None:
            sec_cfg["enable_audit_trail"] = os.getenv("AML_ENABLE_AUDIT_TRAIL").lower() in ("true", "1", "yes")
        if os.getenv("AML_APPEND_ONLY_HISTORY") is not None:
            sec_cfg["append_only_history"] = os.getenv("AML_APPEND_ONLY_HISTORY").lower() in ("true", "1", "yes")


    def _validate_config(self):
        """Temel konfigürasyon doğrulama kurallarını çalıştırır."""
        # 1. 0 <= threshold <= 1
        scoring = self.config.get("scoring", {})
        for key in ["high_threshold", "medium_threshold"]:
            val = scoring.get(key)
            if val is not None and not (0.0 <= float(val) <= 1.0):
                raise ValueError(f"Config validation error: scoring.{key} ({val}) must be between 0 and 1.")

        retrieval = self.config.get("retrieval", {})
        for key in ["min_trgm_score", "min_vector_score", "reranker_prefilter_score"]:
            val = retrieval.get(key)
            if val is not None and not (0.0 <= float(val) <= 1.0):
                raise ValueError(f"Config validation error: retrieval.{key} ({val}) must be between 0 and 1.")

        # 2. 0 <= weight <= 1 ve Ağırlıkların toplamı ≈ 1
        weights = scoring.get("weights", {})
        if weights:
            total_weight = 0.0
            for w_key, w_val in weights.items():
                w_num = float(w_val)
                if not (0.0 <= w_num <= 1.0):
                    raise ValueError(f"Config validation error: scoring.weights.{w_key} ({w_val}) must be between 0 and 1.")
                total_weight += w_num
            if abs(total_weight - 1.0) > 0.02:
                raise ValueError(f"Config validation error: sum of scoring.weights ({total_weight:.4f}) must be approximately 1.")

        # 3. Top-k değerleri pozitif integer
        top_k_keys = ["pg_trgm_top_k", "full_text_top_k", "pgvector_top_k", "merged_top_k", "reranker_top_k"]
        for key in top_k_keys:
            val = retrieval.get(key)
            if val is not None and (not isinstance(val, int) or val <= 0):
                raise ValueError(f"Config validation error: retrieval.{key} ({val}) must be a positive integer.")

        # 4. Dimension ve batch_size pozitif integer
        emb_cfg = self.config.get("embedding", {})
        dim = emb_cfg.get("dimension")
        if dim is not None and (not isinstance(dim, int) or dim <= 0):
            raise ValueError(f"Config validation error: embedding.dimension ({dim}) must be a positive integer.")
        batch_size = emb_cfg.get("batch_size")
        if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
            raise ValueError(f"Config validation error: embedding.batch_size ({batch_size}) must be a positive integer.")

        # 5. Database alanları boş değil
        db_cfg = self.config.get("database", {})
        for key in ["host", "port", "name", "user"]:
            if not db_cfg.get(key):
                raise ValueError(f"Config validation error: database.{key} must not be empty.")

        # 6. Device: auto / cpu / cuda
        for section_name in ["embedding", "ner", "reranker"]:
            sec = self.config.get(section_name, {})
            dev = sec.get("device")
            if dev is not None and dev not in ("auto", "cpu", "cuda"):
                raise ValueError(f"Config validation error: {section_name}.device ('{dev}') must be one of: 'auto', 'cpu', 'cuda'.")

        # 7. Calibration method desteklenen değerlerden biri
        calib_cfg = self.config.get("calibration", {})
        method = calib_cfg.get("method")
        allowed_methods = [None, "PLATT_SCALING", "ISOTONIC_REGRESSION"]
        if method not in allowed_methods:
            raise ValueError(f"Config validation error: calibration.method ('{method}') must be one of {allowed_methods}.")

    def get_db_config(self):
        db_cfg = self.config.get("database", {}).copy()
        sec_cfg = self.config.get("security", {})
        db_cfg["sslmode"] = sec_cfg.get("db_sslmode", "prefer")
        db_cfg["enable_audit_trail"] = sec_cfg.get("enable_audit_trail", True)
        db_cfg["append_only_history"] = sec_cfg.get("append_only_history", True)
        return db_cfg


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

