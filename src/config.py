# =========================
# File Paths
# =========================

EFT_FILE_PATH = "data/mock_bank_efts.csv"
COMPANY_FILE_PATH = "data/company_list.csv"
OUTPUT_DIR = "outputs"

RESULTS_OUTPUT_PATH = f"{OUTPUT_DIR}/results.csv"
BEST_MATCHES_OUTPUT_PATH = f"{OUTPUT_DIR}/best_matches.csv"
SUSPICIOUS_EFTS_OUTPUT_PATH = f"{OUTPUT_DIR}/suspicious_efts.csv"


# =========================
# Chunk Processing
# =========================

CHUNK_SIZE = 200


# =========================
# Candidate Filtering
# =========================

MIN_CANDIDATE_SCORE = 0.4
MAX_CANDIDATES = 20
FUZZY_FALLBACK_LIMIT = 5


# =========================
# Final Score Weights
# =========================

FUZZY_WEIGHT = 0.50
VECTOR_WEIGHT = 0.30
ACRONYM_WEIGHT = 0.15
RULE_WEIGHT = 0.05


# =========================
# Risk Thresholds
# =========================

HIGH_RISK_THRESHOLD = 0.85
MEDIUM_RISK_THRESHOLD = 0.70
LOW_RISK_THRESHOLD = 0.65


# =========================
# Embedding Model
# =========================

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 32


# =========================
# Vector Database Engine
# =========================

# Options: "faiss", "postgres", "compare"
VECTOR_ENGINE = "compare"

# PostgreSQL Settings
PG_HOST = "localhost"
PG_PORT = "5432"
PG_DB = "aml_db"
PG_USER = "postgres"
PG_PASSWORD = "password"