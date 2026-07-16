-- 06_create_config_tables.sql

-- 1. Scoring Weight Tablosu
CREATE TABLE IF NOT EXISTS aml_config.scoring_weight (
    config_version TEXT PRIMARY KEY,
    fuzzy_weight NUMERIC(8,5) CHECK (fuzzy_weight >= 0),
    vector_weight NUMERIC(8,5) CHECK (vector_weight >= 0),
    acronym_weight NUMERIC(8,5) CHECK (acronym_weight >= 0),
    rule_weight NUMERIC(8,5) CHECK (rule_weight >= 0),
    reranker_weight NUMERIC(8,5) CHECK (reranker_weight >= 0),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT now()
);

-- 2. Threshold Tablosu
CREATE TABLE IF NOT EXISTS aml_config.threshold (
    threshold_id BIGSERIAL PRIMARY KEY,
    config_version TEXT,
    risk_level TEXT,
    min_score NUMERIC(8,5) CHECK (min_score >= 0 AND min_score <= 1),
    max_score NUMERIC(8,5) CHECK (max_score >= 0 AND max_score <= 1),
    CHECK (min_score <= max_score),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT now(),
    UNIQUE (config_version, risk_level)
);
