-- 05_create_core_tables.sql

-- 1. Candidate Match Tablosu
CREATE TABLE IF NOT EXISTS aml_core.candidate_match (
    candidate_id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    eft_id BIGINT,
    company_id BIGINT,
    variant_id BIGINT REFERENCES aml_stage.company_variant(variant_id) ON DELETE CASCADE,
    candidate_source TEXT,
    candidate_rank INT,
    candidate_score NUMERIC(8,5),
    created_at TIMESTAMP DEFAULT now()
);

-- 2. Scoring Result Tablosu
CREATE TABLE IF NOT EXISTS aml_core.scoring_result (
    scoring_id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    eft_id BIGINT,
    variant_id BIGINT REFERENCES aml_stage.company_variant(variant_id) ON DELETE CASCADE,
    fuzzy_score NUMERIC(8,5),
    vector_score NUMERIC(8,5),
    acronym_score NUMERIC(8,5),
    rule_score NUMERIC(8,5),
    reranker_score NUMERIC(8,5),
    final_score NUMERIC(8,5),
    fuzzy_score NUMERIC(8,5),
    vector_score NUMERIC(8,5),
    reranker_score NUMERIC(8,5),
    risk_level TEXT,
    scoring_config_version TEXT,
    threshold_config_version TEXT,
    embedding_model_version TEXT,
    reranker_model_version TEXT,
    matched_text_span TEXT,
    matched_tokens TEXT,
    explanation TEXT,
    created_at TIMESTAMP DEFAULT now()
);

-- 3. Alert Tablosu
CREATE TABLE IF NOT EXISTS aml_core.alert (
    alert_id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    eft_id BIGINT,
    company_id BIGINT,
    variant_id BIGINT REFERENCES aml_stage.company_variant(variant_id) ON DELETE CASCADE,
    risk_level TEXT,
    alert_status TEXT DEFAULT 'OPEN',
    final_score NUMERIC(8,5),
    extracted_entity TEXT,
    created_at TIMESTAMP DEFAULT now()
);
