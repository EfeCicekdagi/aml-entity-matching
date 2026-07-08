CREATE TABLE IF NOT EXISTS bronze_eft_raw (
    eft_id BIGINT PRIMARY KEY,
    transaction_date DATE,
    amount NUMERIC(18,2),
    sender_account_id TEXT,
    receiver_account_id TEXT,
    explanation TEXT,
    source_system TEXT,
    batch_id TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS bronze_blacklist_company_raw (
    company_id BIGSERIAL PRIMARY KEY,
    company_name TEXT NOT NULL,
    source_list TEXT,
    list_version TEXT,
    risk_category TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS silver_eft_clean (
    eft_id BIGINT PRIMARY KEY,
    transaction_date DATE,
    amount NUMERIC(18,2),
    original_explanation TEXT,
    normalized_explanation TEXT,
    explanation_tsv TSVECTOR,
    batch_id TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS silver_company_variant (
    variant_id BIGSERIAL PRIMARY KEY,
    company_id BIGINT REFERENCES bronze_blacklist_company_raw(company_id),
    original_company_name TEXT,
    variant_name TEXT,
    normalized_variant_name TEXT,
    variant_type TEXT,
    list_version TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS gold_company_embedding (
    embedding_id BIGSERIAL PRIMARY KEY,
    variant_id BIGINT REFERENCES silver_company_variant(variant_id),
    company_id BIGINT,
    embedding VECTOR(384),
    embedding_model_name TEXT,
    embedding_model_version TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS aml_candidate_match (
    candidate_id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    eft_id BIGINT,
    company_id BIGINT,
    variant_id BIGINT,
    candidate_source TEXT,
    candidate_rank INT,
    candidate_score NUMERIC(8,5),
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS aml_scoring_result (
    scoring_id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    eft_id BIGINT,
    company_id BIGINT,
    variant_id BIGINT,

    fuzzy_score NUMERIC(8,5),
    vector_score NUMERIC(8,5),
    acronym_score NUMERIC(8,5),
    rule_score NUMERIC(8,5),
    reranker_score NUMERIC(8,5),

    final_score NUMERIC(8,5),
    risk_level TEXT,

    scoring_config_version TEXT,
    threshold_config_version TEXT,
    embedding_model_version TEXT,
    reranker_model_version TEXT,
    
    matched_text_span TEXT,
    matched_tokens TEXT[],
    explanation JSONB,

    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS aml_alert (
    alert_id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    eft_id BIGINT,
    company_id BIGINT,
    variant_id BIGINT,
    final_score NUMERIC(8,5),
    risk_level TEXT,
    alert_status TEXT DEFAULT 'OPEN',
    created_at TIMESTAMP DEFAULT now(),
    reviewed_at TIMESTAMP,
    review_result TEXT,
    review_note TEXT
);

CREATE TABLE IF NOT EXISTS aml_run_log (
    run_id TEXT PRIMARY KEY,
    pipeline_name TEXT,
    batch_id TEXT,
    started_at TIMESTAMP,
    finished_at TIMESTAMP,
    input_row_count BIGINT,
    processed_row_count BIGINT,
    candidate_count BIGINT,
    alert_count BIGINT,
    status TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS aml_scoring_weight_config (
    config_id BIGSERIAL PRIMARY KEY,
    config_version TEXT,
    fuzzy_weight NUMERIC(5,4),
    vector_weight NUMERIC(5,4),
    acronym_weight NUMERIC(5,4),
    rule_weight NUMERIC(5,4),
    reranker_weight NUMERIC(5,4),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS aml_threshold_config (
    threshold_id BIGSERIAL PRIMARY KEY,
    config_version TEXT,
    risk_level TEXT,
    min_score NUMERIC(8,5),
    max_score NUMERIC(8,5),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS aml_reranker_cache (
    cache_key TEXT PRIMARY KEY,
    normalized_explanation TEXT,
    variant_id BIGINT,
    reranker_score NUMERIC(8,5),
    reranker_model_version TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS aml_quality_check_result (
    check_id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    check_name TEXT,
    check_status TEXT,
    check_value TEXT,
    threshold_value TEXT,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS aml_performance_log (
    perf_id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    input_count BIGINT,
    duration_seconds NUMERIC(12,2),
    rows_per_second NUMERIC(12,2),
    avg_candidate_count NUMERIC(8,2),
    reranker_duration_seconds NUMERIC(12,2),
    db_duration_seconds NUMERIC(12,2),
    created_at TIMESTAMP DEFAULT now()
);
