-- 07_create_audit_tables.sql

-- 1. Run Log Tablosu
CREATE TABLE IF NOT EXISTS aml_audit.run_log (
    run_id TEXT PRIMARY KEY,
    pipeline_name TEXT,
    started_at TIMESTAMP DEFAULT now(),
    finished_at TIMESTAMP,
    status TEXT,
    processed_row_count INT,
    alert_count INT,
    embedding_model TEXT,
    reranker_model TEXT,
    error_message TEXT
);

-- 2. Quality Check Result Tablosu
CREATE TABLE IF NOT EXISTS aml_audit.quality_check_result (
    check_id BIGSERIAL PRIMARY KEY,
    run_id TEXT,
    check_name TEXT,
    check_status TEXT,
    check_value TEXT,
    threshold_value TEXT,
    created_at TIMESTAMP DEFAULT now()
);

-- 3. Performance Log Tablosu
CREATE TABLE IF NOT EXISTS aml_audit.performance_log (
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
