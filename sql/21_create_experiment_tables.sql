-- 25_create_experiment_tables.sql
-- Isolated schema for experiment frameworks. Safe to drop anytime.
CREATE SCHEMA IF NOT EXISTS aml_experiment;

-- 1. Experiment Runs
CREATE TABLE IF NOT EXISTS aml_experiment.experiment_run (
    experiment_id BIGSERIAL PRIMARY KEY,
    run_name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT now()
);

-- 2. Weight Analysis (Grid Search Results)
CREATE TABLE IF NOT EXISTS aml_experiment.weight_analysis (
    id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT REFERENCES aml_experiment.experiment_run(experiment_id) ON DELETE CASCADE,
    fuzzy_weight NUMERIC(4,2),
    vector_weight NUMERIC(4,2),
    reranker_weight NUMERIC(4,2),
    high_threshold NUMERIC(4,2),
    medium_threshold NUMERIC(4,2),
    precision_score NUMERIC(5,4),
    recall_score NUMERIC(5,4),
    f1_score NUMERIC(5,4),
    accuracy NUMERIC(5,4),
    tp INT,
    fp INT,
    tn INT,
    fn INT,
    created_at TIMESTAMP DEFAULT now()
);

-- 3. Threshold Analysis
CREATE TABLE IF NOT EXISTS aml_experiment.threshold_analysis (
    id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT REFERENCES aml_experiment.experiment_run(experiment_id) ON DELETE CASCADE,
    base_weight_id BIGINT REFERENCES aml_experiment.weight_analysis(id) ON DELETE CASCADE,
    high_threshold NUMERIC(4,2),
    medium_threshold NUMERIC(4,2),
    precision_score NUMERIC(5,4),
    recall_score NUMERIC(5,4),
    f1_score NUMERIC(5,4),
    roc_auc NUMERIC(5,4),
    pr_auc NUMERIC(5,4),
    tp INT,
    fp INT,
    tn INT,
    fn INT,
    created_at TIMESTAMP DEFAULT now()
);

-- 4. Decision Analysis & Score Comparison
CREATE TABLE IF NOT EXISTS aml_experiment.decision_analysis (
    id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT REFERENCES aml_experiment.experiment_run(experiment_id) ON DELETE CASCADE,
    eft_id BIGINT,
    expected_company TEXT,
    retrieved_company TEXT,
    fuzzy_score NUMERIC(5,4),
    vector_score NUMERIC(5,4),
    reranker_score NUMERIC(5,4),
    final_score NUMERIC(5,4),
    predicted_label TEXT,
    expected_label TEXT,
    is_correct BOOLEAN,
    error_type TEXT, -- FP, FN
    reason_code TEXT,
    created_at TIMESTAMP DEFAULT now()
);

COMMENT ON SCHEMA aml_experiment IS 'Isolated schema for ML experiments. Safe to drop.';
