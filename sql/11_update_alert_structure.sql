-- 12_update_alert_structure.sql

-- 1. aml_core.alert tablosuna eklenecek yeni kolonlar
ALTER TABLE aml_core.alert
    ADD COLUMN IF NOT EXISTS entity_extraction_status VARCHAR(50),
    ADD COLUMN IF NOT EXISTS fuzzy_score NUMERIC(8,5),
    ADD COLUMN IF NOT EXISTS vector_score NUMERIC(8,5),
    ADD COLUMN IF NOT EXISTS reranker_score NUMERIC(8,5),
    ADD COLUMN IF NOT EXISTS matched_variant_name TEXT,
    ADD COLUMN IF NOT EXISTS variant_type TEXT,
    ADD COLUMN IF NOT EXISTS watchlist_company_name TEXT,
    ADD COLUMN IF NOT EXISTS reviewed_by VARCHAR(100),
    ADD COLUMN IF NOT EXISTS reviewed_at TIMESTAMP,
    ADD COLUMN IF NOT EXISTS review_result VARCHAR(50),
    ADD COLUMN IF NOT EXISTS analyst_note TEXT,
    ADD COLUMN IF NOT EXISTS false_positive_reason TEXT,
    ADD COLUMN IF NOT EXISTS status_updated_at TIMESTAMP;

-- Mevcut run_id için sıkılaştırma (NOT NULL garantisi, öncelikle NULL olanlara dummy değer atayalım)
UPDATE aml_core.alert SET run_id = 'unknown_run' WHERE run_id IS NULL;
ALTER TABLE aml_core.alert ALTER COLUMN run_id SET NOT NULL;

-- 2. aml_audit.run_log tablosuna versiyon bilgileri
ALTER TABLE aml_audit.run_log
    ADD COLUMN IF NOT EXISTS scoring_config_version TEXT,
    ADD COLUMN IF NOT EXISTS threshold_version TEXT,
    ADD COLUMN IF NOT EXISTS pipeline_version TEXT,
    ADD COLUMN IF NOT EXISTS embedding_model_version TEXT,
    ADD COLUMN IF NOT EXISTS reranker_model_version TEXT;
