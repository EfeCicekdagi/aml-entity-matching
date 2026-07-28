-- 21_create_alert_export_table.sql
-- Alert ve EFT verilerinin birleştirilmiş haliyle tek bir tabloda tutulması.

DROP TABLE IF EXISTS aml_core.alert_export CASCADE;

CREATE TABLE IF NOT EXISTS aml_core.alert_export (
    export_id BIGSERIAL PRIMARY KEY,
    alert_id BIGINT NOT NULL,
    run_id TEXT NOT NULL,
    eft_id BIGINT NOT NULL,
    transaction_date DATE,
    amount NUMERIC,
    sender_account_id TEXT,
    receiver_account_id TEXT,
    original_explanation TEXT,
    source_system TEXT,
    batch_id TEXT,
    original_company_name TEXT,
    final_score NUMERIC(8,5),
    fuzzy_score NUMERIC(8,5),
    vector_score NUMERIC(8,5),
    reranker_score NUMERIC(8,5),
    risk_level TEXT,
    alert_status TEXT,
    extracted_entity TEXT,
    match_reason TEXT,
    created_at TIMESTAMP,
    entity_extraction_status TEXT,
    matched_variant_name TEXT,
    variant_type TEXT,
    watchlist_company_name TEXT,
    reviewed_by TEXT,
    reviewed_at TIMESTAMP,
    review_result TEXT,
    analyst_note TEXT,
    false_positive_reason TEXT,
    status_updated_at TIMESTAMP,
    decision_status TEXT,
    reason_codes JSONB,
    calibrated_probability NUMERIC(8,5),
    calibration_applied BOOLEAN,
    entity_type TEXT,
    extraction_method TEXT,
    candidate_count INT,
    human_explanation TEXT,
    retrieval_sources JSONB,
    candidate_rank INT DEFAULT 1,  -- Her EFT icin en iyi eslesme = 1; UI'da rank=1 filtresiyle tekrar onlenir
    exported_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_alert_export_run_id ON aml_core.alert_export(run_id);
CREATE INDEX IF NOT EXISTS idx_alert_export_alert_id ON aml_core.alert_export(alert_id);
CREATE INDEX IF NOT EXISTS idx_alert_export_risk_level ON aml_core.alert_export(risk_level);
CREATE INDEX IF NOT EXISTS idx_alert_export_final_score ON aml_core.alert_export(final_score DESC);
CREATE INDEX IF NOT EXISTS idx_alert_export_run_rank ON aml_core.alert_export(run_id, candidate_rank);

COMMENT ON TABLE aml_core.alert_export IS
    'EFT input bilgileriyle Alert skor sonuçlarının birleştiği Flat tablo. UI ve Dış sistemler için üretilir.';
COMMENT ON COLUMN aml_core.alert_export.candidate_rank IS
    'Bu EFT için adayın nihai skor sıralaması (1 = en yüksek). UI filtrelemesinde sadece rank=1 gösterilir.';

