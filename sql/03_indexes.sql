CREATE INDEX IF NOT EXISTS idx_company_variant_trgm
ON silver_company_variant
USING gin (normalized_variant_name gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_eft_explanation_tsv
ON silver_eft_clean
USING gin (explanation_tsv);

CREATE INDEX IF NOT EXISTS idx_company_embedding_hnsw
ON gold_company_embedding
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_silver_eft_batch_id
ON silver_eft_clean(batch_id);

CREATE INDEX IF NOT EXISTS idx_silver_eft_transaction_date
ON silver_eft_clean(transaction_date);

CREATE INDEX IF NOT EXISTS idx_candidate_run_id
ON aml_candidate_match(run_id);

CREATE INDEX IF NOT EXISTS idx_scoring_run_id
ON aml_scoring_result(run_id);

CREATE INDEX IF NOT EXISTS idx_alert_run_id
ON aml_alert(run_id);
