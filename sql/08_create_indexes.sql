-- 08_create_indexes.sql

-- ==========================================
-- 1. Metin Eslesmesi Icin GIN/GiST Indeksleri (pg_trgm)
-- ==========================================
CREATE INDEX IF NOT EXISTS idx_eft_clean_norm_exp_trgm 
    ON aml_stage.eft_clean USING gist (normalized_explanation gist_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_comp_var_norm_name_trgm 
    ON aml_stage.company_variant USING gist (normalized_variant_name gist_trgm_ops);

-- ==========================================
-- 2. Vektor Aramalari Icin HNSW Indeksi
-- ==========================================
-- Not: HNSW indeksi okuma performansi acisindan IVFFlat'a gore cok daha hizlidir.
-- pgvector 0.5.0+ uzerinde HNSW desteklenmektedir. 
-- vector_cosine_ops kullanarak Cosine Similarity bazli aramalari hizlandiriyoruz.
CREATE INDEX IF NOT EXISTS idx_comp_embedding_hnsw 
    ON aml_ml.company_embedding USING hnsw (embedding vector_cosine_ops);

-- ==========================================
-- 3. B-Tree Indeksleri (Performans ve Join)
-- ==========================================
-- Stage
CREATE INDEX IF NOT EXISTS idx_eft_clean_batch_id ON aml_stage.eft_clean(batch_id);
CREATE INDEX IF NOT EXISTS idx_eft_clean_date ON aml_stage.eft_clean(transaction_date);

CREATE INDEX IF NOT EXISTS idx_comp_var_company_id ON aml_stage.company_variant(company_id);
CREATE INDEX IF NOT EXISTS idx_comp_var_list_version ON aml_stage.company_variant(list_version);
CREATE INDEX IF NOT EXISTS idx_comp_var_active ON aml_stage.company_variant(is_active);

-- ML
CREATE INDEX IF NOT EXISTS idx_comp_emb_variant_id ON aml_ml.company_embedding(variant_id);
CREATE INDEX IF NOT EXISTS idx_comp_emb_model ON aml_ml.company_embedding(embedding_model_name, embedding_model_version);

-- Core
CREATE INDEX IF NOT EXISTS idx_cand_match_run_id ON aml_core.candidate_match(run_id);
CREATE INDEX IF NOT EXISTS idx_cand_match_eft_id ON aml_core.candidate_match(eft_id);

CREATE INDEX IF NOT EXISTS idx_score_res_run_id ON aml_core.scoring_result(run_id);
CREATE INDEX IF NOT EXISTS idx_score_res_eft_id ON aml_core.scoring_result(eft_id);

CREATE INDEX IF NOT EXISTS idx_alert_run_id ON aml_core.alert(run_id);
CREATE INDEX IF NOT EXISTS idx_alert_eft_id ON aml_core.alert(eft_id);
CREATE INDEX IF NOT EXISTS idx_alert_status ON aml_core.alert(alert_status);
CREATE INDEX IF NOT EXISTS idx_alert_risk ON aml_core.alert(risk_level);
