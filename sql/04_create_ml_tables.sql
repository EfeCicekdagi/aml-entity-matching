-- 04_create_ml_tables.sql

-- 1. Sirket Vektorleri Tablosu
CREATE TABLE IF NOT EXISTS aml_ml.company_embedding (
    embedding_id BIGSERIAL PRIMARY KEY,
    variant_id BIGINT REFERENCES aml_stage.company_variant(variant_id) ON DELETE CASCADE,
    company_id BIGINT,
    embedding VECTOR(1024),
    embedding_model_name TEXT,
    embedding_model_version TEXT,
    normalization_version TEXT,
    embedding_hash TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT now(),
    
    -- Ayni varyasyon icin ayni modelin birden fazla vektor uretmesini engelle
    UNIQUE (variant_id, embedding_model_name, embedding_model_version)
);

-- 2. Reranker Cache Tablosu
-- Reranker skorlari aciklama metnine gore (EFT'ye gore degil) cachelenir ki benzer aciklamalar tekrar hesaplanmasin
CREATE TABLE IF NOT EXISTS aml_ml.reranker_cache (
    cache_key TEXT PRIMARY KEY,
    normalized_explanation TEXT,
    variant_id BIGINT REFERENCES aml_stage.company_variant(variant_id) ON DELETE CASCADE,
    reranker_score NUMERIC(8,5),
    reranker_model_version TEXT,
    created_at TIMESTAMP DEFAULT now()
);
