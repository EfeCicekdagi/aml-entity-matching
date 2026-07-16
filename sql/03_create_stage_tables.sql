-- 03_create_stage_tables.sql

-- 1. Temizlenmis EFT Tablosu
CREATE TABLE IF NOT EXISTS aml_stage.eft_clean (
    eft_id BIGINT PRIMARY KEY,
    transaction_date DATE,
    amount NUMERIC,
    original_explanation TEXT,
    normalized_explanation TEXT,
    explanation_tsv tsvector,
    source_system TEXT,
    batch_id TEXT,
    normalization_version TEXT,
    created_at TIMESTAMP DEFAULT now()
);

-- 2. Sirket Varyasyonlari Tablosu
CREATE TABLE IF NOT EXISTS aml_stage.company_variant (
    variant_id BIGSERIAL PRIMARY KEY,
    company_id BIGINT, -- Oracle/Source tablosundan gelen ID. Dis sisteme bagli oldugu icin Foreign Key koymuyoruz.
    original_company_name TEXT,
    variant_name TEXT,
    normalized_variant_name TEXT,
    variant_type TEXT,
    list_version TEXT,
    normalization_version TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT now(),
    
    -- Ayni sirket varyasyonunun tekrar eklenmesini engelleyen Unique Constraint
    UNIQUE (company_id, normalized_variant_name, variant_type, list_version)
);
