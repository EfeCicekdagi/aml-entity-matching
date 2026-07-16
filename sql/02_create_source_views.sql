-- 02_create_source_views.sql

-- 1. Eger tablolar yoksa mock tablolar olustur (Development icin)
CREATE TABLE IF NOT EXISTS bronze_eft_raw (
    eft_id BIGINT PRIMARY KEY,
    transaction_date DATE,
    amount NUMERIC,
    explanation TEXT,
    sender_account_id TEXT,
    receiver_account_id TEXT,
    batch_id TEXT,
    source_system TEXT
);

CREATE TABLE IF NOT EXISTS bronze_blacklist_company_raw (
    company_id BIGINT PRIMARY KEY,
    company_name TEXT,
    source_list TEXT,
    list_version TEXT,
    risk_category TEXT,
    is_active BOOLEAN
);

-- 2. Oracle -> PostgreSQL Adaptasyon View'lari
-- Disaridan (Oracle vb.) gelen NUMBER, VARCHAR2, CLOB veri tipleri
-- burada ::NUMERIC, ::TEXT vb. sekilde PostgreSQL formatina cevrilir (CAST)

CREATE OR REPLACE VIEW aml_source.v_eft_input AS
SELECT 
    eft_id::BIGINT AS eft_id,
    transaction_date::DATE AS transaction_date,
    amount::NUMERIC AS amount,
    sender_account_id::TEXT AS sender_account_id,
    receiver_account_id::TEXT AS receiver_account_id,
    explanation::TEXT AS explanation,
    source_system::TEXT AS source_system,
    batch_id::TEXT AS batch_id
FROM bronze_eft_raw;

CREATE OR REPLACE VIEW aml_source.v_company_input AS
SELECT 
    company_id::BIGINT AS company_id,
    company_name::TEXT AS company_name,
    source_list::TEXT AS source_list,
    list_version::TEXT AS list_version,
    risk_category::TEXT AS risk_category,
    is_active::BOOLEAN AS is_active
FROM bronze_blacklist_company_raw;
