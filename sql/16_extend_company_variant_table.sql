-- 16_extend_company_variant_table.sql
-- aml_stage.company_variant tablosuna watchlist lifecycle ve transliteration alanları ekler.

ALTER TABLE aml_stage.company_variant
    -- Watchlist kaynak bilgileri
    ADD COLUMN IF NOT EXISTS source_name          TEXT,
    ADD COLUMN IF NOT EXISTS source_authority     TEXT,        -- Resmi kaynak adı (ör. OFAC, UN, EU)
    ADD COLUMN IF NOT EXISTS source_record_id     TEXT,        -- Kaynak sistemdeki ID
    ADD COLUMN IF NOT EXISTS list_version_tag     TEXT,        -- Kaynak listenin versiyonu
    ADD COLUMN IF NOT EXISTS publication_date     DATE,
    ADD COLUMN IF NOT EXISTS effective_date       DATE,
    ADD COLUMN IF NOT EXISTS ingestion_date       TIMESTAMP DEFAULT now(),
    ADD COLUMN IF NOT EXISTS last_seen_date       TIMESTAMP,
    ADD COLUMN IF NOT EXISTS deactivation_reason  TEXT,
    ADD COLUMN IF NOT EXISTS source_hash          TEXT,        -- Kaynak verinin SHA-256 hash'i
    ADD COLUMN IF NOT EXISTS raw_payload          JSONB,       -- Ham veri (debug için)

    -- Alias güven seviyesi (resmi mi, algoritmik mi?)
    ADD COLUMN IF NOT EXISTS alias_confidence     NUMERIC(4,3) DEFAULT 1.0,
    -- 1.0: Resmi alias, 0.7: Former name, 0.5: Transliteration, 0.3: Algoritmik typo

    -- Transliteration bilgisi
    ADD COLUMN IF NOT EXISTS transliterated_name  TEXT,
    ADD COLUMN IF NOT EXISTS detected_script      TEXT,        -- LATIN, ARABIC, CYRILLIC, HANGUL, vb.
    ADD COLUMN IF NOT EXISTS detected_language    TEXT,

    -- Yeni variant_type değerleri için alan genişletme
    -- (Mevcut TEXT alan, yeni değerlere otomatik uyumlu)
    -- Mevcut: OFFICIAL, ALIAS, ABBREVIATION, TYPO, NORMALIZED, vb.
    -- Yeni eklemeler: OFFICIAL_ALIAS, FORMER_NAME, TRANSLITERATION,
    --                  LEGAL_SUFFIX_VARIANT, GENERATED_TYPO, NORMALIZED_VARIANT
    ADD COLUMN IF NOT EXISTS is_official_alias    BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS is_former_name       BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS parent_company_id    BIGINT;

-- Alias uniqueness için ek index
CREATE INDEX IF NOT EXISTS idx_company_variant_source ON aml_stage.company_variant(source_name, source_record_id);
CREATE INDEX IF NOT EXISTS idx_company_variant_source_hash ON aml_stage.company_variant(source_hash);
CREATE INDEX IF NOT EXISTS idx_company_variant_alias_confidence ON aml_stage.company_variant(alias_confidence);

COMMENT ON COLUMN aml_stage.company_variant.alias_confidence IS
    'Alias güven seviyesi: 1.0=Resmi, 0.7=Eski İsim, 0.5=Transliterasyon, 0.3=Algoritmik Typo';
COMMENT ON COLUMN aml_stage.company_variant.source_hash IS
    'Kaynak verinin SHA-256 hash değeri. Aynı kayıt tekrar yüklenirse deduplication için.';
COMMENT ON COLUMN aml_stage.company_variant.detected_script IS
    'Tespit edilen alfabe: LATIN, ARABIC, CYRILLIC, GEORGIAN, HANGUL, UNKNOWN';
