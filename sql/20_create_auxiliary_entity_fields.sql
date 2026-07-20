-- 20_create_auxiliary_entity_fields.sql
-- Yardımcı alan tabanlı entity resolution için genişletilebilir altyapı.
-- Şu an tüm alanların dolu olması gerekmiyor; gelecekte kullanılacak.

-- Şirket ek bilgileri (watchlist kaydına bağlı)
CREATE TABLE IF NOT EXISTS aml_stage.company_detail (
    detail_id           BIGSERIAL PRIMARY KEY,
    company_id          BIGINT NOT NULL,
    country             TEXT,
    registration_number TEXT,
    tax_number          TEXT,
    address             TEXT,
    website             TEXT,
    bic_swift           TEXT,
    parent_company_name TEXT,
    parent_company_id   BIGINT,
    incorporation_date  DATE,
    dissolution_date    DATE,
    detail_source       TEXT,
    created_at          TIMESTAMP DEFAULT now(),
    updated_at          TIMESTAMP DEFAULT now()
);

-- Kişi ek bilgileri
CREATE TABLE IF NOT EXISTS aml_stage.person_detail (
    detail_id           BIGSERIAL PRIMARY KEY,
    company_id          BIGINT NOT NULL,          -- aml_stage.company_variant.company_id (person olduğunda)
    date_of_birth       DATE,
    place_of_birth      TEXT,
    nationality         TEXT,
    citizenship         TEXT,
    passport_number     TEXT,
    national_id         TEXT,
    address             TEXT,
    gender              TEXT,
    detail_source       TEXT,
    created_at          TIMESTAMP DEFAULT now()
);

-- Gemi/Uçak bilgileri
CREATE TABLE IF NOT EXISTS aml_stage.vessel_detail (
    detail_id           BIGSERIAL PRIMARY KEY,
    company_id          BIGINT NOT NULL,
    entity_type         TEXT NOT NULL,            -- VESSEL, AIRCRAFT
    imo_number          TEXT,                     -- Gemi için
    mmsi                TEXT,                     -- Gemi için
    call_sign           TEXT,
    flag                TEXT,
    aircraft_registration TEXT,                   -- Uçak için
    build_year          INT,
    gross_tonnage       NUMERIC(12,2),
    detail_source       TEXT,
    created_at          TIMESTAMP DEFAULT now()
);

-- Alan bazlı skor alanları için match_result genişletme
ALTER TABLE aml_core.match_result
    ADD COLUMN IF NOT EXISTS name_score           NUMERIC(8,5),
    ADD COLUMN IF NOT EXISTS country_score        NUMERIC(8,5),
    ADD COLUMN IF NOT EXISTS identifier_score     NUMERIC(8,5),
    ADD COLUMN IF NOT EXISTS address_score        NUMERIC(8,5),
    ADD COLUMN IF NOT EXISTS date_of_birth_score  NUMERIC(8,5),
    ADD COLUMN IF NOT EXISTS entity_type_score    NUMERIC(8,5),
    ADD COLUMN IF NOT EXISTS auxiliary_field_reason_codes JSONB;
-- Örnek: {"COUNTRY_CONFLICT": true, "IDENTIFIER_EXACT_MATCH": true}

CREATE INDEX IF NOT EXISTS idx_company_detail_company ON aml_stage.company_detail(company_id);
CREATE INDEX IF NOT EXISTS idx_person_detail_company ON aml_stage.person_detail(company_id);
CREATE INDEX IF NOT EXISTS idx_vessel_detail_imo ON aml_stage.vessel_detail(imo_number);

COMMENT ON TABLE aml_stage.company_detail IS
    'Şirket yardımcı alanları. İleride multi-field entity resolution için kullanılacak.';
COMMENT ON TABLE aml_stage.person_detail IS
    'Kişi yardımcı alanları (DOB, passport, nationality). İleride isim+alan skorlaması için.';
COMMENT ON TABLE aml_stage.vessel_detail IS
    'Gemi ve uçak yardımcı alanları (IMO, MMSI, call sign, flag).';
