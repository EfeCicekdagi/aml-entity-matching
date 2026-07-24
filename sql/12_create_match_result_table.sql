-- 13_create_match_result_table.sql
-- Tüm eşleştirme sonuçlarını saklayan tablo.
-- aml_core.alert yalnızca MEDIUM ve HIGH için kullanılırken,
-- bu tablo tüm kanallardan gelen tüm skorları ve kararları saklar.

-- Önce aml_eval schema oluştur (benchmark için de lazım)
CREATE SCHEMA IF NOT EXISTS aml_eval;

-- Pipeline status enum: teknik retrieval sonucu
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'pipeline_status_type') THEN
        CREATE TYPE aml_core.pipeline_status_type AS ENUM (
            'TRIGRAM_NO_RESULT',
            'FTS_NO_RESULT',
            'VECTOR_NO_RESULT',
            'ALL_RETRIEVAL_CHANNELS_EMPTY',
            'CANDIDATES_FOUND',
            'PROCESSING_ERROR'
        );
    END IF;
END $$;

-- Decision status enum: iş kararı
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'decision_status_type') THEN
        CREATE TYPE aml_core.decision_status_type AS ENUM (
            'NO_CANDIDATE_FOUND',
            'MATCH_BELOW_THRESHOLD',
            'MEDIUM_ALERT',
            'HIGH_ALERT',
            'NO_MATCH',
            'ERROR'
        );
    END IF;
END $$;

-- Ana match_result tablosu
CREATE TABLE IF NOT EXISTS aml_core.match_result (
    match_result_id     BIGSERIAL PRIMARY KEY,
    run_id              TEXT NOT NULL,
    eft_id              BIGINT NOT NULL,
    candidate_company_id BIGINT,
    variant_id          BIGINT,                    -- FK olarak company_variant'a bağlı, ama DROP CASCADE değil
    extracted_entity    TEXT,
    entity_type         TEXT,                      -- PERSON, ORGANIZATION, VESSEL, AIRCRAFT, LOCATION, UNKNOWN
    extraction_method   TEXT,                      -- NER_MODEL, RULE_BASED, CANDIDATE_SUPPORTED, FALLBACK_MATCHED_VARIANT, FULL_TEXT_FALLBACK, ENTITY_NOT_FOUND
    extraction_confidence NUMERIC(5,4),
    entity_extraction_status TEXT,

    -- Retrieval skorları
    trigram_score       NUMERIC(8,5),
    full_text_score     NUMERIC(8,5),
    vector_score        NUMERIC(8,5),
    fuzzy_score         NUMERIC(8,5),             -- Difflib / sequence matcher skoru

    -- Reranker (3 ayrı alan)
    reranker_raw_score          NUMERIC(8,5),
    reranker_normalized_score   NUMERIC(8,5),
    calibrated_probability      NUMERIC(8,5),
    calibration_applied         BOOLEAN DEFAULT FALSE,
    calibration_method          TEXT,              -- PLATT_SCALING, ISOTONIC_REGRESSION
    calibration_version         TEXT,

    -- Final skor
    final_score         NUMERIC(8,5),

    -- Pipeline durumu (teknik)
    pipeline_status     TEXT,                      -- ALL_RETRIEVAL_CHANNELS_EMPTY, CANDIDATES_FOUND, vb.
    no_candidate_reason TEXT,                      -- Neden aday bulunamadı

    -- İş kararı
    decision_status     TEXT NOT NULL,             -- NO_CANDIDATE_FOUND, MATCH_BELOW_THRESHOLD, MEDIUM_ALERT, HIGH_ALERT

    -- Aday sayısı
    candidate_count     INT DEFAULT 0,

    -- Reason codes (açıklanabilir karar)
    reason_codes        JSONB,                     -- ["EXACT_OFFICIAL_NAME", "HIGH_VECTOR_SIMILARITY", ...]
    human_explanation   TEXT,                      -- Analist için okunabilir açıklama

    -- Retrieval kanal bilgisi
    retrieval_sources   JSONB,                     -- {"trigram": 5, "fts": 3, "vector": 8}
    candidate_rank      INT,                       -- Bu adayın birleşik listede sırası

    -- Variant bilgisi
    matched_variant_name TEXT,
    variant_type        TEXT,
    watchlist_company_name TEXT,

    created_at          TIMESTAMP DEFAULT now()
);

-- İndexler
CREATE INDEX IF NOT EXISTS idx_match_result_run_id ON aml_core.match_result(run_id);
CREATE INDEX IF NOT EXISTS idx_match_result_eft_id ON aml_core.match_result(eft_id);
CREATE INDEX IF NOT EXISTS idx_match_result_decision_status ON aml_core.match_result(decision_status);
CREATE INDEX IF NOT EXISTS idx_match_result_final_score ON aml_core.match_result(final_score DESC);
CREATE INDEX IF NOT EXISTS idx_match_result_company ON aml_core.match_result(candidate_company_id);

COMMENT ON TABLE aml_core.match_result IS
    'Tüm eşleştirme sonuçları. HIGH/MEDIUM/NO_MATCH/NO_CANDIDATE_FOUND hepsi burada saklanır. '
    'alert tablosu sadece MEDIUM ve HIGH için kullanılır.';
