-- 18_create_benchmark_tables.sql
-- Benchmark değerlendirme altyapısı tabloları.

-- Test case tablosu
CREATE TABLE IF NOT EXISTS aml_eval.test_case (
    test_case_id        TEXT PRIMARY KEY,         -- ör. TC_001, TC_TYPO_001
    eft_explanation     TEXT NOT NULL,
    expected_entity     TEXT,
    expected_company_id BIGINT,
    expected_variant_id BIGINT,
    expected_label      TEXT NOT NULL,            -- MATCH, NO_MATCH, ALERT_CREATED
    difficulty_level    TEXT,                     -- EASY, MEDIUM, HARD, EXPERT
    case_type           TEXT,                     -- EXACT_MATCH, TYPO, ABBREVIATION, TRANSLITERATION, vb.
    language            TEXT,                     -- TR, EN, AR, RU, vb.
    script              TEXT,                     -- LATIN, ARABIC, CYRILLIC, vb.
    entity_type         TEXT,                     -- ORGANIZATION, PERSON, VESSEL, AIRCRAFT
    name_length         INT,                      -- Karakter sayısı
    token_count         INT,                      -- Kelime sayısı
    notes               TEXT,
    created_at          TIMESTAMP DEFAULT now(),
    updated_at          TIMESTAMP DEFAULT now()
);

-- Benchmark çalışma sonucu tablosu
CREATE TABLE IF NOT EXISTS aml_eval.benchmark_result (
    benchmark_id        BIGSERIAL PRIMARY KEY,
    test_case_id        TEXT REFERENCES aml_eval.test_case(test_case_id),
    run_id              TEXT,                     -- Production run ile ilişkilendirme (opsiyonel)
    benchmark_run_name  TEXT,                     -- Benchmark çalışmasının adı
    pipeline_version    TEXT,
    threshold_version   TEXT,

    -- Retrieval sonuçları
    retrieved_company_ids   BIGINT[],             -- Retrieval'dan gelen company ID'leri
    retrieved_variant_ids   BIGINT[],
    retrieval_rank          INT,                  -- Doğru sonucun kaçıncı sırada geldiği
    recall_at_1             BOOLEAN,
    recall_at_5             BOOLEAN,
    recall_at_10            BOOLEAN,
    recall_at_20            BOOLEAN,
    recall_at_30            BOOLEAN,
    reciprocal_rank         NUMERIC(8,6),         -- 1/rank, MRR hesabı için

    -- Nihai karar
    predicted_label     TEXT,                     -- MATCH, NO_MATCH, HIGH_ALERT, MEDIUM_ALERT
    predicted_company_id BIGINT,
    predicted_score     NUMERIC(8,5),
    predicted_decision_status TEXT,

    -- Doğruluk
    is_correct          BOOLEAN,
    is_true_positive    BOOLEAN,
    is_false_positive   BOOLEAN,
    is_true_negative    BOOLEAN,
    is_false_negative   BOOLEAN,

    -- Reason codes
    reason_codes        JSONB,

    -- Süre
    processing_time_ms  NUMERIC(10,2),

    created_at          TIMESTAMP DEFAULT now()
);

-- İndexler
CREATE INDEX IF NOT EXISTS idx_test_case_type ON aml_eval.test_case(case_type);
CREATE INDEX IF NOT EXISTS idx_test_case_language ON aml_eval.test_case(language);
CREATE INDEX IF NOT EXISTS idx_benchmark_result_run ON aml_eval.benchmark_result(benchmark_run_name);
CREATE INDEX IF NOT EXISTS idx_benchmark_result_correct ON aml_eval.benchmark_result(is_correct);

COMMENT ON TABLE aml_eval.test_case IS
    'Sistemin performansını değerlendirmek için kullanılan test vakası koleksiyonu.';
COMMENT ON TABLE aml_eval.benchmark_result IS
    'Her test vakası için pipeline çıktıları ve doğruluk değerlendirmesi.';
