-- 19_create_model_governance_table.sql
-- Model registry ve threshold validation tabloları.

-- Model Registry
CREATE TABLE IF NOT EXISTS aml_config.model_registry (
    model_id            BIGSERIAL PRIMARY KEY,
    model_name          TEXT NOT NULL,
    model_version       TEXT NOT NULL,
    model_purpose       TEXT,                     -- EMBEDDING, RERANKER, NER, CALIBRATION
    model_hash          TEXT,                     -- Dosyanın SHA-256 hash'i
    model_path          TEXT,                     -- Yerel dosya yolu veya HuggingFace model ID
    supported_languages TEXT[],
    supported_entity_types TEXT[],
    known_limitations   TEXT,
    validation_metrics  JSONB,                    -- {"precision": 0.91, "recall": 0.87, ...}
    approval_status     TEXT DEFAULT 'PENDING',   -- PENDING, APPROVED, REJECTED, RETIRED
    owner               TEXT,
    approved_by         TEXT,
    approved_at         TIMESTAMP,
    retired_at          TIMESTAMP,
    is_active           BOOLEAN DEFAULT TRUE,
    created_at          TIMESTAMP DEFAULT now(),
    updated_at          TIMESTAMP DEFAULT now(),

    UNIQUE (model_name, model_version)
);

-- Threshold Validation
CREATE TABLE IF NOT EXISTS aml_config.threshold_validation (
    validation_id           BIGSERIAL PRIMARY KEY,
    threshold_version       TEXT NOT NULL,
    validation_dataset_version TEXT,
    high_threshold          NUMERIC(5,4) NOT NULL DEFAULT 0.70,
    medium_threshold        NUMERIC(5,4) NOT NULL DEFAULT 0.62,

    -- Validation sonuçları
    precision_high          NUMERIC(6,4),
    recall_high             NUMERIC(6,4),
    f1_high                 NUMERIC(6,4),
    precision_medium        NUMERIC(6,4),
    recall_medium           NUMERIC(6,4),
    f1_medium               NUMERIC(6,4),
    false_positive_rate     NUMERIC(6,4),
    false_negative_rate     NUMERIC(6,4),
    alert_volume            INT,
    alerts_per_1k           NUMERIC(8,2),

    -- Önerilen değerler (production'a otomatik uygulanmaz)
    recommended_high        NUMERIC(5,4),
    recommended_medium      NUMERIC(5,4),
    recommendation_notes    TEXT,

    -- Onay süreci
    approved_by             TEXT,
    approved_at             TIMESTAMP,
    effective_from          TIMESTAMP,
    is_active               BOOLEAN DEFAULT FALSE,
    validation_report       JSONB,               -- Tam rapor JSON formatında

    created_at              TIMESTAMP DEFAULT now()
);

-- Calibration Model Registry
CREATE TABLE IF NOT EXISTS aml_config.calibration_model (
    calibration_id      BIGSERIAL PRIMARY KEY,
    calibration_version TEXT NOT NULL UNIQUE,
    calibration_method  TEXT NOT NULL,           -- PLATT_SCALING, ISOTONIC_REGRESSION
    base_model_name     TEXT,                    -- Hangi reranker modeli için
    base_model_version  TEXT,
    model_params        JSONB,                   -- Platt parametreleri veya Isotonic eğim noktaları
    training_dataset    TEXT,
    training_sample_count INT,
    validation_metrics  JSONB,
    is_active           BOOLEAN DEFAULT FALSE,
    created_at          TIMESTAMP DEFAULT now()
);

COMMENT ON TABLE aml_config.model_registry IS
    'Kullanılan tüm ML modellerinin kayıt defteri. Hash kontrolü ile model değişikliği tespiti yapılır.';
COMMENT ON TABLE aml_config.threshold_validation IS
    'Threshold doğrulama sonuçları. Önerilen değerler otomatik olarak production config''e uygulanmaz.';
COMMENT ON TABLE aml_config.calibration_model IS
    'Reranker çıktılarını kalibre etmek için kullanılan istatistiksel modeller.';
