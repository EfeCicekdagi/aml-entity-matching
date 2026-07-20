-- 14_extend_alert_table.sql
-- aml_core.alert tablosuna yeni alanlar ekler.
-- Sadece MEDIUM ve HIGH alertler bu tabloya yazılmaya devam eder.
-- LOW kayıtlar artık sadece aml_core.match_result'ta saklanır.

ALTER TABLE aml_core.alert
    -- Decision & pipeline status ayrımı
    ADD COLUMN IF NOT EXISTS decision_status      TEXT,
    ADD COLUMN IF NOT EXISTS pipeline_status      TEXT,
    ADD COLUMN IF NOT EXISTS no_candidate_reason  TEXT,
    ADD COLUMN IF NOT EXISTS candidate_count      INT DEFAULT 0,

    -- Açıklanabilir karar sistemi
    ADD COLUMN IF NOT EXISTS reason_codes         JSONB,
    ADD COLUMN IF NOT EXISTS human_explanation    TEXT,

    -- Reranker skor ayrımı (3 ayrı alan)
    ADD COLUMN IF NOT EXISTS reranker_raw_score       NUMERIC(8,5),
    ADD COLUMN IF NOT EXISTS reranker_normalized_score NUMERIC(8,5),
    ADD COLUMN IF NOT EXISTS calibrated_probability   NUMERIC(8,5),
    ADD COLUMN IF NOT EXISTS calibration_applied      BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS calibration_method       TEXT,
    ADD COLUMN IF NOT EXISTS calibration_version      TEXT,

    -- Entity extraction metadata
    ADD COLUMN IF NOT EXISTS entity_type          TEXT,
    ADD COLUMN IF NOT EXISTS extraction_method    TEXT,
    ADD COLUMN IF NOT EXISTS extraction_confidence NUMERIC(5,4),

    -- Retrieval kanal bilgisi
    ADD COLUMN IF NOT EXISTS retrieval_sources    JSONB,
    ADD COLUMN IF NOT EXISTS candidate_rank       INT,

    -- Analist genişletilmiş alanları
    ADD COLUMN IF NOT EXISTS analyst_status       TEXT,
    ADD COLUMN IF NOT EXISTS previous_status      TEXT,
    ADD COLUMN IF NOT EXISTS decision_reason      TEXT,
    ADD COLUMN IF NOT EXISTS confidence           NUMERIC(5,4),
    ADD COLUMN IF NOT EXISTS false_positive_category TEXT,
    ADD COLUMN IF NOT EXISTS escalation_reason    TEXT,

    -- match_result ile ilişki (opsiyonel)
    ADD COLUMN IF NOT EXISTS match_result_id      BIGINT;

-- Mevcut HIGH/MEDIUM/LOW kayıtları için decision_status backfill
UPDATE aml_core.alert
SET decision_status = CASE
    WHEN risk_level = 'HIGH'   THEN 'HIGH_ALERT'
    WHEN risk_level = 'MEDIUM' THEN 'MEDIUM_ALERT'
    WHEN risk_level = 'LOW'    THEN 'MATCH_BELOW_THRESHOLD'
    ELSE 'NO_MATCH'
END
WHERE decision_status IS NULL;

COMMENT ON COLUMN aml_core.alert.decision_status IS
    'İş kararı: HIGH_ALERT, MEDIUM_ALERT, MATCH_BELOW_THRESHOLD, NO_MATCH';
COMMENT ON COLUMN aml_core.alert.pipeline_status IS
    'Teknik pipeline durumu: CANDIDATES_FOUND, ALL_RETRIEVAL_CHANNELS_EMPTY, vb.';
COMMENT ON COLUMN aml_core.alert.reason_codes IS
    'Karar gerekçe kodları: ["EXACT_OFFICIAL_NAME", "HIGH_VECTOR_SIMILARITY", ...]';
COMMENT ON COLUMN aml_core.alert.calibrated_probability IS
    'Platt Scaling veya Isotonic Regression ile elde edilen olasılık tahmini';
