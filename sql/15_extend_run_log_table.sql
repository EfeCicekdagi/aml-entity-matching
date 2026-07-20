-- 15_extend_run_log_table.sql
-- aml_audit.run_log tablosuna model governance ve operasyonel metrik alanları ekler.

ALTER TABLE aml_audit.run_log
    -- Model governance
    ADD COLUMN IF NOT EXISTS git_commit_hash          TEXT,
    ADD COLUMN IF NOT EXISTS embedding_model_name     TEXT,
    ADD COLUMN IF NOT EXISTS embedding_model_hash     TEXT,
    ADD COLUMN IF NOT EXISTS reranker_model_name      TEXT,
    ADD COLUMN IF NOT EXISTS reranker_model_hash      TEXT,
    ADD COLUMN IF NOT EXISTS ner_model_name           TEXT,
    ADD COLUMN IF NOT EXISTS ner_model_version        TEXT,
    ADD COLUMN IF NOT EXISTS ner_model_hash           TEXT,
    ADD COLUMN IF NOT EXISTS calibration_version      TEXT,
    ADD COLUMN IF NOT EXISTS normalization_version    TEXT,
    ADD COLUMN IF NOT EXISTS watchlist_version        TEXT,

    -- Sayım alanları
    ADD COLUMN IF NOT EXISTS input_count              INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS match_result_count       INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS no_candidate_count       INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS high_alert_count         INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS medium_alert_count       INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS no_match_count           INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS error_count              INT DEFAULT 0,
    ADD COLUMN IF NOT EXISTS prescreen_skipped_count  INT DEFAULT 0,

    -- Operasyonel süreler (saniye)
    ADD COLUMN IF NOT EXISTS total_duration_s         NUMERIC(12,2),
    ADD COLUMN IF NOT EXISTS ner_duration_s           NUMERIC(12,2),
    ADD COLUMN IF NOT EXISTS retrieval_duration_s     NUMERIC(12,2),
    ADD COLUMN IF NOT EXISTS reranker_duration_s      NUMERIC(12,2),
    ADD COLUMN IF NOT EXISTS scoring_duration_s       NUMERIC(12,2),

    -- Latency percentile (ms)
    ADD COLUMN IF NOT EXISTS p50_latency_ms           NUMERIC(10,2),
    ADD COLUMN IF NOT EXISTS p95_latency_ms           NUMERIC(10,2),
    ADD COLUMN IF NOT EXISTS p99_latency_ms           NUMERIC(10,2),

    -- Throughput
    ADD COLUMN IF NOT EXISTS rows_per_second          NUMERIC(10,2),
    ADD COLUMN IF NOT EXISTS avg_candidate_per_row    NUMERIC(8,2),

    -- Watchlist kullanılan versiyon (kesin kayıt)
    ADD COLUMN IF NOT EXISTS watchlist_snapshot_at    TIMESTAMP,

    -- finished_at -> completed_at alias (geriye uyumluluk için yeni kolon, eski tutuluyor)
    ADD COLUMN IF NOT EXISTS completed_at             TIMESTAMP;

-- Mevcut finished_at değerlerini completed_at'a kopyala
UPDATE aml_audit.run_log
SET completed_at = finished_at
WHERE completed_at IS NULL AND finished_at IS NOT NULL;

COMMENT ON TABLE aml_audit.run_log IS
    'Pipeline çalışma kayıtları. Model hash, watchlist version ve latency percentile bilgileri dahil.';
COMMENT ON COLUMN aml_audit.run_log.embedding_model_hash IS
    'Embedding model dosyasının SHA-256 hash değeri. Model dosyası değişti mi kontrolü için.';
COMMENT ON COLUMN aml_audit.run_log.watchlist_version IS
    'Bu run sırasında kullanılan watchlist versiyonu. Önceki versiyonla yeniden çalışma için.';
