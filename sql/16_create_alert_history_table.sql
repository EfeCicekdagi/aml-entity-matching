-- 17_create_alert_history_table.sql
-- Analist kararlarının append-only geçmişi.
-- Her durum değişikliği bu tabloya yazılır, hiçbir şey silinmez/güncellenmez.

CREATE TABLE IF NOT EXISTS aml_audit.alert_status_history (
    history_id          BIGSERIAL PRIMARY KEY,
    alert_id            BIGINT NOT NULL,
    run_id              TEXT,

    -- Kim/ne zaman
    reviewed_by         TEXT NOT NULL,
    reviewed_at         TIMESTAMP DEFAULT now(),

    -- Durum geçişi
    previous_status     TEXT,
    new_status          TEXT NOT NULL,        -- OPEN, IN_REVIEW, CONFIRMED_MATCH, FALSE_POSITIVE, INSUFFICIENT_INFORMATION, ESCALATED, CLOSED
    analyst_status      TEXT,                -- Analist terminolojisi ile karar

    -- Karar detayları
    analyst_note        TEXT,
    decision_reason     TEXT,
    confidence          NUMERIC(4,3),         -- Analistin kararına güveni: 0.0 - 1.0
    false_positive_category TEXT,             -- Eğer FP ise neden: NAME_ONLY_MATCH, DIFFERENT_ENTITY_TYPE, vb.
    escalation_reason   TEXT,                 -- Eğer ESCALATED ise sebep

    -- Eğitim verisi için kullanılabilecek sonuç
    -- CONFIRMED_MATCH, FALSE_POSITIVE, INSUFFICIENT_INFORMATION, ESCALATED, CLOSED
    final_analyst_label TEXT,

    created_at          TIMESTAMP DEFAULT now()
);

-- İndexler
CREATE INDEX IF NOT EXISTS idx_alert_history_alert_id ON aml_audit.alert_status_history(alert_id);
CREATE INDEX IF NOT EXISTS idx_alert_history_reviewed_by ON aml_audit.alert_status_history(reviewed_by);
CREATE INDEX IF NOT EXISTS idx_alert_history_new_status ON aml_audit.alert_status_history(new_status);
CREATE INDEX IF NOT EXISTS idx_alert_history_created_at ON aml_audit.alert_status_history(created_at DESC);

COMMENT ON TABLE aml_audit.alert_status_history IS
    'Analist kararlarının append-only audit tablosu. Hiçbir kayıt silinmez veya güncellenmez. '
    'Benchmark ve model iyileştirme süreçlerinde eğitim verisi olarak kullanılabilir.';
COMMENT ON COLUMN aml_audit.alert_status_history.final_analyst_label IS
    'Model eğitimi için kullanılacak etiket: CONFIRMED_MATCH, FALSE_POSITIVE, INSUFFICIENT_INFORMATION, ESCALATED, CLOSED';
