-- 11_add_match_reason_to_alerts.sql

ALTER TABLE aml_core.alert ADD COLUMN IF NOT EXISTS match_reason TEXT;
ALTER TABLE aml_core.scoring_result ADD COLUMN IF NOT EXISTS match_reason TEXT;
