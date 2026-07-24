-- sql/30_add_compact_match_fields.sql
-- Exact compact match feature alanlarının match_result ve alert tablolarına eklenmesi.

ALTER TABLE aml_core.match_result
ADD COLUMN IF NOT EXISTS exact_compact_match BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS compact_explanation TEXT,
ADD COLUMN IF NOT EXISTS compact_matched_variant TEXT,
ADD COLUMN IF NOT EXISTS rule_score NUMERIC(8,5);

ALTER TABLE aml_core.alert
ADD COLUMN IF NOT EXISTS exact_compact_match BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS compact_explanation TEXT,
ADD COLUMN IF NOT EXISTS compact_matched_variant TEXT,
ADD COLUMN IF NOT EXISTS rule_score NUMERIC(8,5);
