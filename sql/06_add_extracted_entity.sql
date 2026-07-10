-- Adds extracted_entity column to aml_alert to store NER results
ALTER TABLE aml_alert ADD COLUMN IF NOT EXISTS extracted_entity TEXT;
