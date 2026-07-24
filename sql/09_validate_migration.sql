-- 10_validate_migration.sql

-- Bu script yeni semalardaki kayit sayisi ile eski tablolardaki kayit sayisini karsilastirir.
-- Sonuclarin birebir ayni olmasi (veya yeni tabloda daha fazla olmasi) beklenir.

SELECT 'aml_stage.eft_clean' AS table_name, COUNT(*) AS new_count, (SELECT COUNT(*) FROM silver_eft_clean) AS old_count FROM aml_stage.eft_clean
UNION ALL
SELECT 'aml_stage.company_variant', COUNT(*), (SELECT COUNT(*) FROM silver_company_variant) FROM aml_stage.company_variant
UNION ALL
SELECT 'aml_ml.company_embedding', COUNT(*), (SELECT COUNT(*) FROM gold_company_embedding) FROM aml_ml.company_embedding
UNION ALL
SELECT 'aml_ml.reranker_cache', COUNT(*), (SELECT COUNT(*) FROM aml_reranker_cache) FROM aml_ml.reranker_cache
UNION ALL
SELECT 'aml_core.alert', COUNT(*), (SELECT COUNT(*) FROM aml_alert) FROM aml_core.alert
UNION ALL
SELECT 'aml_core.scoring_result', COUNT(*), (SELECT COUNT(*) FROM aml_scoring_result) FROM aml_core.scoring_result
UNION ALL
SELECT 'aml_core.candidate_match', COUNT(*), (SELECT COUNT(*) FROM aml_candidate_match) FROM aml_core.candidate_match
UNION ALL
SELECT 'aml_audit.run_log', COUNT(*), (SELECT COUNT(*) FROM aml_run_log) FROM aml_audit.run_log;
