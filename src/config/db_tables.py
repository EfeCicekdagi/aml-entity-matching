"""
db_tables.py — Merkezi tablo adı haritası.
Tüm schema.table referansları buradan yönetilir.
Yeni tablo eklenince sadece bu dosyayı güncellemek yeterli.
"""

TABLES = {
    # Source / Input
    "bronze_eft_raw":     "public.bronze_eft_raw",
    "eft_input":          "aml_source.v_eft_input",
    "company_input":      "aml_source.v_company_input",

    # Stage
    "eft_clean":          "aml_stage.eft_clean",
    "company_variant":    "aml_stage.company_variant",
    "company_detail":     "aml_stage.company_detail",
    "person_detail":      "aml_stage.person_detail",
    "vessel_detail":      "aml_stage.vessel_detail",

    # ML
    "company_embedding":  "aml_ml.company_embedding",
    "reranker_cache":     "aml_ml.reranker_cache",

    # Core — Eşleştirme ve alert sonuçları
    "match_result":       "aml_core.match_result",      # Ana eşleştirme sonuç tablosu (Tüm işlemler)
    "candidate_match":    "aml_core.candidate_match",   # Geriye dönük şema uyumluluğu
    "scoring_result":     "aml_core.scoring_result",    # Geriye dönük şema uyumluluğu
    "alert":              "aml_core.alert",             # Riskli varlıklar tablosu (Sadece HIGH/MEDIUM)
    "alert_export":       "aml_core.alert_export",      # Dışa aktarma ve analitik raporlama tablosu

    # Config
    "scoring_weight":     "aml_config.scoring_weight",
    "threshold":          "aml_config.threshold",
    "model_registry":     "aml_config.model_registry",
    "model_governance":   "aml_eval.model_governance",
    "threshold_validation": "aml_config.threshold_validation",
    "calibration_model":  "aml_config.calibration_model",

    # Audit
    "run_log":            "aml_audit.run_log",
    "quality_check":      "aml_audit.quality_check_result",
    "performance_log":    "aml_audit.performance_log",
    "alert_history":      "aml_audit.alert_status_history",
    "schema_migration":   "aml_audit.schema_migration",

    # Evaluation
    "test_case":          "aml_eval.test_case",
    "benchmark_result":   "aml_eval.benchmark_result",
    "experiment_run":     "aml_experiment.experiment_run",
    "experiment_result":  "aml_experiment.experiment_result",
    "decision_analysis":  "aml_experiment.decision_analysis",
}
