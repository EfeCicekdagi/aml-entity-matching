"""
db_tables.py — Merkezi tablo adı haritası.
Tüm schema.table referansları buradan yönetilir.
Yeni tablo eklenince sadece bu dosyayı güncellemek yeterli.
"""

TABLES = {
    # Source / Input
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

    # Core — eşleştirme sonuçları
    "match_result":       "aml_core.match_result",      # YENİ: tüm sonuçlar
    "candidate_match":    "aml_core.candidate_match",   # Eski (uyumluluk)
    "scoring_result":     "aml_core.scoring_result",    # Eski (uyumluluk)
    "alert":              "aml_core.alert",             # Sadece HIGH/MEDIUM

    # Config
    "scoring_weight":     "aml_config.scoring_weight",
    "threshold":          "aml_config.threshold",
    "model_registry":     "aml_config.model_registry",          # YENİ
    "threshold_validation": "aml_config.threshold_validation",  # YENİ
    "calibration_model":  "aml_config.calibration_model",       # YENİ

    # Audit
    "run_log":            "aml_audit.run_log",
    "quality_check":      "aml_audit.quality_check_result",
    "performance_log":    "aml_audit.performance_log",
    "alert_history":      "aml_audit.alert_status_history",     # YENİ

    # Evaluation
    "test_case":          "aml_eval.test_case",                 # YENİ
    "benchmark_result":   "aml_eval.benchmark_result",          # YENİ
}
