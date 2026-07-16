TABLES = {
    "eft_input": "aml_source.v_eft_input",
    "company_input": "aml_source.v_company_input",
    "eft_clean": "aml_stage.eft_clean",
    "company_variant": "aml_stage.company_variant",
    "company_embedding": "aml_ml.company_embedding",
    "reranker_cache": "aml_ml.reranker_cache",
    "candidate_match": "aml_core.candidate_match",
    "scoring_result": "aml_core.scoring_result",
    "alert": "aml_core.alert",
    "scoring_weight": "aml_config.scoring_weight",
    "threshold": "aml_config.threshold",
    "run_log": "aml_audit.run_log",
    "quality_check": "aml_audit.quality_check_result",
    "performance_log": "aml_audit.performance_log"
}
