INSERT INTO aml_scoring_weight_config (
    config_version,
    fuzzy_weight,
    vector_weight,
    acronym_weight,
    rule_weight,
    reranker_weight,
    is_active
)
VALUES (
    'scoring_v2_reranker',
    0.15,
    0.20,
    0.05,
    0.10,
    0.50,
    true
);

INSERT INTO aml_threshold_config (
    config_version,
    risk_level,
    min_score,
    max_score,
    is_active
)
VALUES
('threshold_v2_reranker', 'HIGH', 0.85, 1.00, true),
('threshold_v2_reranker', 'MEDIUM', 0.70, 0.85, true),
('threshold_v2_reranker', 'LOW', 0.50, 0.70, true),
('threshold_v2_reranker', 'NO_MATCH', 0.00, 0.50, true);
