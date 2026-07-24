import pytest
from src.utils.text_utils import compact_normalize
from src.scoring.score_features import build_score_features

def test_compact_normalize():
    # Pozitif senaryolar
    assert compact_normalize("Indiaforensic Services Pvt Ltd") == "indiaforensicservicespvtltd"
    assert compact_normalize("indiaforensicservicespvtltd") == "indiaforensicservicespvtltd"
    assert compact_normalize("Microsoft Corporation") == "microsoftcorp"
    assert compact_normalize("Apple Inc.") == "appleinc"
    
    # Noktalama ve boşluklar
    assert compact_normalize("oracle-corporation") == "oraclecorp"
    assert compact_normalize("A.B.C. Limited") == "abcltd"
    assert compact_normalize("Test Company") == "testco"

def test_exact_compact_match_in_score_features():
    norm_exp = "settlement related to indiaforensicservicespvtltd contract 562987 odeme tipi ticari"
    raw_exp = "Settlement related to indiaforensicservicespvtltd contract 562987 ödeme tipi: ticari"
    variant = "Indiaforensic Services Pvt Ltd"
    
    cand = {
        "variant_name": variant,
        "trgm_score": 0.6,
        "vector_score": 0.7,
        "normalized_reranker_score": 0.3
    }
    
    scores = build_score_features(norm_exp, cand, extracted_entity="Indiaforensics", raw_explanation=raw_exp)
    
    assert scores["exact_compact_match"] is True
    assert scores["compact_matched_variant"] == "indiaforensicservicespvtltd"
    assert scores["rule_score"] == 1.0

def test_negative_exact_compact_match():
    # Short names shouldn't necessarily fail here (they fail in final_scorer), but let's check substring logic
    norm_exp = "transfer to microsoftsupplier"
    cand = {"variant_name": "Microsoft Corporation"}
    
    scores = build_score_features(norm_exp, cand, raw_explanation=norm_exp)
    assert scores["exact_compact_match"] is False

    # Partial match
    norm_exp = "indiaforensicservice transfer"
    cand = {"variant_name": "Indiaforensic Services Pvt Ltd"}
    scores = build_score_features(norm_exp, cand, raw_explanation=norm_exp)
    assert scores["exact_compact_match"] is False
