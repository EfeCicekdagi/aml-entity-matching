#!/usr/bin/env python3
"""
[*] AML Pipeline Adim Adim (Step-by-Step) Debug ve Deger Izleme Araci [*]

Bu betik, bir EFT transfer aciklamasinin ve aday sirket isminin AML motorundan gecerken
6 katmanda (Breakpoint / Debug Kesme Noktalari) bellek degiskenlerinde hangi degerleri
aldigini, harflerin nasil donustugunu ve hibrit formulun skoru nasil hesapladigini
gorsel olarak denetlemek icin tasarlanmistir.

Kullanim:
    1. Varsayilan 4 Kritik AML Senaryosunu Test Etmek Icin:
       python scripts/debug_pipeline_step_by_step.py --demo

    2. Ozel Bir Transfer ve Sirket Ikisilisini Debug Etmek Icin:
       python scripts/debug_pipeline_step_by_step.py \
           --explanation "Odeme M!cr0s0ft C0rp0r4t!0n lisans bedeli" \
           --candidate "Microsoft Corporation" \
           --vector 0.88 \
           --reranker 0.94
"""

import sys
import os
import argparse
import difflib
from typing import Dict, Any

# Python 3.6+ Windows konsolunda Unicode'u otomatik destekler.
# Manuel UTF-8 zorlamasi PowerShell'de karakterlerin bozuk cikmasina sebep oluyordu, bu yuzden kaldirildi.
# Proje dizinini yola ekle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.text_utils import (
    normalize_text,
    remove_company_suffixes,
    get_normalized_core_name,
    get_compact_core_name,
    check_leetspeak_evasion,
    normalize_leetspeak
)
from src.scoring.score_features import build_score_features
from src.scoring.final_scorer import FinalScorer
from src.scoring.reason_codes import list_to_codes, build_human_explanation, ReasonCode


class MockRepository:
    """Veritabanina baglanmadan bagimsiz debug yapabilmek icin tasarlanmis taslak depo."""
    def get_connection(self):
        return None
    def release_connection(self, conn):
        pass


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" [*] {title}")
    print("=" * 80)


def print_breakpoint(bp_num: int, bp_title: str, variables: Dict[str, Any]):
    print(f"\n   [BP {bp_num}] {bp_title.upper()}")
    print("   " + "-" * 72)
    for var_name, var_value in variables.items():
        if isinstance(var_value, float):
            val_str = f"{var_value:.4f}"
        elif isinstance(var_value, list):
            val_str = ", ".join([str(v) for v in var_value]) if var_value else "(bos)"
        else:
            val_str = str(var_value)
        print(f"     • {var_name:<30} : {val_str}")
    print("   " + "-" * 72)


def run_pipeline_step_by_step(
    explanation: str,
    candidate_name: str,
    simulated_vector_score: float = 0.85,
    simulated_reranker_score: float = 0.90,
    simulated_entity: str = None
):
    """Bir transfer aciklamasini 6 kesme noktasindan gecirerek bellek degisimini yazdirir."""
    print_header(f"TRANSFER ANALIZI: '{explanation}' <---> '{candidate_name}'")
    
    # ── BREAKPOINT 0: GIRDILER VE PARAMETRELER ─────────────────────────────────
    print_breakpoint(0, "Ham Girdiler ve Baslangic Durumu", {
        "raw_explanation": explanation,
        "candidate_name_in_db": candidate_name,
        "simulated_vector_score (MPNet)": simulated_vector_score,
        "simulated_reranker_score (BGE-M3)": simulated_reranker_score
    })

    # ── BREAKPOINT 1: NORMALIZASYON VE TEMIZLEME ───────────────────────────────
    norm_exp = normalize_text(explanation)
    norm_cand = normalize_text(candidate_name)
    core_cand = remove_company_suffixes(norm_cand)
    compact_cand = get_compact_core_name(norm_cand)
    
    print_breakpoint(1, "Metin Temizleme ve Cekirdek Ayristirma (Pre-processing)", {
        "norm_exp (Kucuk Harf & Noktalamasiz)": norm_exp,
        "norm_cand (Normalize Sirket Adi)": norm_cand,
        "core_cand (Yapi Ekleri Temizlenmis)": core_cand,
        "compact_cand (Bosluksuz Cekirdek)": compact_cand
    })

    # ── BREAKPOINT 2: SIBER ATLATMA (LEETSPEAK EVASION) DENETIMI ───────────────
    leet_evasion, leet_sim, leet_exp_norm = check_leetspeak_evasion(explanation, candidate_name)
    print_breakpoint(2, "Siber Atlatma ve Karakter Gizleme Kontrolu (Evasion Check)", {
        "leetspeak_evasion_detected": leet_evasion,
        "normalized_leetspeak_text": leet_exp_norm if leet_evasion else "(Manipulasyon Yok - Normal Metin)",
        "leetspeak_similarity_boost": leet_sim if leet_evasion else 0.0
    })

    # ── BREAKPOINT 3: VARLIK CIKARIMI (NER & FALLBACK) ─────────────────────────
    # Eger ozel simule edilmis varlik verilmediyse akilli fallback calisir
    extracted_entity = simulated_entity
    extraction_method = "BERT_NER_MODEL"
    if not extracted_entity:
        if candidate_name.casefold() in norm_exp:
            extracted_entity = candidate_name
            extraction_method = "FALLBACK_MATCHED_VARIANT"
        elif core_cand and core_cand in norm_exp:
            extracted_entity = core_cand
            extraction_method = "FALLBACK_CORE_MATCH"
        elif leet_evasion:
            extracted_entity = core_cand or candidate_name
            extraction_method = "LEETSPEAK_EVASION_FALLBACK"
        else:
            extracted_entity = explanation
            extraction_method = "FULL_TEXT_FALLBACK"

    print_breakpoint(3, "Varlik Cikarimi ve Alt-Dizi Esleme (NER Extraction)", {
        "extracted_entity": extracted_entity,
        "extraction_method": extraction_method,
        "query_token_count": len(norm_exp.split())
    })

    # ── BREAKPOINT 4: OZELLIK MIMARISI VE SKOR BILESENLERI (scores_dict) ───────
    cand_dict = {
        "variant_name": candidate_name,
        "normalized_variant_name": norm_cand,
        "candidate_score": simulated_vector_score,
        "normalized_reranker_score": simulated_reranker_score,
        "alias_confidence": 1.0
    }
    
    scores_dict = build_score_features(
        norm_exp=extracted_entity,
        cand=cand_dict,
        extracted_entity=extracted_entity,
        raw_explanation=explanation
    )
    
    # Reranker skoru dict'e yerlestirilir
    scores_dict["reranker_score"] = simulated_reranker_score
    scores_dict["vector_score"] = simulated_vector_score

    print_breakpoint(4, "Bellek Ozellik Sozlugu (scores_dict Inspection)", {
        "vector_score (Agirlik: %20)": scores_dict.get("vector_score", 0.0),
        "reranker_score (Agirlik: %40)": scores_dict.get("reranker_score", 0.0),
        "fuzzy_score (Agirlik: %40)": scores_dict.get("fuzzy_score", 0.0),
        "acronym_score": scores_dict.get("acronym_score", 0.0),
        "rule_score": scores_dict.get("rule_score", 0.0),
        "exact_normalized_match": scores_dict.get("exact_normalized_match", False),
        "exact_core_match": scores_dict.get("exact_core_match", False),
        "leetspeak_evasion_detected": scores_dict.get("leetspeak_evasion_detected", False),
        "substantial_missing_info": scores_dict.get("substantial_missing_info", False)
    })

    # ── BREAKPOINT 5: HIBRIT ENSEMBLE HESAPLAYICI VE NIHAI KARAR ───────────────
    scorer = FinalScorer(MockRepository())
    final_score, match_reason, reason_codes_str = scorer.calculate_final_score(scores_dict)
    risk_level = scorer.assign_risk_level(final_score)
    decision_status = scorer.assign_decision_status(risk_level)
    
    reason_codes_obj = list_to_codes(reason_codes_str)
    human_exp = build_human_explanation(
        entity=extracted_entity,
        matched_name=candidate_name,
        codes=reason_codes_obj,
        final_score=final_score
    )

    # Dogrusal formulun ciplak sonucunu hesaplayalim (kiyaslama icin)
    w_vec = scorer.weights.get("vector_weight", 0.20)
    w_rer = scorer.weights.get("reranker_weight", 0.40)
    w_fuz = scorer.weights.get("fuzzy_weight", 0.40)
    raw_linear = (w_vec * simulated_vector_score) + (w_rer * simulated_reranker_score) + (w_fuz * scores_dict.get("fuzzy_score", 0.0))

    print_breakpoint(5, "Nihai Skor Degerlendirmesi ve Kural Yaptirimi (Ensemble Engine)", {
        "Ham Dogrusal Formul Sonucu": raw_linear,
        "Uygulanan Kesin Kural / Override": match_reason,
        "NIHAI RISK SKORU (final_score)": final_score,
        "ATANAN RISK SEVIYESI": risk_level,
        "OPERASYONEL KARAR": decision_status,
        "Tetiklenen Neden Kodlari (Reason Codes)": reason_codes_str,
        "Turkce Insani Aciklama": human_exp
    })
    print("\n" + "=" * 80 + "\n")


def run_demo_scenarios():
    print_header("CRITICAL AML SENARYOLARI BIRLESIK DEBUG VE TEST KUMESI")
    print("Bu test kumesi, yoneticiler ve denetciler icin 4 zorlu bankacilik senaryosunu adim adim test eder.")
    
    # 1. Senaryo: Leetspeak Evasion (Siber Saldiri Gizlemesi)
    run_pipeline_step_by_step(
        explanation="Odeme M!cr0s0ft C0rp0r4t!0n lisans bedeli",
        candidate_name="Microsoft Corporation",
        simulated_vector_score=0.72,
        simulated_reranker_score=0.88
    )

    # 2. Senaryo: Typo & Harf Eksikligi (Yazim Hatasi)
    run_pipeline_step_by_step(
        explanation="Indaforensic Services Pvt Ltd sozlesme odemesi",
        candidate_name="Indiaforensic Services Pvt Ltd",
        simulated_vector_score=0.89,
        simulated_reranker_score=0.95
    )

    # 3. Senaryo: Kisaltma (Acronym Match)
    run_pipeline_step_by_step(
        explanation="Transfer to IB M sistem hizmetleri faturasi",
        candidate_name="International Business Machines",
        simulated_vector_score=0.82,
        simulated_reranker_score=0.91
    )

    # 4. Senaryo: Kismi Bilgi ve Eksik Varlik (Analyst Review / Medium Risk)
    run_pipeline_step_by_step(
        explanation="Global Sanayi ve Ticaret odemesi",
        candidate_name="Global Sanayi Ticaret A.S. Istanbul Subesi",
        simulated_vector_score=0.68,
        simulated_reranker_score=0.62
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Pipeline Step-by-Step Debug Tool")
    parser.add_argument("--demo", action="store_true", help="4 Kritik AML senaryosunu adim adim test eder.")
    parser.add_argument("--explanation", type=str, help="Test edilecek EFT aciklamasi.")
    parser.add_argument("--candidate", type=str, help="Veritabanindaki aday sirket ismi.")
    parser.add_argument("--vector", type=float, default=0.85, help="Simule edilmis MPNet vektor skoru (0.0-1.0).")
    parser.add_argument("--reranker", type=float, default=0.90, help="Simule edilmis BGE-M3 reranker skoru (0.0-1.0).")

    args = parser.parse_args()

    if args.demo or not args.explanation:
        run_demo_scenarios()
    else:
        run_pipeline_step_by_step(
            explanation=args.explanation,
            candidate_name=args.candidate or "Test Company Inc",
            simulated_vector_score=args.vector,
            simulated_reranker_score=args.reranker
        )
