#!/usr/bin/env python3
"""
[*] AML Pipeline Adım Adım (Step-by-Step) Debug ve Değer İzleme Aracı [*]

Bu betik, bir EFT transfer açıklamasının ve aday şirket isminin AML motorundan geçerken
6 katmanda (Breakpoint / Debug Kesme Noktaları) bellek değişkenlerinde hangi değerleri
aldığını, harflerin nasıl dönüştüğünü ve hibrit formülün skoru nasıl hesapladığını
görsel olarak denetlemek için tasarlanmıştır.

Kullanım:
    1. Varsayılan 4 Kritik AML Senaryosunu Test Etmek İçin:
       python scripts/debug_pipeline_step_by_step.py --demo

    2. Özel Bir Transfer ve Şirket İkişilisini Debug Etmek İçin:
       python scripts/debug_pipeline_step_by_step.py \
           --explanation "Ödeme M!cr0s0ft C0rp0r4t!0n lisans bedeli" \
           --candidate "Microsoft Corporation" \
           --vector 0.88 \
           --reranker 0.94
"""

import sys
import os
import argparse
import difflib
from typing import Dict, Any

# Windows konsolunda karakter kodlama hatalarını engelle
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

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
    """Veritabanına bağlanmadan bağımsız debug yapabilmek için tasarlanmış taslak depo."""
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
            val_str = ", ".join([str(v) for v in var_value]) if var_value else "(boş)"
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
    """Bir transfer açıklamasını 6 kesme noktasından geçirerek bellek değişimini yazdırır."""
    print_header(f"TRANSFER ANALİZİ: '{explanation}' <---> '{candidate_name}'")
    
    # ── BREAKPOINT 0: GİRDİLER VE PARAMETRELER ─────────────────────────────────
    print_breakpoint(0, "Ham Girdiler ve Başlangıç Durumu", {
        "raw_explanation": explanation,
        "candidate_name_in_db": candidate_name,
        "simulated_vector_score (MPNet)": simulated_vector_score,
        "simulated_reranker_score (BGE-M3)": simulated_reranker_score
    })

    # ── BREAKPOINT 1: NORMALİZASYON VE TEMİZLEME ───────────────────────────────
    norm_exp = normalize_text(explanation)
    norm_cand = normalize_text(candidate_name)
    core_cand = remove_company_suffixes(norm_cand)
    compact_cand = get_compact_core_name(norm_cand)
    
    print_breakpoint(1, "Metin Temizleme ve Çekirdek Ayrıştırma (Pre-processing)", {
        "norm_exp (Küçük Harf & Noktalamasız)": norm_exp,
        "norm_cand (Normalize Şirket Adı)": norm_cand,
        "core_cand (Yapı Ekleri Temizlenmiş)": core_cand,
        "compact_cand (Boşluksuz Çekirdek)": compact_cand
    })

    # ── BREAKPOINT 2: SİBER ATLATMA (LEETSPEAK EVASION) DENETİMİ ───────────────
    leet_evasion, leet_sim, leet_exp_norm = check_leetspeak_evasion(explanation, candidate_name)
    print_breakpoint(2, "Siber Atlatma ve Karakter Gizleme Kontrolü (Evasion Check)", {
        "leetspeak_evasion_detected": leet_evasion,
        "normalized_leetspeak_text": leet_exp_norm if leet_evasion else "(Manipülasyon Yok - Normal Metin)",
        "leetspeak_similarity_boost": leet_sim if leet_evasion else 0.0
    })

    # ── BREAKPOINT 3: VARLIK ÇIKARIMI (NER & FALLBACK) ─────────────────────────
    # Eğer özel simüle edilmiş varlık verilmediyse akıllı fallback çalışır
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

    print_breakpoint(3, "Varlık Çıkarımı ve Alt-Dizi Eşleme (NER Extraction)", {
        "extracted_entity": extracted_entity,
        "extraction_method": extraction_method,
        "query_token_count": len(norm_exp.split())
    })

    # ── BREAKPOINT 4: ÖZELLİK MİMARİSİ VE SKOR BİLEŞENLERİ (scores_dict) ───────
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
    
    # Reranker skoru dict'e yerleştirilir
    scores_dict["reranker_score"] = simulated_reranker_score
    scores_dict["vector_score"] = simulated_vector_score

    print_breakpoint(4, "Bellek Özellik Sözlüğü (scores_dict Inspection)", {
        "vector_score (Ağırlık: %50)": scores_dict.get("vector_score", 0.0),
        "reranker_score (Ağırlık: %40)": scores_dict.get("reranker_score", 0.0),
        "fuzzy_score (Ağırlık: %10)": scores_dict.get("fuzzy_score", 0.0),
        "acronym_score": scores_dict.get("acronym_score", 0.0),
        "rule_score": scores_dict.get("rule_score", 0.0),
        "exact_normalized_match": scores_dict.get("exact_normalized_match", False),
        "exact_core_match": scores_dict.get("exact_core_match", False),
        "leetspeak_evasion_detected": scores_dict.get("leetspeak_evasion_detected", False),
        "substantial_missing_info": scores_dict.get("substantial_missing_info", False)
    })

    # ── BREAKPOINT 5: HİBRİT ENSEMBLE HESAPLAYICI VE NİHAİ KARAR ───────────────
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

    # Doğrusal formülün çıplak sonucunu hesaplayalım (kıyaslama için)
    w_vec = scorer.weights.get("vector_weight", 0.50)
    w_rer = scorer.weights.get("reranker_weight", 0.40)
    w_fuz = scorer.weights.get("fuzzy_weight", 0.10)
    raw_linear = (w_vec * simulated_vector_score) + (w_rer * simulated_reranker_score) + (w_fuz * scores_dict.get("fuzzy_score", 0.0))

    print_breakpoint(5, "Nihai Skor Değerlendirmesi ve Kural Yaptırımı (Ensemble Engine)", {
        "Ham Doğrusal Formül Sonucu": raw_linear,
        "Uygulanan Kesin Kural / Override": match_reason,
        "NİHAİ RİSK SKORU (final_score)": final_score,
        "ATANAN RİSK SEVİYESİ": risk_level,
        "OPERASYONEL KARAR": decision_status,
        "Tetiklenen Neden Kodları (Reason Codes)": reason_codes_str,
        "Türkçe İnsani Açıklama": human_exp
    })
    print("\n" + "=" * 80 + "\n")


def run_demo_scenarios():
    print_header("CRITICAL AML SENARYOLARI BİRLEŞİK DEBUG VE TEST KÜMESİ")
    print("Bu test kümesi, yöneticiler ve denetçiler için 4 zorlu bankacılık senaryosunu adım adım test eder.")
    
    # 1. Senaryo: Leetspeak Evasion (Siber Saldırı Gizlemesi)
    run_pipeline_step_by_step(
        explanation="Ödeme M!cr0s0ft C0rp0r4t!0n lisans bedeli",
        candidate_name="Microsoft Corporation",
        simulated_vector_score=0.72,
        simulated_reranker_score=0.88
    )

    # 2. Senaryo: Typo & Harf Eksikliği (Yazım Hatası)
    run_pipeline_step_by_step(
        explanation="Indaforensic Services Pvt Ltd sözleşme ödemesi",
        candidate_name="Indiaforensic Services Pvt Ltd",
        simulated_vector_score=0.89,
        simulated_reranker_score=0.95
    )

    # 3. Senaryo: Kısaltma (Acronym Match)
    run_pipeline_step_by_step(
        explanation="Transfer to IB M sistem hizmetleri faturası",
        candidate_name="International Business Machines",
        simulated_vector_score=0.82,
        simulated_reranker_score=0.91
    )

    # 4. Senaryo: Kısmi Bilgi ve Eksik Varlık (Analyst Review / Medium Risk)
    run_pipeline_step_by_step(
        explanation="Global Sanayi ve Ticaret ödemesi",
        candidate_name="Global Sanayi Ticaret A.Ş. İstanbul Şubesi",
        simulated_vector_score=0.68,
        simulated_reranker_score=0.62
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML Pipeline Step-by-Step Debug Tool")
    parser.add_argument("--demo", action="store_true", help="4 Kritik AML senaryosunu adım adım test eder.")
    parser.add_argument("--explanation", type=str, help="Test edilecek EFT açıklaması.")
    parser.add_argument("--candidate", type=str, help="Veritabanındaki aday şirket ismi.")
    parser.add_argument("--vector", type=float, default=0.85, help="Simüle edilmiş MPNet vektör skoru (0.0-1.0).")
    parser.add_argument("--reranker", type=float, default=0.90, help="Simüle edilmiş BGE-M3 reranker skoru (0.0-1.0).")

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
