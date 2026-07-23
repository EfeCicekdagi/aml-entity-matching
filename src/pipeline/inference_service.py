import time
import logging
from typing import List, Dict, Any, Optional

from src.scoring.score_features import build_score_features
from src.scoring.reason_codes import list_to_codes, build_human_explanation, ReasonCode

logger = logging.getLogger(__name__)

class AMLInferenceService:
    def __init__(
        self,
        config: dict,
        retriever,
        reranker,
        entity_extractor,
        embedding_model,
        scorer,
        calibration=None
    ):
        self.config = config
        self.retriever = retriever
        self.reranker = reranker
        self.entity_extractor = entity_extractor
        self.embedding_model = embedding_model
        self.scorer = scorer
        self.calibration = calibration
        
        self.batch_size = config.get("embedding", {}).get("batch_size", 32)
        self.reranker_prefilter_score = config.get("retrieval", {}).get("reranker_prefilter_score", 0.60)
        self.reranker_top_k = config.get("retrieval", {}).get("reranker_top_k", 10)

    def analyze_batch(self, raw_explanations: List[str], row_ids: List[str], run_id: str, eft_ids: List[int]) -> List[Dict]:
        """
        Processes a batch of EFTs end-to-end.
        Returns a list of result dictionaries per row.
        """
        # Duplicate row_id guard
        row_index: dict[str, int] = {}
        for i, rid in enumerate(row_ids):
            if rid in row_index:
                raise ValueError(f"Duplicate row_id detected: '{rid}'. All row_ids must be unique.")
            row_index[rid] = i

        # 1. Normalization
        norm_exps = [str(exp).casefold() for exp in raw_explanations]

        # 2. Entity Extraction
        extractions = None
        if self.entity_extractor:
            extractions = self.entity_extractor.batch_extract(norm_exps)

        clean_texts = []
        for i, exp in enumerate(norm_exps):
            clean_text = exp
            if extractions and extractions[i] and getattr(extractions[i], "extracted_entity", None):
                clean_text = extractions[i].extracted_entity
            clean_texts.append(clean_text)

        # 3. Embedding
        embeddings = []
        if self.embedding_model and clean_texts:
            embeddings = self.embedding_model.encode(
                clean_texts, batch_size=self.batch_size, show_progress_bar=False
            )

        # 4. Retrieval
        rows_for_batch = [
            {
                "row_id": row_ids[i],
                "normalized_explanation": clean_texts[i],
                "embedding": embeddings[i].tolist() if i < len(embeddings) else None,
            }
            for i in range(len(row_ids))
        ]
        all_retrieval_data = self.retriever.batch_get_candidates(rows_for_batch)

        # 5. Reranking
        reranker_bulk_data = {}
        for rid, retrieval_data in all_retrieval_data.items():
            candidates = retrieval_data.get("candidates", [])
            idx = row_index[rid]
            clean_text = clean_texts[idx]

            strong = [
                c for c in candidates
                if c.get("candidate_score", 0.0) >= self.reranker_prefilter_score
                or self._exact_name_score(clean_text, c.get("variant_name", "")) == 1.0
            ]
            if not strong and candidates:
                strong = candidates

            strong = sorted(strong, key=lambda c: c.get("candidate_score", 0.0), reverse=True)
            strong = strong[:self.reranker_top_k]

            reranker_bulk_data[rid] = {"norm_exp": clean_text, "candidates": strong}

        if self.reranker:
            self.reranker.score_candidates_bulk(reranker_bulk_data)

        # 6. Final Scoring & Decision
        final_results = []
        for i, rid in enumerate(row_ids):
            eft_id = eft_ids[i]
            ret_data = all_retrieval_data.get(rid, {})
            cands = reranker_bulk_data.get(rid, {}).get("candidates", [])
            extraction = extractions[i] if extractions else None
            
            row_result = self._score_row(
                row_id=rid,
                eft_id=eft_id,
                run_id=run_id,
                norm_exp=norm_exps[i],
                clean_text=clean_texts[i],
                extraction=extraction,
                retrieval_data=ret_data,
                candidates=cands
            )
            final_results.append(row_result)

        return final_results

    def _score_row(self, row_id, eft_id, run_id, norm_exp, clean_text, extraction, retrieval_data, candidates):
        result = {
            "row_id": row_id,
            "match_results": [],
            "alerts": [],
            "no_candidate": False,
            "high_count": 0,
            "medium_count": 0,
            "no_match_count": 0,
        }
        
        pipeline_status = retrieval_data.get("pipeline_status", "CANDIDATES_FOUND")
        no_cand_reason = retrieval_data.get("no_candidate_reason")
        channel_counts = retrieval_data.get("channel_counts", {})
        
        ex_entity = getattr(extraction, "extracted_entity", None) if extraction else None
        ex_type = getattr(extraction, "entity_type", "UNKNOWN") if extraction else "UNKNOWN"
        ex_method = getattr(extraction, "extraction_method", "ENTITY_NOT_FOUND") if extraction else "ENTITY_NOT_FOUND"
        ex_conf = getattr(extraction, "extraction_confidence", 0.0) if extraction else 0.0
        ex_status = getattr(extraction, "entity_extraction_status", "NOT_FOUND") if extraction else "NOT_FOUND"

        if not candidates:
            result["no_candidate"] = True
            result["match_results"].append({
                "run_id": run_id,
                "eft_id": eft_id,
                "pipeline_status": pipeline_status,
                "no_candidate_reason": no_cand_reason or "ALL_RETRIEVAL_CHANNELS_EMPTY",
                "decision_status": "NO_CANDIDATE_FOUND",
                "candidate_count": 0,
                "extracted_entity": ex_entity,
                "entity_type": ex_type,
                "extraction_method": ex_method,
                "extraction_confidence": ex_conf,
                "entity_extraction_status": ex_status,
                "reason_codes": [ReasonCode.NO_CANDIDATE_FOUND.value],
                "human_explanation": "Kara liste veya yaptırım listesinde bu metinle eşleşen kayıt bulunamadı.",
                "retrieval_sources": channel_counts,
            })
            return result

        for rank_idx, cand in enumerate(candidates):
            # Fallback: if variant name is contained in text and no entity was extracted, set extraction info
            cand_is_contained = bool(clean_text and cand.get("variant_name") and cand.get("variant_name").casefold() in clean_text)
            current_ex_entity = ex_entity
            current_ex_method = ex_method
            current_ex_status = ex_status
            if not current_ex_entity and cand_is_contained:
                current_ex_entity = cand.get("variant_name")
                current_ex_method = "FALLBACK_MATCHED_VARIANT"
                current_ex_status = "FALLBACK"
            
            scores_dict = build_score_features(clean_text, cand, current_ex_entity)
            
            final_score, match_reason, reason_codes = self.scorer.calculate_final_score(
                scores_dict, alias_confidence=cand.get("alias_confidence", 1.0)
            )
            risk_level = self.scorer.assign_risk_level(final_score)
            decision_status = self.scorer.assign_decision_status(risk_level)
            
            calibrated_prob = None
            calibration_applied = False
            calibration_method = None
            calibration_version = None
            if self.calibration:
                cal_result = self.calibration.calibrate(cand.get("normalized_reranker_score", 0.0))
                calibrated_prob = cal_result.calibrated_probability
                calibration_applied = cal_result.calibration_applied
                calibration_method = cal_result.calibration_method
                calibration_version = cal_result.calibration_version

                if not calibration_applied and ReasonCode.CALIBRATION_NOT_APPLIED.value not in reason_codes:
                    reason_codes.append(ReasonCode.CALIBRATION_NOT_APPLIED.value)
                elif calibration_applied and ReasonCode.CALIBRATION_APPLIED.value not in reason_codes:
                    reason_codes.append(ReasonCode.CALIBRATION_APPLIED.value)

            human_exp = build_human_explanation(
                entity=current_ex_entity,
                matched_name=cand.get("variant_name"),
                codes=list_to_codes(reason_codes),
                final_score=final_score,
                calibrated_probability=calibrated_prob
            )

            match_record = {
                "run_id": run_id,
                "eft_id": eft_id,
                "candidate_company_id": cand.get("company_id"),
                "variant_id": cand.get("variant_id"),
                "extracted_entity": current_ex_entity,
                "entity_type": ex_type,
                "extraction_method": current_ex_method,
                "extraction_confidence": ex_conf,
                "entity_extraction_status": current_ex_status,
                "trigram_score": cand.get("trgm_score", 0.0),
                "full_text_score": cand.get("full_text_score", 0.0),
                "vector_score": cand.get("vector_score", 0.0),
                "fuzzy_score": scores_dict["fuzzy_score"],
                "reranker_raw_score": cand.get("raw_reranker_score", 0.0),
                "reranker_normalized_score": cand.get("normalized_reranker_score", 0.0),
                "calibrated_probability": calibrated_prob,
                "calibration_applied": calibration_applied,
                "calibration_method": calibration_method,
                "calibration_version": calibration_version,
                "final_score": final_score,
                "pipeline_status": pipeline_status,
                "no_candidate_reason": no_cand_reason,
                "decision_status": decision_status,
                "candidate_count": len(candidates),
                "reason_codes": reason_codes,
                "human_explanation": human_exp,
                "retrieval_sources": channel_counts,
                "candidate_rank": rank_idx + 1,
                "matched_variant_name": cand.get("variant_name"),
                "variant_type": cand.get("variant_type"),
                "watchlist_company_name": cand.get("company_name"),
            }
            result["match_results"].append(match_record)
            
            if self.scorer.is_alert_worthy(risk_level):
                alert_record = {
                    **match_record, 
                    "company_id": cand.get("company_id"), 
                    "risk_level": risk_level, 
                    "match_reason": match_reason, 
                    "reranker_score": cand.get("normalized_reranker_score", 0.0)
                }
                result["alerts"].append(alert_record)
                if risk_level == "HIGH": result["high_count"] += 1
                elif risk_level == "MEDIUM": result["medium_count"] += 1
            else:
                result["no_match_count"] += 1

        return result

    def analyze_text(self, text: str, run_id: str = "TEXT_ANALYSIS", eft_id: int = 0) -> Dict:
        res = self.analyze_batch([text], ["text_0"], run_id, [eft_id])
        return res[0] if res else {}

    def _exact_name_score(self, query: str, variant_name: str) -> float:
        return 1.0 if query.strip().casefold() == variant_name.strip().casefold() else 0.0
