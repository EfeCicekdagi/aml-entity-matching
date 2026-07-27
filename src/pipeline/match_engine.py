import logging
import time
from typing import List, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

class MatchEngine:
    """
    Unified engine for handling the AML entity matching pipeline.
    It encapsulates:
      - Text Normalization
      - NER / Entity Extraction
      - Vector Embedding Generation
      - Candidate Retrieval (Trigram, Full-Text, Vector)
      - Reranking
      
    This engine is stateless with respect to the database updates. It takes raw text inputs
    and returns retrieval and reranker candidates.
    Final fuzzy/rule/ensemble scoring is handled by AMLInferenceService.
    """

    def __init__(self, config: dict, retriever, reranker, entity_extractor, embedding_model):
        self.config = config
        self.retriever = retriever
        self.reranker = reranker
        self._entity_extractor = entity_extractor
        self.embedding_model = embedding_model
        
        self.batch_size = config.get("embedding", {}).get("batch_size", 32)
        self.reranker_prefilter_score = config.get("retrieval", {}).get("reranker_prefilter_score", 0.60)
        self.reranker_top_k = config.get("retrieval", {}).get("reranker_top_k", 10)


    def process_batch(self, raw_explanations: List[str], row_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Process a batch of EFT explanations up to the reranker stage.
        
        Args:
            raw_explanations: List of raw EFT explanations.
            row_ids: Unique identifier for each explanation to map results back.
            
        Returns:
            A dict mapping row_id -> {
                "norm_exp": str,
                "clean_text": str,
                "extraction": ExtractionResult or None,
                "candidates": list,
                "pipeline_status": str,
                "no_candidate_reason": str,
                "channel_counts": dict,
                "metrics": dict
            }
        """
        if len(raw_explanations) != len(row_ids):
            raise ValueError("raw_explanations and row_ids must have equal lengths.")

        metrics = {"ner_duration_s": 0.0, "retrieval_duration_s": 0.0, "reranker_duration_s": 0.0, "embedding_duration_s": 0.0}

        # Duplicate row_id guard — early fail before any expensive work
        row_index: dict[str, int] = {}
        for i, rid in enumerate(row_ids):
            if rid in row_index:
                raise ValueError(f"Duplicate row_id detected: '{rid}'. All row_ids must be unique.")
            row_index[rid] = i

        # 1. Prepare raw texts for NER and guard against Empty/NaN expressions becoming "nan"
        def _is_empty_or_nan(val: Any) -> bool:
            if val is None or pd.isna(val):
                return True
            s = str(val).strip()
            return not s or s.casefold() == "nan"

        raw_texts = ["" if _is_empty_or_nan(exp) else str(exp) for exp in raw_explanations]
        norm_exps = [text.casefold() for text in raw_texts]

        # 2. NER / Entity Extraction (on original cased text)
        ner_start = time.time()
        extractions = None
        if self._entity_extractor:
            extractions = self._entity_extractor.batch_extract(raw_texts, row_ids=row_ids)
        metrics["ner_duration_s"] = time.time() - ner_start

        # Determine the cleanest text to use for retrieval and embedding
        from src.utils.text_utils import normalize_for_matching
        clean_texts = []
        for i, exp in enumerate(norm_exps):
            if not exp:
                clean_texts.append("")
                continue
            clean_text = exp
            if extractions and extractions[i] and extractions[i].extracted_entity:
                clean_text = extractions[i].extracted_entity
            clean_text = normalize_for_matching(clean_text)
            clean_texts.append(clean_text)

        # 3. Embedding Generation (using clean_texts, skip empty ones)
        emb_start = time.time()
        embeddings = [None] * len(clean_texts)
        valid_indices = [i for i, text in enumerate(clean_texts) if text]
        if self.embedding_model and valid_indices:
            valid_texts = [clean_texts[i] for i in valid_indices]
            valid_embs = self.embedding_model.encode(
                valid_texts, batch_size=self.batch_size, show_progress_bar=False
            )
            for idx, emb in zip(valid_indices, valid_embs):
                embeddings[idx] = emb
        metrics["embedding_duration_s"] = time.time() - emb_start

        # 4. Batch Candidate Retrieval (only query DB for valid non-empty rows)
        rows_for_batch = [
            {
                "row_id": row_ids[i],
                "normalized_explanation": clean_texts[i],
                "embedding": embeddings[i].tolist() if embeddings[i] is not None else None,
            }
            for i in valid_indices
        ]

        retrieval_start = time.time()
        all_retrieval_data = self.retriever.batch_get_candidates(rows_for_batch)
        metrics["retrieval_duration_s"] = time.time() - retrieval_start

        # Populate explicit EMPTY_EXPLANATION / INVALID_INPUT status for empty rows
        for i, rid in enumerate(row_ids):
            if i not in valid_indices:
                all_retrieval_data[rid] = {
                    "candidates": [],
                    "pipeline_status": "EMPTY_EXPLANATION",
                    "no_candidate_reason": "INVALID_INPUT",
                    "channel_counts": {}
                }

        # 4.5. Candidate-Supported Extraction Pass (Stage 2)
        # For rows where NER and Rule-based extraction did not find an entity (or returned ENTITY_NOT_FOUND),
        # utilize the retrieved candidates to run candidate-supported extraction layers (Layers 3 & 4).
        if self._entity_extractor and extractions:
            for i in valid_indices:
                ext = extractions[i]
                if ext and (ext.extraction_method == "ENTITY_NOT_FOUND" or not ext.extracted_entity):
                    rid = row_ids[i]
                    candidates = all_retrieval_data.get(rid, {}).get("candidates", [])
                    if candidates:
                        cand_ext = self._entity_extractor._extract_via_candidates(clean_texts[i], candidates)
                        if not cand_ext:
                            cand_ext = self._entity_extractor._extract_via_variant_fallback(clean_texts[i], candidates)
                        if cand_ext:
                            extractions[i] = cand_ext

        # 5. Batched Reranking
        reranker_start = time.time()
        reranker_bulk_data = {}
        for rid, retrieval_data in all_retrieval_data.items():
            candidates = retrieval_data.get("candidates", [])
            idx = row_index[rid]  # O(1) lookup via pre-built map
            clean_text = clean_texts[idx]

            # Pre-filter strong candidates to send to reranker
            strong = [
                c for c in candidates
                if c.get("candidate_score", 0.0) >= self.reranker_prefilter_score
                or self._exact_name_score(clean_text, c.get("variant_name", "")) == 1.0
            ]
            if not strong and candidates:
                strong = candidates

            # Apply reranker_top_k — don't flood reranker with all candidates
            strong = sorted(strong, key=lambda c: c.get("candidate_score", 0.0), reverse=True)
            strong = strong[:self.reranker_top_k]

            reranker_bulk_data[rid] = {"norm_exp": clean_text, "candidates": strong}

        if self.reranker:
            self.reranker.score_candidates_bulk(reranker_bulk_data)
            # Re-sort candidates by normalized_reranker_score after reranking
            for rid in reranker_bulk_data:
                cands = reranker_bulk_data[rid].get("candidates", [])
                reranker_bulk_data[rid]["candidates"] = sorted(
                    cands,
                    key=lambda c: c.get("normalized_reranker_score", c.get("candidate_score", 0.0)),
                    reverse=True
                )
        metrics["reranker_duration_s"] = time.time() - reranker_start


        # Assemble final results map
        results = {}
        for i, rid in enumerate(row_ids):
            ret_data = all_retrieval_data.get(rid, {})
            cands = reranker_bulk_data.get(rid, {}).get("candidates", [])
            
            results[rid] = {
                "norm_exp": norm_exps[i],
                "clean_text": clean_texts[i],
                "extraction": extractions[i] if extractions else None,
                "candidates": cands,
                "pipeline_status": ret_data.get("pipeline_status", "CANDIDATES_FOUND"),
                "no_candidate_reason": ret_data.get("no_candidate_reason"),
                "channel_counts": ret_data.get("channel_counts", {}),
                "metrics": metrics
            }
            
        return results

    def _exact_name_score(self, query: str, variant_name: str) -> float:
        return 1.0 if query.strip().casefold() == variant_name.strip().casefold() else 0.0
