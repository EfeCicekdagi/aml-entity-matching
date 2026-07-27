import logging
import re
from typing import Optional, List, Dict, Any
from transformers import pipeline

logger = logging.getLogger(__name__)

_COMPANY_SUFFIX_PATTERN = re.compile(
    r"\b(ltd|limited|llc|inc|incorporated|corp|corporation|co|company|"
    r"plc|llp|pvt|a\.s\.|a\.ş\.|anonim|sirketi|holding|group|"
    r"international|intl|trading|import|export|logistics|energy|san\.|tic\.|ve|sti\.|şti\.)\b",
    re.IGNORECASE
)

class NERExtractor:
    """
    Named Entity Recognition (NER) Extractor.
    Extracts Organization (ORG) or Person (PER) names from raw text using a Turkish BERT model.
    """
    def __init__(self, model_name: str = "savasy/bert-base-turkish-ner-cased", device: int = -1):
        """
        Initializes the NER pipeline.
        :param device: -1 for CPU, 0+ for GPU.
        """
        logger.info(f"Loading NER model: {model_name} on device {device}...")
        self.ner_pipeline = pipeline(
            "ner", 
            model=model_name, 
            aggregation_strategy="simple",
            device=device
        )
        logger.info("NER model loaded successfully.")

    def extract_entity(self, text: str) -> Optional[dict]:
        """
        Extracts the first valid ORG or PER entity found in the text as metadata dict.
        Returns:
            dict: {"text": str, "entity_type": str, "confidence": float, "start": int, "end": int}
            or None if no entity is found.
        """
        if not text or len(text.strip()) == 0:
            return None

        # The pipeline returns a list of dicts:
        # [{'entity_group': 'ORG', 'score': 0.99, 'word': 'XYZ Sirketi', 'start': 10, 'end': 21}]
        try:
            from src.utils.text_utils import clean_spaced_characters
            text = clean_spaced_characters(text)

            results = self.ner_pipeline(text)
            
            valid_results = [
                res for res in results 
                if isinstance(res, dict) and res.get('entity_group') in ['ORG', 'PER'] and res.get('score', 0) > 0.50
            ]
            
            return self._select_best_entity(valid_results)
        except Exception as e:
            logger.error(f"NER Extraction failed for text '{text}': {e}")
            return None

    def batch_extract_entities(self, texts: list[str]) -> list[Optional[dict]]:
        """
        Runs NER over a batch of texts using GPU batching.
        Returns a list of extracted entity metadata dicts (or None if not found) matching the input order.
        """
        if not texts:
            return []
            
        logger.debug(f"Running batched NER extraction for {len(texts)} texts...")
        
        # Preprocess text (clean spaced characters)
        from src.utils.text_utils import clean_spaced_characters
        processed_texts = [clean_spaced_characters(text) if text else "" for text in texts]
        
        try:
            # Batch size 64 for good GPU utilization on 3060
            batch_results = self.ner_pipeline(processed_texts, batch_size=64)
        except Exception as e:
            logger.error(f"Batched NER Extraction failed: {e}")
            return [None] * len(texts)
            
        extracted = []
        # If len(processed_texts) == 1, pipeline returns list[dict].
        # If > 1, it returns list[list[dict]].
        if len(processed_texts) == 1 and (not batch_results or isinstance(batch_results[0], dict)):
            batch_results = [batch_results]
            
        for results in batch_results:
            if not results:
                extracted.append(None)
                continue
                
            valid_results = [
                res for res in results 
                if isinstance(res, dict) and res.get('entity_group') in ['ORG', 'PER'] and res.get('score', 0) > 0.50
            ]
            
            extracted.append(self._select_best_entity(valid_results))
                
        return extracted

    def _select_best_entity(self, valid_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Gelen geçerli NER sonuçları arasından en doğru entity'yi seçer.
        Yalnızca uzunluğa (max key=len) bakmak yerine:
        1. Kuruluş (ORG) önceliği (Şirket eşleştirmesi birincil amacımızdır) -> +100.0 puan
        2. Şirket uzantısı (suffix) barındırma -> +20.0 puan
        3. Model güven skoru (confidence) -> +10.0 * score
        4. Metin uzunluğu heuristiği -> +0.1 * min(len(cleaned), 50)
        kriterlerini birlikte değerlendirir.
        """
        if not valid_results:
            return None
            
        best_res = None
        best_cleaned = ""
        best_score = -1.0
        
        for res in valid_results:
            cleaned = res.get("word", "").replace("##", "").strip()
            if not cleaned:
                continue
                
            score = 0.0
            if res.get("entity_group") == "ORG":
                score += 100.0
            if _COMPANY_SUFFIX_PATTERN.search(cleaned):
                score += 20.0
            score += float(res.get("score", 0.0)) * 10.0
            score += min(len(cleaned), 50) * 0.1
            
            if score > best_score:
                best_score = score
                best_cleaned = cleaned
                best_res = res
                
        if best_cleaned and best_res:
            entity_type_map = {
                "ORG": "ORGANIZATION",
                "PER": "PERSON"
            }
            mapped_type = entity_type_map.get(best_res.get("entity_group"), "UNKNOWN")
            return {
                "text": best_cleaned,
                "entity_type": mapped_type,
                "confidence": float(best_res.get("score", 0.0)),
                "start": best_res.get("start"),
                "end": best_res.get("end")
            }
        return None
