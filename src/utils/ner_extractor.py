import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

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

    def extract_entity(self, text: str) -> str:
        """
        Extracts the first valid ORG or PER entity found in the text.
        If multiple exist, it takes the longest or first one based on heuristics.
        For simplicity, returns the highest scored or longest ORG/PER found.
        If no entity is found, returns None.
        """
        if not text or len(text.strip()) == 0:
            return None

        # The pipeline returns a list of dicts:
        # [{'entity_group': 'ORG', 'score': 0.99, 'word': 'XYZ Sirketi', 'start': 10, 'end': 21}]
        try:
            # Title case the text to help the cased NER model if it's all lowercase
            if text.islower():
                text_to_process = text.title()
            else:
                text_to_process = text

            results = self.ner_pipeline(text_to_process)
            
            entities = [
                res['word'] for res in results 
                if res['entity_group'] in ['ORG', 'PER'] and res['score'] > 0.50
            ]
            
            if entities:
                # Often the first extracted ORG/PER is the main one in an EFT.
                # We join them if they are split, but simple strategy already groups them.
                # Clean up WordPiece artifacts like ## from the tokens
                cleaned_entities = [ent.replace("##", "").strip() for ent in entities]
                cleaned_entities = [ent for ent in cleaned_entities if ent]
                
                if cleaned_entities:
                    # Let's return the longest one to be safe (captures full names better)
                    return max(cleaned_entities, key=len)
            
            return None
        except Exception as e:
            logger.error(f"NER Extraction failed for text '{text}': {e}")
            return None

    def batch_extract_entities(self, texts: list[str]) -> list[str]:
        """
        Runs NER over a batch of texts.
        Returns a list of extracted entities (or None if not found) matching the input order.
        """
        logger.debug(f"Running NER extraction for batch of {len(texts)} texts...")
        results = []
        for text in texts:
            results.append(self.extract_entity(text))
        return results
