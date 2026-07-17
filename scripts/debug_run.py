import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository

def debug():
    config_loader = ConfigLoader()
    cfg = config_loader.get_db_config()
    repo = AMLRepository(host=cfg['host'], port=cfg['port'], dbname=cfg['name'], user=cfg['user'], password=cfg['password'])
    conn = repo.get_connection()
    if not conn:
        print("Failed to connect")
        return
        
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
        scores = model.predict([
            ('indo gibl indiaforensic stl01071', 'indiaforensic'),
            ('indiaforensic', 'indiaforensic'),
            ('apple inc', 'apple inc'),
            ('payment to apple inc for laptops', 'apple inc')
        ])
        print("RERANKER DIRECT TEST:")
        for s in scores:
            import math
            print(f"  Raw: {s}, Sigmoid: {1 / (1 + math.exp(-s))}")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    debug()
