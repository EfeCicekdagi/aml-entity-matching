import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository

def analyze_false_negatives():
    config_loader = ConfigLoader()
    db_cfg = config_loader.get_db_config()
    repo = AMLRepository(
        host=db_cfg.get("host"), port=db_cfg.get("port"),
        dbname=db_cfg.get("name"), user=db_cfg.get("user"), password=db_cfg.get("password")
    )
    conn = repo.get_connection()
    
    query = """
    SELECT 
        d.expected_company,
        d.retrieved_company,
        d.fuzzy_score,
        d.vector_score,
        d.reranker_score,
        d.final_score,
        d.reason_code,
        e.explanation,
        e.eft_id
    FROM aml_experiment.decision_analysis d
    LEFT JOIN bronze_eft_raw e ON e.eft_id::varchar = d.eft_id::varchar
    WHERE d.error_type = 'FN'
    """
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df = pd.read_sql_query(query, conn)
    
    repo.release_connection(conn)
    
    print(f"Toplam False Negative (FN) Sayısı: {len(df)}")
    
    if len(df) == 0:
        print("İncelenecek False Negative kaydı bulunamadı.")
        return
        
    print("\n--- Neden (Reason Code) Dağılımı ---")
    print(df['reason_code'].value_counts())
    
    print("\n--- Retrieval Aşaması Analizi ---")
    # Aday bulunamamış olanları None veya boş olarak sayalım
    missing_retrieval = df[df['retrieved_company'].isnull() | (df['retrieved_company'] == '') | (df['retrieved_company'] == 'None')]
    print(f"Retrieval aşamasında hiç aday bulunamayan (veya doğru aday getirilemeyen) kayıt sayısı: {len(missing_retrieval)}")
    print(f"Bunun tüm FN'lere oranı: {len(missing_retrieval) / len(df):.1%}")
    
    if len(df) > len(missing_retrieval):
        retrieved_but_missed = df[~df.index.isin(missing_retrieval.index)]
        print("\n--- Aday Bulunup Skor Yetersizliğinden Kaçanlar Analizi ---")
        print(f"Kayıt sayısı: {len(retrieved_but_missed)}")
        print("Ortalama Skorlar:")
        print(f"  Fuzzy: {retrieved_but_missed['fuzzy_score'].mean():.3f}")
        print(f"  Vector: {retrieved_but_missed['vector_score'].mean():.3f}")
        print(f"  Reranker: {retrieved_but_missed['reranker_score'].mean():.3f}")
        print(f"  Final: {retrieved_but_missed['final_score'].mean():.3f}")
    
    print("\n--- Örnek 5 FN Kaydı ---")
    print(df.head(5).to_string())

if __name__ == "__main__":
    analyze_false_negatives()
