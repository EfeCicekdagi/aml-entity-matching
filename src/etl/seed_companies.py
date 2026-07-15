import os
import sys
import pandas as pd
import logging
from sentence_transformers import SentenceTransformer

# Add src to python path for easier imports if running from root
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def seed_companies():
    logger.info("Starting company seed process...")
    
    config_loader = ConfigLoader()
    db_config = config_loader.get_db_config()
    emb_config = config_loader.get_embedding_config()
    
    repo = AMLRepository(
        host=db_config.get("host"),
        port=db_config.get("port"),
        dbname=db_config.get("name"),
        user=db_config.get("user"),
        password=db_config.get("password")
    )
    
    conn = repo.get_connection()
    if not conn:
        logger.error("Database connection failed.")
        sys.exit(1)

    try:
        # Load the company CSV
        company_file = "data/company_list.csv"
        if not os.path.exists(company_file):
            logger.error(f"Company file not found: {company_file}")
            sys.exit(1)
            
        df = pd.read_csv(company_file)
        
        # Load embedding model
        model_name = emb_config.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"Loading embedding model {model_name}...")
        model = SentenceTransformer(model_name)
        
        with conn.cursor() as cur:
            # Eski verileri temizle (cascade ile embedding ve variant da silinir)
            logger.info("Clearing existing company data from all tables...")
            cur.execute("DELETE FROM gold_company_embedding;")
            cur.execute("DELETE FROM silver_company_variant;")
            cur.execute("DELETE FROM bronze_blacklist_company_raw;")
            conn.commit()
            logger.info("Old company data cleared.")

        with conn.cursor() as cur:
            for idx, row in df.iterrows():
                # CSV kolonları: id, company_name
                comp_name = str(row.get("company_name", row.iloc[1]))
                norm_name = comp_name.lower().strip()
                
                # 1. Insert into bronze
                cur.execute("""
                    INSERT INTO bronze_blacklist_company_raw (company_name, source_list)
                    VALUES (%s, 'SDN') RETURNING company_id;
                """, (comp_name,))
                db_company_id = cur.fetchone()[0]
                
                # 2. Insert into silver (We'll just insert the original name as the main variant for now)
                cur.execute("""
                    INSERT INTO silver_company_variant (company_id, original_company_name, variant_name, normalized_variant_name, variant_type)
                    VALUES (%s, %s, %s, %s, 'ORIGINAL') RETURNING variant_id;
                """, (db_company_id, comp_name, comp_name, norm_name))
                variant_id = cur.fetchone()[0]
                
                # 3. Create Embedding and insert into gold
                emb = model.encode([norm_name])[0].tolist()
                cur.execute("""
                    INSERT INTO gold_company_embedding (variant_id, company_id, embedding, embedding_model_name)
                    VALUES (%s, %s, %s::vector, %s);
                """, (variant_id, db_company_id, emb, model_name))
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"Inserted {idx + 1} companies...")
                    
            conn.commit()
            logger.info(f"Successfully seeded {len(df)} companies into the database!")
            
    except Exception as e:
        logger.error(f"Failed to seed companies: {e}")
        conn.rollback()
    finally:
        repo.release_connection(conn)

if __name__ == "__main__":
    seed_companies()
