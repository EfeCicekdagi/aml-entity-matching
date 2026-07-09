"""
bge-m3 migration runner:
  1. gold_company_embedding kolonunu 384 -> 1024 boyuta migrate eder
  2. HNSW index'i yeniden olusturur
  3. Tum sirket variant'larini yeni modelle yeniden embed eder
"""
import os
import sys
import logging
import psycopg2

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run():
    config_loader = ConfigLoader()
    db_cfg  = config_loader.get_db_config()
    emb_cfg = config_loader.get_embedding_config()

    repo = AMLRepository(
        host=db_cfg["host"], port=db_cfg["port"],
        dbname=db_cfg["name"], user=db_cfg["user"], password=db_cfg["password"]
    )
    conn = repo.get_connection()
    if not conn:
        logger.error("DB baglantisi basarisiz.")
        sys.exit(1)

    # ── STEP 1: Schema migration ──────────────────────────────────────────────
    logger.info("STEP 1: Schema migration (VECTOR(384) -> VECTOR(1024))...")
    migration_sql = open("sql/05_migrate_embedding_dim.sql", encoding="utf-8").read()
    try:
        with conn.cursor() as cur:
            cur.execute(migration_sql)
        conn.commit()
        logger.info("Migration tamamlandi.")
    except Exception as e:
        logger.error(f"Migration hatasi: {e}")
        conn.rollback()
        repo.release_connection(conn)
        sys.exit(1)

    # ── STEP 2: Load bge-m3 model ─────────────────────────────────────────────
    model_name = emb_cfg.get("model_name", "BAAI/bge-m3")
    batch_size = emb_cfg.get("batch_size", 32)
    logger.info(f"STEP 2: Embedding modeli yukleniyor: {model_name}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    logger.info(f"Model yuklendi. Boyut: {model.get_sentence_embedding_dimension()}")

    # ── STEP 3: Fetch all active variants ────────────────────────────────────
    logger.info("STEP 3: Aktif variant'lar cekiliyor...")
    with conn.cursor() as cur:
        cur.execute("""
            SELECT variant_id, company_id, normalized_variant_name
            FROM silver_company_variant
            WHERE is_active = true
            ORDER BY variant_id
        """)
        variants = cur.fetchall()
    logger.info(f"Toplam {len(variants)} variant bulundu.")

    # ── STEP 4: Re-embed in batches ───────────────────────────────────────────
    logger.info(f"STEP 4: Embedding uretiliyor (batch_size={batch_size})...")
    texts       = [v[2] for v in variants]
    variant_ids = [v[0] for v in variants]
    company_ids = [v[1] for v in variants]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True   # bge-m3 icin cosine similarity icin L2 normalize
    )

    # ── STEP 5: Bulk insert ───────────────────────────────────────────────────
    logger.info("STEP 5: Embedding'ler DB'ye yaziliyor...")
    from psycopg2.extras import execute_values

    rows = []
    for vid, cid, emb in zip(variant_ids, company_ids, embeddings):
        vec_str = "[" + ",".join(str(x) for x in emb.tolist()) + "]"
        rows.append((vid, cid, vec_str, model_name))

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO gold_company_embedding
                (variant_id, company_id, embedding, embedding_model_name)
            VALUES %s
        """, rows, template="(%s, %s, %s::vector, %s)")
    conn.commit()
    logger.info(f"{len(rows)} embedding basariyla yazildi.")

    # ── STEP 6: Verify ────────────────────────────────────────────────────────
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*), embedding_model_name FROM gold_company_embedding GROUP BY 2")
        logger.info("Dogrulama:")
        for r in cur.fetchall():
            logger.info(f"  {r[1]}: {r[0]} embedding")

    repo.release_connection(conn)
    logger.info("Migration ve reseed tamamlandi!")


if __name__ == "__main__":
    run()
