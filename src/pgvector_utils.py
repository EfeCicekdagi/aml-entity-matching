import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
import logging

from config import PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD

logger = logging.getLogger(__name__)

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        return None

def init_db(conn):
    """
    Initializes pgvector extension and creates the necessary tables.
    """
    if conn is None: return
    
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS company_aliases (
                    id serial PRIMARY KEY,
                    alias_row_id integer,
                    company_id text,
                    company_name text,
                    alias text,
                    normalized_alias text,
                    embedding vector(384)
                );
            """)
            # Clear existing data for fresh run (for testing purposes)
            cur.execute("TRUNCATE TABLE company_aliases;")
        conn.commit()
        register_vector(conn)
        logger.info("PostgreSQL database initialized with pgvector.")
    except Exception as e:
        logger.error(f"Error initializing DB: {e}")
        conn.rollback()

def insert_alias_embeddings(conn, alias_df, alias_embeddings):
    """
    Inserts alias vectors into PostgreSQL.
    """
    if conn is None or alias_embeddings is None: return
    
    register_vector(conn)
    
    try:
        with conn.cursor() as cur:
            for idx, row in alias_df.iterrows():
                embedding = alias_embeddings[idx]
                # convert embedding to list
                cur.execute("""
                    INSERT INTO company_aliases (alias_row_id, company_id, company_name, alias, normalized_alias, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    row.get('alias_row_id'),
                    str(row.get('company_id')),
                    row.get('company_name'),
                    row.get('alias'),
                    row.get('normalized_alias'),
                    embedding.tolist()
                ))
        conn.commit()
        logger.info(f"Inserted {len(alias_df)} records into PostgreSQL company_aliases table.")
    except Exception as e:
        logger.error(f"Error inserting embeddings: {e}")
        conn.rollback()

def search_pgvector(conn, query_embedding, k=5):
    """
    Searches for the most similar vectors using cosine distance (<=>).
    Returns list of dictionaries containing alias details and scores.
    Cosine similarity = 1 - cosine_distance
    """
    if conn is None or query_embedding is None: return []
    
    try:
        with conn.cursor() as cur:
            # We use cosine distance operator <=> 
            # 1 - (embedding <=> %s) gives cosine similarity
            cur.execute("""
                SELECT alias_row_id, company_id, company_name, alias, normalized_alias, 
                       1 - (embedding <=> %s::vector) AS similarity
                FROM company_aliases
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_embedding.tolist(), query_embedding.tolist(), k))
            
            results = cur.fetchall()
            
            # Format results
            matches = []
            for r in results:
                matches.append({
                    "alias_row_id": r[0],
                    "company_id": r[1],
                    "company_name": r[2],
                    "alias": r[3],
                    "normalized_alias": r[4],
                    "candidate_filter_score": float(r[5]),
                    "candidate_source": "Postgres Vector Match"
                })
            return matches
    except Exception as e:
        logger.error(f"Error during pgvector search: {e}")
        conn.rollback()
        return []
