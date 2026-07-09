-- ============================================================
-- Migration: VECTOR(384) -> VECTOR(1024)
-- Embedding modeli: all-MiniLM-L6-v2 -> BAAI/bge-m3
-- ============================================================

-- 1. Eski HNSW index'i dusur (boyut degisince rebuild gerekiyor)
DROP INDEX IF EXISTS idx_company_embedding_hnsw;

-- 2. Embedding kolonunu yeni boyuta degistir
--    (Onceki veriler silinir — zaten yeniden seed edilecek)
ALTER TABLE gold_company_embedding
    DROP COLUMN embedding;

ALTER TABLE gold_company_embedding
    ADD COLUMN embedding VECTOR(1024);

-- 3. Eski kayitlari temizle (boyut degisti, gecersizler)
TRUNCATE TABLE gold_company_embedding;

-- 4. Yeni HNSW index olustur (1024 boyut icin)
CREATE INDEX idx_company_embedding_hnsw
ON gold_company_embedding
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
