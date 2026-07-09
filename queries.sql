-- =============================================================
-- AML Entity Matching - Analiz Sorguları
-- Aktif tablolar: aml_alert, aml_run_log, silver_company_variant,
--                 bronze_blacklist_company_raw, aml_reranker_cache
-- =============================================================

-- ── 1. Son run'un özet sonuçları ─────────────────────────────
SELECT
    run_id,
    pipeline_name,
    started_at,
    finished_at,
    ROUND(EXTRACT(EPOCH FROM (finished_at - started_at)) / 60, 2) AS sure_dakika,
    input_row_count,
    processed_row_count,
    candidate_count,
    alert_count,
    status
FROM aml_run_log
ORDER BY started_at DESC
LIMIT 10;

-- ── 2. Belirli bir run için risk dağılımı ────────────────────
-- (run_id'yi değiştirin)
SELECT
    risk_level,
    COUNT(*)                                  AS alert_sayisi,
    ROUND(AVG(final_score)::numeric, 3)       AS avg_skor,
    ROUND(MIN(final_score)::numeric, 3)       AS min_skor,
    ROUND(MAX(final_score)::numeric, 3)       AS max_skor
FROM aml_alert
WHERE run_id = 'RUN-C0CF1AC4'
GROUP BY risk_level
ORDER BY alert_sayisi DESC;

-- ── 3. En çok alert üreten şirketler ─────────────────────────
SELECT
    v.original_company_name,
    COUNT(*)                                  AS alert_sayisi,
    ROUND(AVG(a.final_score)::numeric, 3)     AS avg_skor,
    COUNT(*) FILTER (WHERE a.risk_level = 'HIGH')   AS high_count,
    COUNT(*) FILTER (WHERE a.risk_level = 'MEDIUM') AS medium_count
FROM aml_alert a
JOIN silver_company_variant v ON a.variant_id = v.variant_id
WHERE a.run_id = 'RUN-C0CF1AC4'
GROUP BY v.original_company_name
ORDER BY alert_sayisi DESC;

-- ── 4. En yüksek skorlu 50 alert ─────────────────────────────
SELECT
    a.eft_id,
    v.original_company_name,
    a.final_score,
    a.risk_level,
    a.alert_status,
    a.created_at
FROM aml_alert a
JOIN silver_company_variant v ON a.variant_id = v.variant_id
WHERE a.run_id = 'RUN-C0CF1AC4'
ORDER BY a.final_score DESC
LIMIT 50;

-- ── 5. Run bazında alert trendi ──────────────────────────────
SELECT
    r.run_id,
    r.started_at::date             AS tarih,
    r.alert_count,
    r.input_row_count,
    ROUND(100.0 * r.alert_count / NULLIF(r.input_row_count, 0), 2) AS alert_orani_pct,
    r.status
FROM aml_run_log r
ORDER BY r.started_at DESC;

-- ── 6. Aynı EFT'ye gelen çoklu alertler (multi-match) ────────
SELECT
    a.eft_id,
    COUNT(*)                              AS eslesen_sirket_sayisi,
    STRING_AGG(v.original_company_name, ' | ' ORDER BY a.final_score DESC) AS sirketler,
    MAX(a.final_score)                    AS max_skor
FROM aml_alert a
JOIN silver_company_variant v ON a.variant_id = v.variant_id
WHERE a.run_id = 'RUN-C0CF1AC4'
GROUP BY a.eft_id
HAVING COUNT(*) > 1
ORDER BY eslesen_sirket_sayisi DESC
LIMIT 20;

-- ── 7. Reranker cache doluluk oranı ──────────────────────────
SELECT
    COUNT(*)                              AS toplam_cache_kaydi,
    reranker_model_version,
    ROUND(AVG(reranker_score)::numeric, 3) AS avg_reranker_skor,
    MIN(created_at)                        AS ilk_kayit,
    MAX(created_at)                        AS son_kayit
FROM aml_reranker_cache
GROUP BY reranker_model_version;

-- ── 8. Kara listedeki şirketler ──────────────────────────────
SELECT
    b.company_name,
    b.list_type,
    b.is_active,
    COUNT(a.alert_id)                     AS toplam_alert
FROM bronze_blacklist_company_raw b
LEFT JOIN silver_company_variant v ON LOWER(v.original_company_name) = LOWER(b.company_name)
LEFT JOIN aml_alert a ON a.variant_id = v.variant_id AND a.run_id = 'RUN-C0CF1AC4'
GROUP BY b.company_name, b.list_type, b.is_active
ORDER BY toplam_alert DESC;

-- ── 9. OPEN alertleri REVIEWED olarak işaretle ───────────────
-- (Sadece belirli bir run için)
-- UPDATE aml_alert
-- SET alert_status = 'REVIEWED'
-- WHERE run_id = 'RUN-C0CF1AC4' AND alert_status = 'OPEN';