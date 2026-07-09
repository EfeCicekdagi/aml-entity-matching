"""
bge-m3 icin skor dagilimini analiz et ve optimal threshold'lari bul.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository

cfg = ConfigLoader().get_db_config()
repo = AMLRepository(host=cfg['host'], port=cfg['port'], dbname=cfg['name'],
                     user=cfg['user'], password=cfg['password'])
conn = repo.get_connection()
RUN = 'RUN-36D6BD0D'

with conn.cursor() as cur:
    # Mevcut threshold config
    cur.execute("SELECT config_version, risk_level, min_score, max_score, is_active FROM aml_threshold_config ORDER BY config_version, min_score DESC")
    print("=== Mevcut Threshold Config ===")
    for r in cur.fetchall():
        print(f"  {r[0]:<35} {r[1]:<8} {r[2]:>6} - {r[3]:>6}  active={r[4]}")

    # Skor dagilimi (percentile)
    cur.execute(f"""
        SELECT
            ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY final_score)::numeric,3) AS p50,
            ROUND(PERCENTILE_CONT(0.70) WITHIN GROUP (ORDER BY final_score)::numeric,3) AS p70,
            ROUND(PERCENTILE_CONT(0.80) WITHIN GROUP (ORDER BY final_score)::numeric,3) AS p80,
            ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY final_score)::numeric,3) AS p90,
            ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY final_score)::numeric,3) AS p95,
            ROUND(AVG(final_score)::numeric,3) AS avg,
            ROUND(MIN(final_score)::numeric,3) AS min,
            ROUND(MAX(final_score)::numeric,3) AS max,
            COUNT(*) as total
        FROM aml_alert WHERE run_id = '{RUN}'
    """)
    r = cur.fetchone()
    print(f"\n=== bge-m3 Skor Dagilimi (tum alertler) ===")
    print(f"  Min={r[6]}  Max={r[7]}  Avg={r[5]}  Toplam={r[8]}")
    print(f"  P50={r[0]}  P70={r[1]}  P80={r[2]}  P90={r[3]}  P95={r[4]}")

    # Her 0.05 araligiyla kacinc alert var
    print(f"\n=== Score Bucket Dagilimi ===")
    print(f"  {'Aralik':<15} {'Alert':>8} {'%':>8}")
    print(f"  {'-'*33}")
    buckets = [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),
               (0.70,0.75),(0.75,0.80),(0.80,0.85),(0.85,1.01)]
    total = r[8]
    for lo, hi in buckets:
        cur.execute(f"""
            SELECT COUNT(*) FROM aml_alert
            WHERE run_id='{RUN}' AND final_score >= {lo} AND final_score < {hi}
        """)
        cnt = cur.fetchone()[0]
        pct = 100*cnt/total if total else 0
        bar = '#' * int(pct/2)
        print(f"  {lo:.2f}-{hi:.2f}       {cnt:>8}   {pct:>6.1f}%  {bar}")

    # Sub-score dagilimi — hangi score'lar yuksek?
    print(f"\n=== Sub-Score Katkisi (bge-m3 run) ===")
    cur.execute(f"""
        SELECT
            ROUND(AVG(fuzzy_score)::numeric,3)    AS avg_fuzzy,
            ROUND(AVG(vector_score)::numeric,3)   AS avg_vector,
            ROUND(AVG(acronym_score)::numeric,3)  AS avg_acronym,
            ROUND(AVG(rule_score)::numeric,3)     AS avg_rule,
            ROUND(AVG(reranker_score)::numeric,3) AS avg_reranker
        FROM aml_scoring_result WHERE run_id='{RUN}'
    """)
    r2 = cur.fetchone()
    if r2 and r2[0] is not None:
        print(f"  fuzzy={r2[0]}  vector={r2[1]}  acronym={r2[2]}  rule={r2[3]}  reranker={r2[4]}")
    else:
        # aml_scoring_result bos olabilir, alert tablosundan cikarsayalim
        print("  (aml_scoring_result bos — alert skorlari final_score uzerinden analiz ediliyor)")

    # HIGH vs MEDIUM: skor farki
    print(f"\n=== HIGH vs MEDIUM Skor Farki ===")
    for risk in ['HIGH', 'MEDIUM']:
        cur.execute(f"""
            SELECT COUNT(*), ROUND(AVG(final_score)::numeric,3),
                   ROUND(MIN(final_score)::numeric,3), ROUND(MAX(final_score)::numeric,3)
            FROM aml_alert WHERE run_id='{RUN}' AND risk_level='{risk}'
        """)
        r3 = cur.fetchone()
        print(f"  {risk:<8}: n={r3[0]}  avg={r3[1]}  min={r3[2]}  max={r3[3]}")

conn.close()
print("\nAnaliz tamamlandi.")
