"""
Chunk yoğunluk analizi — hangi şirketler / risk seviyeleri yüksek alert üretiyor?
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository

config_loader = ConfigLoader()
db = config_loader.get_db_config()
repo = AMLRepository(
    host=db["host"], port=db["port"], dbname=db["name"],
    user=db["user"], password=db["password"]
)

RUN_ID = "RUN-C0CF1AC4"

# Chunk sinirlari (chunk_size=10000, 0-indexed eft_id)
CHUNKS = {
    4:  (30000, 39999),
    5:  (40000, 49999),
    10: (90000, 99999),
}

conn = repo.get_connection()
with conn.cursor() as cur:

    # -- 1. Genel ozet ----------------------------------------------------------
    print("=" * 70)
    print(f"RUN: {RUN_ID}")
    print("=" * 70)

    cur.execute("""
        SELECT risk_level, COUNT(*) as cnt, ROUND(AVG(final_score)::numeric, 3) as avg_score
        FROM aml_alert
        WHERE run_id = %s
        GROUP BY risk_level
        ORDER BY cnt DESC
    """, (RUN_ID,))
    print("\n[Risk Seviyesi Dagilimi - tum run]")
    print(f"  {'Risk':<10} {'Alert':>8} {'Avg Score':>12}")
    print(f"  {'-'*32}")
    for row in cur.fetchall():
        print(f"  {row[0]:<10} {row[1]:>8} {row[2]:>12}")

    # -- 2. Chunk bazinda karsilastirma -----------------------------------------
    print("\n[Chunk Bazinda Alert Dagilimi]")
    for chunk_no, (lo, hi) in CHUNKS.items():
        cur.execute("""
            SELECT COUNT(*) as alerts,
                   ROUND(AVG(final_score)::numeric,3) as avg_score,
                   COUNT(*) FILTER (WHERE risk_level='HIGH')   as high,
                   COUNT(*) FILTER (WHERE risk_level='MEDIUM') as medium
            FROM aml_alert
            WHERE run_id = %s AND eft_id BETWEEN %s AND %s
        """, (RUN_ID, lo, hi))
        r = cur.fetchone()
        print(f"\n  Chunk {chunk_no} (eft {lo}-{hi}): "
              f"{r[0]} alerts | avg_score={r[1]} | HIGH={r[2]} MEDIUM={r[3]}")

        # Top 10 sirket bu chunk'ta
        cur.execute("""
            SELECT v.original_company_name, COUNT(*) as cnt,
                   ROUND(AVG(a.final_score)::numeric,3) as avg_score
            FROM aml_alert a
            JOIN silver_company_variant v ON a.variant_id = v.variant_id
            WHERE a.run_id = %s AND a.eft_id BETWEEN %s AND %s
            GROUP BY v.original_company_name
            ORDER BY cnt DESC
            LIMIT 10
        """, (RUN_ID, lo, hi))
        rows = cur.fetchall()
        if rows:
            print(f"  {'Sirket':<40} {'Alerts':>7} {'Avg':>8}")
            print(f"  {'-'*57}")
            for r2 in rows:
                print(f"  {str(r2[0]):<40} {r2[1]:>7} {r2[2]:>8}")

    # -- 3. Tum run'daki top 15 sirket ------------------------------------------
    print("\n\n[TOP 15 Sirket - tum run]")
    cur.execute("""
        SELECT v.original_company_name, COUNT(*) as cnt,
               ROUND(AVG(a.final_score)::numeric,3) as avg_score,
               COUNT(*) FILTER (WHERE a.risk_level='HIGH') as high_cnt
        FROM aml_alert a
        JOIN silver_company_variant v ON a.variant_id = v.variant_id
        WHERE a.run_id = %s
        GROUP BY v.original_company_name
        ORDER BY cnt DESC
        LIMIT 15
    """, (RUN_ID,))
    print(f"  {'Sirket':<40} {'Alerts':>7} {'Avg':>8} {'HIGH':>6}")
    print(f"  {'-'*63}")
    for row in cur.fetchall():
        print(f"  {str(row[0]):<40} {row[1]:>7} {row[2]:>8} {row[3]:>6}")

    # -- 4. Chunk 4 vs Chunk 5 score dagilimi -----------------------------------
    print("\n\n[Chunk 4 vs Chunk 5 - Score Dagilimi]")
    for chunk_no, (lo, hi) in [(4, (30000,39999)), (5, (40000,49999))]:
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE final_score >= 0.90) as gt90,
                COUNT(*) FILTER (WHERE final_score >= 0.80 AND final_score < 0.90) as r8090,
                COUNT(*) FILTER (WHERE final_score >= 0.70 AND final_score < 0.80) as r7080,
                COUNT(*) FILTER (WHERE final_score < 0.70) as lt70
            FROM aml_alert
            WHERE run_id = %s AND eft_id BETWEEN %s AND %s
        """, (RUN_ID, lo, hi))
        r = cur.fetchone()
        print(f"  Chunk {chunk_no}: >0.90={r[0]}  0.80-0.90={r[1]}  0.70-0.80={r[2]}  <0.70={r[3]}")

conn.close()
print("\nAnaliz tamamlandi.")
