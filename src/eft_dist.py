import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd

df = pd.read_csv("data/bank_efts.csv")
mask = df['description'].str.lower().str.contains('indiaforensic', na=False)
print(f"Toplam EFT: {len(df)}")
print(f"Indiaforensic iceren: {mask.sum()} ({100*mask.mean():.1f}%)")
print(f"Indiaforensic icermeyen: {(~mask).sum()}")

for kw in ['acme', 'north star', 'kuzey', 'abc trading']:
    m = df['description'].str.lower().str.contains(kw, na=False)
    print(f'  [{kw}] iceren: {m.sum()}')

# Indiaforensic olmayan EFT'lerin kac tanesi alert uretti?
from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository
cfg = ConfigLoader().get_db_config()
repo = AMLRepository(host=cfg['host'], port=cfg['port'], dbname=cfg['name'],
                     user=cfg['user'], password=cfg['password'])
conn = repo.get_connection()
with conn.cursor() as cur:
    cur.execute("""
        SELECT COUNT(DISTINCT eft_id) FROM aml_alert
        WHERE run_id='RUN-C0CF1AC4' AND variant_id != 6
    """)
    print(f"\nIndiaforensic DISINDAKI unique EFT alertlari: {cur.fetchone()[0]}")

    cur.execute("""
        SELECT v.original_company_name, COUNT(*) as cnt,
               ROUND(AVG(a.final_score)::numeric,3) as avg
        FROM aml_alert a
        JOIN silver_company_variant v ON a.variant_id=v.variant_id
        WHERE a.run_id='RUN-C0CF1AC4' AND a.variant_id != 6
        GROUP BY v.original_company_name ORDER BY cnt DESC
    """)
    print("\nDiger sirket alertlari:")
    for r in cur.fetchall():
        print(f"  {r[0]:<35} alerts={r[1]}  avg={r[2]}")
conn.close()
