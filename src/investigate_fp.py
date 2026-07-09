"""
Indiaforensic false positive kök neden analizi
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository

cfg = ConfigLoader().get_db_config()
repo = AMLRepository(host=cfg['host'], port=cfg['port'], dbname=cfg['name'],
                     user=cfg['user'], password=cfg['password'])
conn = repo.get_connection()
RUN_ID = "RUN-C0CF1AC4"

with conn.cursor() as cur:

    # 1. Tablolar
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY 1")
    print("=== Mevcut Tablolar ===")
    tables = [r[0] for r in cur.fetchall()]
    for t in tables: print(f"  {t}")

    # 2. EFT tablosu var mi?
    eft_table = None
    for candidate in ["bank_efts", "eft_transactions", "efts", "transactions"]:
        if candidate in tables:
            eft_table = candidate
            break
    print(f"\nEFT tablosu: {eft_table}")

    # 3. Alert'lerdeki sample EFT id'leri cek, CSV'den aciklamalari oku
    cur.execute("""
        SELECT a.eft_id, a.final_score
        FROM aml_alert a
        WHERE a.run_id = %s AND a.variant_id = 6
        ORDER BY a.final_score DESC
        LIMIT 20
    """, (RUN_ID,))
    sample_ids = cur.fetchall()
    print(f"\nSample alert eft_id'leri (variant_id=6, top 20 by score):")
    for r in sample_ids:
        print(f"  eft_id={r[0]}  final_score={r[1]:.4f}")

    # 4. EFT tablosundaki verilere bak (varsa)
    if eft_table:
        cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name='{eft_table}' ORDER BY ordinal_position")
        cols = [r[0] for r in cur.fetchall()]
        print(f"\n{eft_table} kolonlari: {cols}")

        eft_ids = [r[0] for r in sample_ids[:10]]
        cur.execute(f"SELECT * FROM {eft_table} WHERE id = ANY(%s) LIMIT 10", (eft_ids,))
        rows = cur.fetchall()
        print(f"\nSample EFT kayitlari:")
        for r in rows:
            print(f"  {r}")

conn.close()

# 5. CSV'den direkt oku (DB yoksa)
print("\n=== CSV'den sample EFT aciklamalari ===")
import pandas as pd
df = pd.read_csv("data/bank_efts.csv")
print(f"Kolonlar: {list(df.columns)}")
if sample_ids:
    idx_list = [r[0] for r in sample_ids[:15]]
    # eft_id = DataFrame index
    sample_rows = df.loc[df.index.isin(idx_list)]
    print(f"\nEFT aciklamalari (indiaforensic alertlari):")
    for _, row in sample_rows.iterrows():
        desc = row.get('description', row.iloc[0] if len(row) > 0 else '?')
        print(f"  idx={row.name}: {str(desc)[:100]}")
