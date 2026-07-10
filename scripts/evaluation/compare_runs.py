import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository

cfg = ConfigLoader().get_db_config()
repo = AMLRepository(host=cfg['host'], port=cfg['port'], dbname=cfg['name'],
                     user=cfg['user'], password=cfg['password'])
conn = repo.get_connection()

BGE_RUN = 'RUN-36D6BD0D'

with conn.cursor() as cur:

    # Risk dagilimi
    print("=" * 60)
    print("bge-m3 (RUN-36D6BD0D) - Risk Dagilimi")
    print("=" * 60)
    cur.execute("""
        SELECT risk_level, COUNT(*) as alert,
               ROUND(AVG(final_score)::numeric,3) as avg,
               ROUND(MIN(final_score)::numeric,3) as min,
               ROUND(MAX(final_score)::numeric,3) as max
        FROM aml_alert WHERE run_id=%s
        GROUP BY risk_level ORDER BY alert DESC
    """, (BGE_RUN,))
    print(f"\n  {'Risk':<10} {'Alert':>8} {'Avg':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*46}")
    for r in cur.fetchall():
        print(f"  {str(r[0]):<10} {r[1]:>8} {str(r[2]):>8} {str(r[3]):>8} {str(r[4]):>8}")

    # Sirket bazinda
    print(f"\n[Sirket Bazinda Alert]")
    cur.execute("""
        SELECT v.original_company_name, COUNT(*) as alert,
               ROUND(AVG(a.final_score)::numeric,3) as avg,
               COUNT(*) FILTER (WHERE a.risk_level='HIGH') as high,
               COUNT(*) FILTER (WHERE a.risk_level='MEDIUM') as medium
        FROM aml_alert a
        JOIN silver_company_variant v ON a.variant_id=v.variant_id
        WHERE a.run_id=%s GROUP BY v.original_company_name ORDER BY alert DESC
    """, (BGE_RUN,))
    print(f"  {'Sirket':<40} {'Alert':>7} {'Avg':>7} {'HIGH':>6} {'MED':>6}")
    print(f"  {'-'*68}")
    for r in cur.fetchall():
        print(f"  {str(r[0]):<40} {r[1]:>7} {str(r[2]):>7} {r[3]:>6} {r[4]:>6}")

    # Chunk karsilastirma
    print(f"\n[Chunk Karsilastirma - 3 Run]")
    print(f"  {'Chunk':<9} {'C0CF(mini)':>12} {'9AC8(fix)':>12} {'36D6(bge-m3)':>14}  {'Degisim'}")
    print(f"  {'-'*60}")
    chunk_ranges = [
        (1,0,9999),(2,10000,19999),(3,20000,29999),(4,30000,39999),
        (5,40000,49999),(6,50000,59999),(7,60000,69999),(8,70000,79999),
        (9,80000,89999),(10,90000,99999),(11,100000,109999),(12,110000,120000)
    ]
    for chunk_no, lo, hi in chunk_ranges:
        counts = []
        for run in ['RUN-C0CF1AC4','RUN-9AC839D6','RUN-36D6BD0D']:
            cur.execute("SELECT COUNT(*) FROM aml_alert WHERE run_id=%s AND eft_id BETWEEN %s AND %s",
                        (run, lo, hi))
            counts.append(cur.fetchone()[0])
        delta = f"+{counts[2]-counts[1]}" if counts[2] > counts[1] else str(counts[2]-counts[1])
        print(f"  Chunk {chunk_no:<3} {counts[0]:>12} {counts[1]:>12} {counts[2]:>14}  {delta}")

    # Chunk 2 inceleme
    print(f"\n[Chunk 2 - bge-m3'te neden 4712?]")
    cur.execute("""
        SELECT v.original_company_name, COUNT(*) as cnt,
               ROUND(AVG(a.final_score)::numeric,3) as avg
        FROM aml_alert a JOIN silver_company_variant v ON a.variant_id=v.variant_id
        WHERE a.run_id=%s AND a.eft_id BETWEEN 10000 AND 19999
        GROUP BY v.original_company_name ORDER BY cnt DESC
    """, (BGE_RUN,))
    for r in cur.fetchall():
        print(f"  {str(r[0]):<40} {r[1]:>7} alerts  avg={r[2]}")

    # GPU kontrolu
    print(f"\n[Sistem GPU Durumu]")
    import torch
    print(f"  CUDA mevcut: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name}  ({vram:.1f} GB VRAM)")
    else:
        print("  GPU bulunamadi — CPU kullanilacak")

conn.close()
print("\nAnaliz tamamlandi.")
