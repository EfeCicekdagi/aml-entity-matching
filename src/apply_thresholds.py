"""
bge-m3 icin yeni threshold versiyonu DB'ye ekler ve config'i gunceller.
Kalibre edilen degerler:
  HIGH   : >= 0.70  (net kara liste eslesmesi)
  MEDIUM : 0.62-0.70  (inceleme gerektiren)
  LOW    : 0.50-0.62  (zayif eslesme, log'a yazilir ama alert uretilmez)
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils.config_loader import ConfigLoader
from src.repository.aml_repository import AMLRepository

cfg = ConfigLoader().get_db_config()
repo = AMLRepository(host=cfg['host'], port=cfg['port'], dbname=cfg['name'],
                     user=cfg['user'], password=cfg['password'])
conn = repo.get_connection()

NEW_VERSION = 'threshold_v3_bge_m3'

THRESHOLDS = [
    # (risk_level, min_score, max_score)
    ('HIGH',     0.70, 1.00),   # guvenilir esleme — direkt alert
    ('MEDIUM',   0.62, 0.70),   # inceleme gerekli
    ('LOW',      0.50, 0.62),   # zayif, sadece log
    ('NO_MATCH', 0.00, 0.50),   # eslesme yok
]

with conn.cursor() as cur:
    # Eski versiyonlari pasif yap
    cur.execute("UPDATE aml_threshold_config SET is_active=false WHERE config_version != %s",
                (NEW_VERSION,))

    # Yeni versiyonu ekle (varsa guncelle)
    cur.execute("DELETE FROM aml_threshold_config WHERE config_version=%s", (NEW_VERSION,))
    for risk, lo, hi in THRESHOLDS:
        cur.execute("""
            INSERT INTO aml_threshold_config
                (config_version, risk_level, min_score, max_score, is_active)
            VALUES (%s, %s, %s, %s, true)
        """, (NEW_VERSION, risk, lo, hi))

    conn.commit()
    print(f"Threshold versiyonu yazildi: {NEW_VERSION}")
    print(f"  HIGH   : >= 0.70")
    print(f"  MEDIUM : 0.62 - 0.70")
    print(f"  LOW    : 0.50 - 0.62  (alert uretilmez)")
    print(f"  NO_MATCH: < 0.50")

    # Simule: bu yeni threshold ile kac alert olusurdu?
    print(f"\n=== Simulasyon: yeni threshold ile kac alert? ===")
    for risk, lo, hi in THRESHOLDS[:3]:  # LOW dahil
        cur.execute("""
            SELECT COUNT(*) FROM aml_alert
            WHERE run_id='RUN-36D6BD0D' AND final_score >= %s AND final_score < %s
        """, (lo, hi))
        cnt = cur.fetchone()[0]
        print(f"  {risk:<8}: {cnt:>6} alert")

conn.close()
print("\nThreshold kalibrasyonu tamamlandi.")
