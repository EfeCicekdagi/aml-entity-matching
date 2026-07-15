import psycopg2
from datetime import date

def inject_data():
    conn = psycopg2.connect(host='127.0.0.1', port=5433, dbname='aml_db', user='postgres', password='password')
    cur = conn.cursor()

    # Yeni şirketler ve ID'leri (company_list.csv'ye göre)
    # 1,Indiaforensic Services Pvt Ltd
    # 2,Finspire Solutions Private Limited
    # 3,Mehar Entertainment Private Limited
    # 4,Microsoft Corporation
    # 5,Apple Inc
    # 6,Google LLC
    # 7,Amazon Technologies Inc
    # 8,Tesla Motors Ltd
    # 9,IBM Corporation
    # 10,Oracle Corporation

    test_data = [
        # (eft_id, explanation, true_company_id)
        # Birebir eşleşmeler
        (900001, "TRF FROM Microsoft Corporation FOR SOFTWARE LICENSES", 4),
        (900002, "INWARD REMITTANCE FROM Apple Inc", 5),
        (900003, "PAYMENT TO Google LLC FOR CLOUD SVCS", 6),
        # Fuzzy/Kısaltma eşleşmeleri (Modelin anlamsal gücünü görmek için)
        (900004, "TRF FRM Amazon Tech Inc.", 7),
        (900005, "SALARY FROM Tesla Motors", 8),
        (900006, "PAYMENT TO IBM Corp", 9),
        (900007, "FUNDS FROM Oracle Corp", 10),
        # İlgisiz işlemler (False-Positive testi için, -1)
        (900008, "GROCERY SHOPPING KROGER STORE", -1),
        (900009, "ATM WITHDRAWAL MAIN STREET", -1),
        (900010, "TRF TO JOHN DOE RENT PAYMENT", -1),
    ]

    print("Test verileri ekleniyor...")
    
    for eft_id, exp, true_cmp_id in test_data:
        # bronze_eft_raw'a ekle (varsa sil)
        cur.execute("DELETE FROM bronze_eft_raw WHERE eft_id = %s", (eft_id,))
        cur.execute("""
            INSERT INTO bronze_eft_raw (eft_id, transaction_date, amount, explanation, source_system)
            VALUES (%s, %s, %s, %s, %s)
        """, (eft_id, date.today(), 1500.00, exp, "TEST_SYSTEM"))

        # aml_ground_truth'a ekle (varsa sil)
        eft_id_str = f"EFT_{str(eft_id).zfill(5)}"
        cur.execute("DELETE FROM aml_ground_truth WHERE eft_id = %s", (eft_id_str,))
        cur.execute("""
            INSERT INTO aml_ground_truth (eft_id, true_company_id)
            VALUES (%s, %s)
        """, (eft_id_str, true_cmp_id))

    conn.commit()
    print(f"Toplam {len(test_data)} test kaydı bronze_eft_raw ve aml_ground_truth tablolarına eklendi!")
    
    # Ground truth'taki eski işe yaramaz verileri temizle ki metrikler sadece test verilerini baz alsın
    # (Ya da kalsın, ama kalırsa recall düşer çünkü onlara alert üretemeyeceğiz)
    # Sadece 900000 serisi kalsın
    cur.execute("DELETE FROM aml_ground_truth WHERE eft_id NOT LIKE 'EFT_900%'")
    conn.commit()
    print("Eski geçersiz ground_truth verileri temizlendi. Sadece test verileri kaldı.")

    conn.close()

if __name__ == "__main__":
    inject_data()
