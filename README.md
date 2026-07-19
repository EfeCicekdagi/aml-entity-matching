# 🚨 AML Entity Matching Motoru (Yapay Zeka Destekli)

Bu proje, Elektronik Fon Transferi (EFT) ve Swift açıklamalarındaki şüpheli firma, kişi ve varlıkları (Entity) tespit etmek amacıyla geliştirilmiş, **Çok Aşamalı Yapay Zeka (AI) ve Kural Tabanlı** bir "Anti-Money Laundering" (Kara Para Aklamanın Önlenmesi - AML) eşleştirme motorudur. 

Geleneksel anahtar kelime eşleştirme sistemlerinden farklı olarak, bulanık mantık (fuzzy logic), anlamsal vektör aramaları (semantic search) ve çapraz kodlayıcılar (cross-encoder) kullanarak tipografik hataları, kısaltmaları, kelime oyunlarını ve bağlamı anlayarak yüksek doğrulukta çalışır.

---

## 🏗️ Sistem Mimarisi ve Bileşenler

Sistem, verinin veritabanına girmesinden analist ekranında görüntülenmesine kadar aşağıdaki katmanlardan oluşur:

### 1. Veri Hazırlığı ve Ön İşleme (Pre-processing)
- **Metin Temizleme:** Gelen EFT açıklamalarındaki gereksiz noktalama işaretleri, özel karakterler, rakamlar ve bankacılık terimleri (Örn: "TRANSFER", "IBAN") normalize edilir.
- **Varyant Üretimi:** Kara listeye (Blacklist) alınan şirketlerin isimlerinden otomatik olarak varyantlar üretilir (Örn: `APPLE INC` -> `APPLE INCORPORATED`, `APLE INC`, `APPLE`). Bu varyantlar veritabanına kaydedilir ve vektörleri (`pgvector` ile) oluşturulur.

### 2. Aday Çıkarma (Candidate Retrieval - Faz 1)
Bir EFT işleminde geçen ismin kara listeyle potansiyel eşleşmelerini bulmak için 3 farklı indeks mekanizması birlikte (ensemble) kullanılır. Bu aşamanın amacı yüksek "Recall" (duyarlılık) sağlamaktır.
- **PostgreSQL Trigram (pg_trgm):** Harf tabanlı dizilim benzerliklerini bulur. Yazım hatalarını yakalamada etkilidir.
- **Full Text Search (FTS):** Kelime köklerine ve terim eşleşmelerine odaklanır.
- **Semantic Search (PGVector):** Metinlerin anlamsal uzaydaki yerini bularak, kelimeler farklı olsa bile (kısaltmalar, eşanlamlılar vb.) benzerlikleri yakalar. Model olarak `BAAI/bge-m3` embedding modeli kullanılır.

*Optimizasyon:* Sistem, performansı artırmak için işlem başına `Pre-Screening` (Ön eleme) uygular. Eğer Trigram taramasında belirli bir barajı aşan tek bir şirket bile yoksa, işlem doğrudan "Güvenli" (Clean) sayılarak ağır yapay zeka modellerine girmeden sonlandırılır.

### 3. Varlık Çıkarımı (Entity Extraction)
EFT açıklamasının tamamını şirketin adıyla karşılaştırmak yerine, metnin içindeki asıl "İsim" kısmını bulmak için **NER (Named Entity Recognition)** modeli çalıştırılır.
- **Model:** Türkçe metinler için ince ayar (fine-tune) edilmiş `savasy/bert-base-turkish-ner-cased` modeli kullanılır.
- **Fallback (Yedek) Mekanizması:** Eğer yapay zeka bir şirket ismi bulamazsa, metnin içinde aday şirket isminin tam olarak geçip geçmediği kontrol edilir ve gerekiyorsa manuel eşleşme yapılır (`FALLBACK_MATCHED_VARIANT`).

### 4. Yeniden Sıralama ve Kesinleştirme (Reranking - Faz 2)
İlk aşamada (Faz 1) bulunan 20-30 aday şirket, çıkarılan Entity ile ikili olarak karşılaştırılır. Bu aşamada ağır ve çok daha zeki bir model çalışır.
- **Cross-Encoder Model:** `BAAI/bge-reranker-v2-m3` kullanılarak metinler arası en hassas benzerlik puanı (`reranker_score`) üretilir. Bu model tipografik hataları ve bağlamı insan düzeyinde anlayabilir.

### 5. Final Skorlama (Ensemble Scoring)
Sadece yapay zeka modelinin sonucuna güvenmek yerine, kuruma özel yapılandırılabilir bir skorlama konfigürasyonu (`scoring_config.yaml`) ile nihai bir "Risk Skoru" hesaplanır:
- Reranker Skoru (Ağırlık: %45)
- Vektör Skoru (Ağırlık: %20)
- Trigram Skoru (Ağırlık: %20)
- Fuzzy Skor (Levenshtein) (Ağırlık: %15)
- Tam Eşleşme, Kısaltma gibi özel durumlarda skora kural tabanlı bonuslar / cezalar eklenir.

### 6. Karar ve Eşik Değerler (Thresholding)
Hesaplanan `final_score`'a göre alarmın (alert) risk seviyesi belirlenir (`thresholds.yaml`):
- `HIGH`: > 0.70 (Yüksek ihtimalli eşleşme)
- `MEDIUM`: 0.62 - 0.70 (İncelenmesi gereken şüpheli eşleşme)
- `LOW`: < 0.62 (Düşük riskli, genellikle elenir)

---

## 🛠️ Veritabanı ve Denetim (Audit) Altyapısı
Sistem kurumsal AML standartlarına uygun şekilde "Auditability" (Denetlenebilirlik) sağlamak üzere tasarlanmıştır.

- **Run Logları (`aml_audit.run_log`):** Her bir pipeline çalışması benzersiz bir `run_id` alır. Kullanılan Pipeline Versiyonu, Reranker Modeli, Embedding Modeli, Skorlama ve Threshold konfigürasyon versiyonları bu tabloya yazılır. İleride "Neden bu alarm üretilmedi?" sorusuna tam geriye dönük cevap verilebilir.
- **Alert Tablosu (`aml_core.alert`):** Üretilen her alarm; hangi EFT kaydından üretildiği, hangi varyanta eşleştiği, o anki varyantın tam adı, NER işlem durumu (`entity_extraction_status`), tüm alt skorlar ve nihai risk skoru ile kaydedilir.

---

## 📊 Analist Arayüzü (Streamlit Dashboard)
AML Analistlerinin üretilen alarmları inceleyip aksiyon alabilmeleri için geliştirilmiş interaktif paneldir.
Çalıştırmak için: `streamlit run src/ui/dashboard.py`

**Özellikleri:**
- **Ana Sayfa (Canlı Test):** Analistlerin anlık olarak bir EFT açıklamasını yazarak sistemin hangi skorla hangi şirketi bulacağını test ettiği canlı "Sandbox" ortamıdır.
- **Run Detayları:** Geçmiş toplu veri taramalarının (Batch Runs) detayları.
  - *KPI Metrikleri:* İşlenen Girdi, Bulunan Aday, HIGH/MEDIUM/LOW alarm sayıları.
  - *Konfigürasyon İzlenebilirliği:* O çalışmada hangi model ve konfigürasyonların kullanıldığı (Versiyonlar).
  - *Dağılım Analizi:* Skorların histogram dağılım grafikleri.
  - *Alert İnceleme Yönetimi:* Analistin alarmı detaylıca inceleyip, durumunu (`OPEN`, `IN_REVIEW`, `CONFIRMED_MATCH`, `FALSE_POSITIVE`, `ESCALATED`, `CLOSED`) değiştirebildiği, inceleyen adını ve analiz notunu girebildiği yönetim paneli. Değişiklikler anlık olarak veritabanına işlenir.

---

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimler
- Python 3.9+
- PostgreSQL 15+ (Aşağıdaki eklentiler zorunludur)
  - `pg_trgm` (Trigram aramaları için)
  - `pgvector` (Vektörel aramalar için)

### 2. Ortam Hazırlığı
```bash
pip install -r requirements.txt
```
PostgreSQL yapılandırması için `src/config/db_config.yaml` dosyasını kendi veritabanı bilgilerinize göre güncelleyin.

### 3. Veritabanı Göçleri (Migrations)
Sistemi sıfırdan kurmak veya güncellemek için:
```bash
python scripts/migrate_db.py
```

### 4. Örnek Veri Yükleme (Opsiyonel)
Kara listeye şirket eklemek ve vektörlerini oluşturmak için:
```bash
python scripts/load_company_data.py
```

### 5. Pipeline'ı Çalıştırma
Bir tablodaki EFT işlemlerini toplu olarak AML motorundan geçirmek için:
```bash
python src/main.py --input-table aml_source.test_eft_input
```

### 6. Analist Arayüzünü Başlatma
```bash
streamlit run src/ui/dashboard.py
```

---

## 📁 Proje Dizin Yapısı

```
aml-entity-matching/
│
├── sql/                        # Veritabanı tablo kurulumları ve güncellemeler (Migrations)
├── src/
│   ├── config/                 # YAML konfigürasyonları (db, scoring, thresholds)
│   ├── etl/                    # Toplu veri işleme ve akış yönetimi (Batch Processor)
│   ├── models/                 # Model ve Veritabanı sınıfları
│   ├── repository/             # Veritabanı (PostgreSQL) CRUD işlemleri
│   ├── reranker/               # Cross-Encoder (Reranker) entegrasyonları
│   ├── retrieval/              # Vektör, Trigram ve Full-Text arama motorları
│   ├── scoring/                # Skor hesaplama ve kural motorları (Ensemble)
│   ├── ui/                     # Streamlit tabanlı Analist Arayüzü (Dashboard)
│   └── utils/                  # NER, Metin temizleme (Text Utils) ve Logger yardımcıları
│
├── scripts/                    # Test verisi yükleme, veritabanı temizleme betikleri
├── tests/                      # Birim (Unit) testleri
└── requirements.txt            # Python bağımlılıkları
```

## 🔒 Güvenlik ve Gizlilik
- Model işlemleri ve Embedding hesaplamaları tamamen yerel (on-premise) çalışacak şekilde tasarlanmıştır. HuggingFace modelleri bilgisayarınıza/sunucunuza indirilir ve hiçbir EFT datası veya müşteri bilgisi internet (Cloud API) ortamına gönderilmez.
- Veritabanında hassas işlemler (Audit Trail) geriye dönük değiştirilemez şekilde (append-only mantığına yakın) tasarlanmıştır.