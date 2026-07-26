# 🚨 AML Entity Matching Motoru (Yapay Zeka Destekli)

Bu proje, Elektronik Fon Transferi (EFT) ve Swift açıklamalarındaki şüpheli firma, kişi ve varlıkları (Entity) tespit etmek amacıyla geliştirilmiş, **Çok Aşamalı Yapay Zeka (AI) ve Kural Tabanlı** kurumsal bir "Anti-Money Laundering" (Kara Para Aklamanın Önlenmesi - AML) eşleştirme motorudur.

Geleneksel anahtar kelime eşleştirme sistemlerinden farklı olarak; bulanık mantık (fuzzy logic), anlamsal vektör aramaları (semantic search), derin öğrenme tabanlı çapraz kodlayıcılar (cross-encoder) ve siber atlatma (evasion) algılama algoritmaları kullanarak tipografik hataları, kısaltmaları, kelime oyunlarını ve bağlamı insan düzeyinde anlayarak yüksek doğrulukta çalışır.

---

## 🏗️ Sistem Mimarisi ve Bileşenler

Sistem, verinin veritabanına girmesinden analist ekranında görüntülenmesine ve denetim loglarına işlenmesine kadar aşağıdaki katmanlardan oluşur:

### 1. Veri Hazırlığı, Ön İşleme ve Siber Koruma (Pre-processing & Evasion Detection)
- **Metin Temizleme ve Standardizasyon:** Gelen EFT açıklamalarındaki noktalama işaretleri boşluğa çevrilir ve küçük harfe dönüştürülür. Şirket yapı ekleri (`ltd.`, `corp.`, `inc`, `a.s.`, `sanayi`, `ticaret`) ortak standarda oturtulması için açık hallerine (`limited`, `corporation`, `incorporated`) çevrilir veya çekirdek isim (core name) ayrıştırılır (`src/utils/text_utils.py`).
- **Siber Atlatma (Leetspeak Evasion) Algılama:** Kötü niyetli aktörlerin yaptırımlı varlıkları gizlemek için rakam ve özel semboller kullanmasını yakalar (Örn: `M!cr0s0ft C0rp0r4t!0n`, `4ppl3 Inc`, `0r4cl3 Systems`). Harfler otomatik olarak normale döndürülür (`microsoft corporation`) ve manipülasyon tespit edildiğinde `leetspeak_evasion = True` bayrağı açılarak risk skoru doğrudan en üst seviyeye tetiklenir.
- **KVKK / PII Veri Maskeleme:** Metin içindeki kişisel veriler (IBAN, T.C. Kimlik No, Pasaport Numarası, IPv4 adresi) `src/utils/masking.py` ile otomatik maskelenerek (`[IBAN]`, `[TC_ID]`) yapay zekanın ilgisiz rakamlara takılması ve yanlış alarm (false positive) üretmesi engellenir.
- **Varyant Üretimi:** İzleme listesine (Watchlist) alınan şirketlerin isimlerinden otomatik varyantlar üretilir (`APPLE INC` -> `APPLE INCORPORATED`, `APLE INC`, `APPLE`).

### 2. Çok Kanallı Hibrit Aday Getirme (Multi-Channel Retrieval - Faz 1)
140,000+ izleme listesi satırı arasında milisaniyeler içinde en alakalı adayı bulmak için PostgreSQL üzerinde 3 kanallı paralel arama yapılır:
- **PostgreSQL Trigram (`pg_trgm`):** Harf tabanlı dizilim benzerliklerini bulur. Yazım ve OCR hatalarını yakalamada etkilidir (`min_trgm_score >= 0.20`).
- **Full Text Search (FTS):** Kelime köklerine ve kelime sırası değişikliklerine (Word Order Invariance) odaklanır (`to_tsvector`).
- **Semantic Search (PGVector):** Metinlerin anlamsal uzaydaki yerini bularak kelimeler farklı olsa dahi benzerlikleri yakalar. Embedding modeli olarak `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768 boyutlu cosine distance) kullanılır.

*Pipeline Durum Loglama:* Her işleme arama sonucuna göre `CANDIDATES_FOUND`, `TRIGRAM_NO_RESULT`, `VECTOR_NO_RESULT` veya `ALL_RETRIEVAL_CHANNELS_EMPTY` gibi net statü bayrakları atanır.

### 3. Varlık Çıkarımı (Entity Extraction - NER)
EFT açıklamasının tamamını karşılaştırmak yerine metnin içindeki asıl hedef tüzel kişi adını ayrıştırmak için NER (Named Entity Recognition) modeli çalıştırılır.
- **Model:** Türkçe finansal metinler için ince ayar yapılmış `savasy/bert-base-turkish-ner-cased` modeli kullanılır.
- **Fallback (Yedek) Mekanizması:** Yapay zeka bir varlık ayrıştıramazsa, metnin içinde aday şirket isminin geçip geçmediği kontrol edilir (`FALLBACK_MATCHED_VARIANT`).

### 4. Yeniden Sıralama ve Akıllı Önbellek (Reranking & Caching - Faz 2)
İlk aşamadan gelen en güçlü 10 aday (`reranker_top_k = 10`), çıkarılan varlık ile ikili (`[Açıklama, Aday Şirket]`) olarak kıyaslanır.
- **Cross-Encoder Model:** `BAAI/bge-reranker-v2-m3` kullanılarak 0 ile 1 arasında hassas bir semantik olasılık puanı üretilir.
- **Akıllı Önbellek (Reranker Cache):** Reranker hesaplaması yoğun GPU/CPU harcadığı için her işlem çiftinin skoru SHA256 özetiyle `aml_ml.reranker_cache` tablosuna loglanır. Bir sonraki çalışmada aynı çift gelirse model çalıştırılmaz, skor milisaniyeler içinde cache logundan okunur (`cache_hit = True`).

### 5. Final Skorlama ve Açıklanabilirlik (Ensemble Scoring & Explainability)
Tek bir model yerine 1440 işlem üzerinde simüle edilmiş Ağırlık Optimizasyonu (Weight Optimizer) çıktısına dayalı hibrit formül uygulanır:

$$\text{Final Score} = (0.50 \times \text{Vector}) + (0.40 \times \text{Reranker}) + (0.10 \times \text{Fuzzy})$$

- **Belirsiz Kısa İsim Koruması (`_AMBIGUOUS_SHORT_NAMES`):** Tek kelimelik genel kurumsal kelimeler (`apple`, `oracle`, `star`, `global`, `trust`, `prime`) tam eşleşse dahi otomatik olarak 1.0 skoruna yükseltilmez; yapay zekanın ensemble sonucu korunur (False-positive engelleme).
- **Kural Tabanlı Yaptırımlar (Overrides):**
  - *Tam Eşleşme (Exact Match):* Skor **1.0** yapılır.
  - *Kök Eşleşmesi (Compact Core Match):* Skor en az **0.95** yapılır.
  - *Sadece Şirket Türü Farkı (Legal Suffix):* Skor en az **0.92** yapılır.
- **Açıklanabilir Karar Sistemi (Reason Codes & Explanations):** Yapay zeka verdiği her karar için veritabanına JSON formatında Neden Kodları (`EXACT_MATCH`, `LEETSPEAK_EVASION_DETECTED`, `TYPO_DETECTED`, `ACRONYM_MATCH`) ve denetçiler için Türkçe İnsani Açıklama (`human_explanation`) loglar.

### 6. Karar ve Eşik Değerler (Thresholding)
Eşik Optimizasyonu (Threshold Optimizer) sonucuna göre belirlenen kurumsal risk katmanları (`aml_config.yaml`):
- `HIGH` ($\ge$ 0.65): Acil inceleme ve işlem blokajı gerektiren yüksek riskli eşleşme.
- `MEDIUM` (0.45 - 0.65): Analist incelemesine (Analyst Review) sevk edilen şüpheli veya eksik bilgili kısmi eşleşme.
- `LOW / NO_MATCH` ($<$ 0.45): Temiz işlem. Alarm üretilmez, sadece denetim izi için ana tabloya loglanır.

---

## 🛠️ Veritabanı Mimarisi ve Denetim (Audit / MLOps) Altyapısı

Sistem, Medallion mimarisine uygun olarak 8 özelleştirilmiş şema üzerinde çalışır ve tam MASAK/BDDK denetlenebilirliği (Auditability) sağlar:

1. **`aml_source` / `public.bronze_eft_raw`:** Ham EFT transfer açıklamaları katmanı.
2. **`aml_stage.company_variant`:** İzleme listesindeki şirketlerin orijinal adları ve normalize varyasyonları.
3. **`aml_ml`:** 768 boyutlu `pgvector` embedding vektörleri (`company_embedding`) ve model önbellekleri (`reranker_cache`).
4. **`aml_core`:**
   - `match_result`: İşlenen **her bir işlemin** (HIGH, MEDIUM, LOW, NO_MATCH fark etmeksizin) eksiksiz denetim izini saklayan ana kütüphane.
   - `alert`: Sadece **HIGH** ve **MEDIUM** riskli operasyonel alarm kayıtları.
   - `alert_export`: UI (Dashboard) ve BI araçları için hızlı flat (düzleştirilmiş) raporlama tablosu.
5. **`aml_audit`:**
   - `run_log`: Her pipeline partisinin başlangıç/bitiş zamanı, işlenen satır sayısı, alarm dağılımı, P50/P95/P99 milisaniye gecikme süreleri, model versiyonları ve Git commit hash sürümünü loglar.
   - `performance_log` & `quality_check_result`: Sistem performansı ve veri kalitesi denetim sonuçları.
   - `alert_status_history`: Alarmların analist incelemesi sonrası statü değişiklik tarihçesi.

---

## 📊 Analist Arayüzü (Streamlit Dashboard)

AML Analistlerinin üretilen alarmları inceleyip aksiyon alabilmeleri için geliştirilmiş interaktif yönetim panelidir.
Çalıştırmak için: `streamlit run src/ui/dashboard.py`

**Özellikleri:**
- **Ana Sayfa (Canlı Sandbox Test):** Analistlerin anlık olarak bir EFT açıklaması yazarak sistemin hangi skorla hangi şirketi bulacağını, neden kodlarını ve Türkçe açıklamaları test ettiği ortamdı.
- **Run Detayları & KPI Metrikleri:** İşlenen girdi, bulunan aday, HIGH/MEDIUM/LOW alarm dağılımları ve P50/P95/P99 gecikme metrikleri.
- **Alert İnceleme Yönetimi:** Analistin alarmı inceleyip durumunu (`OPEN`, `IN_REVIEW`, `CONFIRMED_MATCH`, `FALSE_POSITIVE`, `ESCALATED`, `CLOSED`) değiştirebildiği, inceleyen adını ve notunu girebildiği operasyon paneli. Değişiklikler anlık olarak veritabanına işlenir.

---

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimler
- Python 3.9+
- PostgreSQL 15+ (`pg_trgm` ve `pgvector` eklentileri zorunludur)

### 2. Ortam Hazırlığı
```bash
pip install -r requirements.txt
```
PostgreSQL bağlantı ayarlarınız için `config/aml_config.yaml` dosyasındaki veritabanı bilgilerini güncelleyin.

### 3. Veritabanı Kurulumu ve Şema Göçleri
Sistemi sıfırdan kurmak veya tabloları güncel şemaya taşımak için:
```bash
python scripts/migrate_db.py
```

### 4. Önceki İşlem ve Önbellek Temizliği (Opsiyonel)
Yeni bir test çalışması öncesi eski alarm ve önbellek kayıtlarını sıfırlamak için:
```bash
python scripts/clear_results.py
```

### 5. Toplu Pipeline Çalıştırma (Batch Inference)
EFT verilerini yapay zeka motorundan geçirip veritabanına loglamak için:
```bash
python src/app/main.py --input-table test_eft_input
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
├── config/                     # aml_config.yaml ayarları ve db_tables.py şema tanımları
├── data/                       # Girdi CSV veri setleri ve örnek dosyalar
├── sql/                        # Veritabanı şema kurulumları ve güncellemeler (Migrations 01-22)
├── src/
│   ├── app/                    # Uygulama başlangıç noktası (main.py) ve dependency container
│   ├── evaluation/             # Model ve ağırlık optimizasyon değerlendirme araçları
│   ├── models/                 # Model sarmalayıcıları (NER, Reranker, Embedding)
│   ├── pipeline/               # Batch processor ve inference orchestration
│   ├── repository/             # Veritabanı (PostgreSQL) CRUD işlemleri ve bağlantı havuzu
│   ├── retrieval/              # Vektör, Trigram ve Full-Text arama motorları
│   ├── scoring/                # Skor hesaplama, hibrit ensemble, Neden Kodları ve açıklanabilirlik
│   ├── ui/                     # Streamlit tabanlı Analist Arayüzü (Dashboard)
│   └── utils/                  # Metin temizleme, Leetspeak algılama ve PII maskeleme
│
├── scripts/                    # Şema taşıma, veritabanı temizleme ve veri yükleme betikleri
├── tests/                      # 137 unit ve integration test suite (pytest)
└── requirements.txt            # Python bağımlılıkları
```

## 🔒 Güvenlik ve Gizlilik
- Model işlemleri ve Embedding hesaplamaları tamamen yerel (on-premise) çalışacak şekilde tasarlanmıştır. HuggingFace modelleri bilgisayarınıza/sunucunuza indirilir ve hiçbir EFT datası veya müşteri bilgisi dış internet (Cloud API) ortamına gönderilmez.
- Veritabanında hassas işlemler (Audit Trail) geriye dönük değiştirilemez şekilde (append-only mantığına yakın) tasarlanmıştır.