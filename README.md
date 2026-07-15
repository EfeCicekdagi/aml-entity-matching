# Akıllı AML Varlık Eşleştirme Sistemi (AML Entity Matching)

Bu proje, kurum içindeki para transferi (EFT/Havale) işlemlerinde yer alan alıcı/gönderici isimlerini yasaklı ve şüpheli kişi/kurum listeleriyle eşleştirmek için geliştirilmiş **Yapay Zeka destekli (AI-driven)** bir Anti-Money Laundering (AML) eşleştirme sistemidir.

Geleneksel ve kural tabanlı sistemlerin aksine, bu sistem **Doğal Dil İşleme (NLP)** ve **Vektörel Benzerlik (Vector Search)** teknolojilerini kullanarak sadece birebir kelime eşleşmelerine değil, **anlamsal yakınlığa (Semantic Search)** da odaklanır. Böylece yanlış alarmları (False-Positive) ciddi oranda düşürerek operasyonel eforu azaltır.

---

## 🚀 Projenin Yönetici Özeti ve İş Değeri (Business Value)

- **Problem:** Mevcut sistemlerde harf hataları, kısaltmalar veya farklı dillerdeki yazım varyasyonları nedeniyle şüpheli işlemler tespit edilememekte veya çok fazla hatalı alarm üretilmektedir.
- **Çözüm:** En güncel NLP modelleri (Hugging Face) ve vektör veritabanları (pgvector) ile güçlendirilmiş **çift aşamalı (Retrieval & Reranking)** bir doğrulama sistemi.
- **Kazanım (ROI):** 
  - Manuel inceleme sürelerinde ciddi düşüş.
  - "False-Positive" oranlarının azaltılması.
  - Saniyeler içinde on binlerce kaydı tarayabilen yüksek performans.

---

## 🛠 Kullanılan Teknolojiler (Tech Stack)

### Altyapı ve Veritabanı
- **Python:** Ana programlama dili (ETL, API, ML modelleri).
- **Docker & Docker Compose:** Servislerin izolasyonu ve ortam yönetimi.
- **PostgreSQL & pgvector:** Vektörel veri saklama ve K-Nearest Neighbors (KNN) aramaları.

### Yapay Zeka (AI) & Doğal Dil İşleme (NLP)
- **Vektörleştirme (Embeddings):** `intfloat/multilingual-e5-large`, `BAAI/bge-m3`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **İsim Çıkarımı (NER - Named Entity Recognition):** Türkçe destekli BERT (`savasy/bert-base-turkish-ner-cased`)
- **Yeniden Sıralama (Cross-Encoder / Reranker):** `BAAI/bge-reranker-v2-m3`

### Veri Bilimi ve Arayüz
- **Pandas & Scikit-Learn:** Veri işleme, model değerlendirme ve metrik hesaplama (Precision, Recall, F1).
- **Streamlit & Plotly:** Etkileşimli yönetim paneli (Dashboard) ve grafik görselleştirmeleri.

---

## 🏗 Sistem Mimarisi

Sistem 4 ana adımdan oluşmaktadır:
1. **ETL ve Veri Girişi:** Banka işlem kayıtlarının (EFT vb.) veritabanına aktarılması.
2. **Retrieval (Geri Getirme):** Veritabanındaki metinlerin vektörleri çıkarılarak (`pgvector`), hedef listesindeki en olası adayların saniyeler içinde getirilmesi.
3. **Scoring & Reranking (Skorlama):** Gelen adayların daha gelişmiş yapay zeka modelleriyle (Cross-Encoder) tekrar değerlendirilerek risk düzeylerinin (HIGH, MEDIUM, LOW) belirlenmesi.
4. **Dashboard:** Sonuçların ve şüpheli işlemlerin Streamlit tabanlı arayüz ile raporlanması.

---

## ⚙️ Kurulum ve Çalıştırma

### 1. Ön Gereksinimler
- Python 3.9 veya üzeri
- Docker Desktop (Veritabanı için zorunludur)
- Gerekli Python kütüphaneleri (Proje içindeki scriptler aracılığıyla kurulabilir)

### 2. Veritabanını Başlatma
Proje dizininde bir terminal açarak PostgreSQL veritabanını Docker üzerinden ayağa kaldırın:
```bash
docker-compose up -d
```

### 3. Pipeline'ı Çalıştırma
Ana modeli ve eşleştirme sürecini çalıştırmak için ana scripti kullanabilirsiniz:
```bash
python src/main.py
```
*(Farklı modelleri test etmek ve kıyaslamak için `scripts/evaluation/benchmark_models.py` dosyasını çalıştırabilirsiniz.)*

### 4. Yönetici Dashboard'unu Başlatma
Eşleşen sonuçları, risk dağılım grafiklerini ve detayları görüntülemek için arayüzü başlatın:
```bash
streamlit run src/ui/dashboard.py
```

---

## 📂 Dosya Yapısı (Project Structure)

```text
aml-entity-matching/
├── docker-compose.yml       # Veritabanı ve altyapı servisleri
├── config/
│   └── aml_config.yaml      # Model ve eşik değeri konfigürasyonları
├── data/                    # Örnek ve test veri setleri
├── outputs/                 # Çıktılar, benchmark raporları (CSV, MD)
├── scripts/
│   ├── data/                # Veri üretme scriptleri
│   └── evaluation/          # Model benchmark ve kalibrasyon testleri
├── sql/                     # Veritabanı tablo ve pgvector yapılandırmaları
└── src/
    ├── main.py              # Ana çalışma scripti
    ├── etl/                 # Veri işleme ve aktarım boru hattı (Pipeline)
    ├── repository/          # Veritabanı bağlantı modülleri
    ├── reranker/            # Cross-encoder yapay zeka modelleri
    ├── retrieval/           # Aday arama (KNN Vector Search) mekanizması
    ├── scoring/             # Risk seviyesi belirleme logiği
    ├── ui/                  # Streamlit tabanlı yönetim paneli
    └── utils/               # Yardımcı (Helper) fonksiyonlar
```