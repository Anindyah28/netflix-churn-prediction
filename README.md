# 🎬 Netflix Customer Churn & Engagement Analysis

Analisis eksploratif (EDA) terhadap data 5.000 pelanggan layanan streaming untuk memahami faktor-faktor yang memengaruhi **churn** (berhentinya pelanggan berlangganan).

![Preview Analysis](assets/preview_banner.png)

## 📌 Latar Belakang

Mempertahankan pelanggan lebih murah dibanding mendapatkan pelanggan baru. Proyek ini menjawab pertanyaan: **"Pelanggan seperti apa yang cenderung berhenti berlangganan, dan sinyal apa yang bisa dideteksi lebih awal?"**

## 📂 Dataset

[Netflix Customer Churn & Engagement Analytics](https://www.kaggle.com/datasets/zeyadmohamed26/netflix-customer-churn-and-engagement-analytics) — 5.000 baris, 14 kolom, mencakup profil pelanggan (usia, gender, region), perilaku engagement (jam nonton, aktivitas login), dan atribut bisnis (tipe subscription, metode pembayaran).

## 🔍 Pertanyaan Analisis

1. Berapa persen pelanggan yang churn?
2. Apakah tipe subscription, metode pembayaran, atau device memengaruhi churn?
3. Apakah pola engagement (jam nonton, recency login) berhubungan dengan churn?
4. Faktor apa yang paling kuat berkorelasi dengan churn?

## 📊 Temuan Utama

| Temuan | Detail |
|---|---|
| Churn rate keseluruhan | ~50,3% (dataset seimbang) |
| Prediktor terkuat | `last_login_days` (corr +0.47) & `watch_hours` (corr -0.48) |
| Subscription paling rentan | Basic → 61,8% churn (vs Premium 43,7%) |
| Metode pembayaran paling rentan | Crypto (59,7%) & Gift Card (57,8%) vs Kartu Kredit/Debit (~44%) |
| Faktor tidak signifikan | Usia, gender, region, device |

**Insight kunci:** Rendahnya engagement (jarang nonton & lama tidak login) adalah sinyal *early-warning* churn yang jauh lebih kuat dibanding faktor demografis.

## 💡 Rekomendasi Bisnis

- Bangun sistem deteksi dini: tandai pelanggan dengan `last_login_days` > 30 hari sebagai "berisiko churn" untuk kampanye re-engagement.
- Fokuskan program retensi pada segmen paket **Basic**.
- Dorong migrasi metode pembayaran non-recurring (Crypto/Gift Card) ke kartu dengan auto-renewal.

## 🤖 Bagian 2: Model Prediksi Churn

Sebagai kelanjutan dari EDA, dibangun model machine learning untuk memprediksi churn pelanggan (lihat `netflix_churn_prediction.ipynb`).

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 88,7% | 87,5% | 90,5% | 88,9% | 0,966 |
| **Random Forest** | **97,3%** | **97,0%** | **97,6%** | **97,3%** | **0,996** |

**Feature importance teratas:** `avg_watch_time_per_day`, `watch_hours`, `last_login_days` — konsisten dengan temuan EDA bahwa engagement adalah sinyal churn paling dominan.

> Catatan: performa model yang sangat tinggi kemungkinan karena dataset ini disintesis dengan hubungan fitur-target yang relatif bersih. Pada data dunia nyata, performa umumnya lebih rendah.

## 🛠️ Tools

- Python, Pandas — data cleaning & manipulasi
- Matplotlib, Seaborn — visualisasi
- Scikit-learn — machine learning (Logistic Regression, Random Forest)
- Jupyter Notebook

## 🌐 Bagian 3: Aplikasi Web Interaktif (Streamlit)

Model Random Forest yang sudah dilatih dibungkus jadi aplikasi web sederhana menggunakan **Streamlit**, sehingga siapa pun bisa input data pelanggan lewat form dan langsung melihat prediksi churn-nya — tanpa perlu membuka notebook atau menulis kode.

**Coba jalankan secara lokal:**
```bash
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi akan terbuka di browser (`http://localhost:8501`). Isi form data pelanggan (usia, tipe subscription, watch hours, dll), lalu klik **"Prediksi Churn"** untuk melihat hasilnya beserta probabilitasnya.

> 💡 File `churn_model.pkl`, `label_encoders.pkl`, dan `feature_columns.pkl` adalah artefak model yang sudah dilatih — tidak perlu training ulang untuk menjalankan app ini.

## 📁 Struktur Repo

```
├── app.py                           # Aplikasi web Streamlit
├── churn_model.pkl                  # Model Random Forest yang sudah dilatih
├── label_encoders.pkl               # Encoder untuk fitur kategorikal
├── feature_columns.pkl              # Urutan kolom fitur
├── netflix_churn_analysis.ipynb     # Bagian 1: EDA (kode + visualisasi + insight)
├── netflix_churn_prediction.ipynb   # Bagian 2: Model prediksi churn (training + evaluasi)
├── netflix_customer_churn.csv       # Dataset
├── requirements.txt
└── README.md
```

## 🚀 Cara Menjalankan

```bash
pip install -r requirements.txt

# Jalankan notebook EDA
jupyter notebook netflix_churn_analysis.ipynb

# Jalankan notebook model prediksi
jupyter notebook netflix_churn_prediction.ipynb

# Jalankan aplikasi web interaktif
streamlit run app.py
```

## 📈 Langkah Lanjutan

- Deploy aplikasi ke **Streamlit Community Cloud** agar bisa diakses publik lewat link, tanpa perlu di-run lokal.
- Hyperparameter tuning (GridSearchCV) untuk optimasi lebih lanjut.
- Cross-validation untuk memastikan model tidak overfitting.
