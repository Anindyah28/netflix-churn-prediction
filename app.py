import streamlit as st
import pandas as pd
import joblib

# =========================
# Load model & artefak
# =========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, encoders, feature_columns

model, encoders, feature_columns = load_artifacts()

st.set_page_config(page_title="Netflix Churn Predictor", page_icon="🎬", layout="centered")

st.title("🎬 Netflix Customer Churn Predictor")
st.write(
    "Aplikasi ini memprediksi apakah seorang pelanggan berpotensi **berhenti berlangganan (churn)** "
    "berdasarkan profil dan perilaku menonton mereka, menggunakan model **Random Forest** "
    "(akurasi ~97% pada data uji)."
)

st.divider()
st.subheader("📋 Masukkan Data Pelanggan")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Usia", 18, 70, 30)
    gender = st.selectbox("Gender", encoders["gender"].classes_)
    subscription_type = st.selectbox("Tipe Subscription", encoders["subscription_type"].classes_)
    region = st.selectbox("Region", encoders["region"].classes_)
    device = st.selectbox("Device", encoders["device"].classes_)
    payment_method = st.selectbox("Metode Pembayaran", encoders["payment_method"].classes_)

with col2:
    favorite_genre = st.selectbox("Genre Favorit", encoders["favorite_genre"].classes_)
    monthly_fee = st.slider("Biaya Bulanan ($)", 8.99, 17.99, 12.99, step=0.01)
    number_of_profiles = st.slider("Jumlah Profil di Akun", 1, 5, 2)
    watch_hours = st.number_input("Total Watch Hours", min_value=0.0, max_value=200.0, value=15.0, step=0.5)
    avg_watch_time_per_day = st.number_input("Rata-rata Jam Nonton per Hari", min_value=0.0, max_value=24.0, value=1.0, step=0.1)
    last_login_days = st.slider("Hari sejak Login Terakhir", 0, 60, 10)

st.divider()

if st.button("🔍 Prediksi Churn", use_container_width=True, type="primary"):
    input_dict = {
        "age": age,
        "gender": encoders["gender"].transform([gender])[0],
        "subscription_type": encoders["subscription_type"].transform([subscription_type])[0],
        "watch_hours": watch_hours,
        "last_login_days": last_login_days,
        "region": encoders["region"].transform([region])[0],
        "device": encoders["device"].transform([device])[0],
        "monthly_fee": monthly_fee,
        "payment_method": encoders["payment_method"].transform([payment_method])[0],
        "number_of_profiles": number_of_profiles,
        "avg_watch_time_per_day": avg_watch_time_per_day,
        "favorite_genre": encoders["favorite_genre"].transform([favorite_genre])[0],
    }

    input_df = pd.DataFrame([input_dict])[feature_columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Hasil Prediksi")

    if prediction == 1:
        st.error(f"⚠️ Pelanggan ini **berpotensi CHURN** (probabilitas: {probability:.1%})")
    else:
        st.success(f"✅ Pelanggan ini **kemungkinan tetap AKTIF** (probabilitas churn: {probability:.1%})")

    st.progress(float(probability))

    st.caption(
        "Catatan: prediksi berdasarkan pola pada data historis. "
        "Faktor paling berpengaruh: rata-rata jam nonton per hari, total watch hours, dan lama tidak login."
    )

st.divider()
st.caption("Dibangun dengan Streamlit • Model: Random Forest • Dataset: Netflix Customer Churn (Kaggle)")
