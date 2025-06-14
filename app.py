import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ===============================
# Load model dan scaler
# ===============================
model = joblib.load('best_model_rf.pkl')
scaler = joblib.load('scaler.pkl')

# ===============================
# Judul aplikasi
# ===============================
st.title("Prediksi Tingkat Obesitas")
st.write("Aplikasi ini memprediksi tingkat obesitas berdasarkan data input pengguna.")

# ===============================
# Input
# ===============================
gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
gender = 1 if gender == "Male" else 0

age = st.number_input("Usia", 10, 100, 25)
height = st.number_input("Tinggi Badan (cm)", 50, 250, 170) / 100
weight = st.number_input("Berat Badan (kg)", 10, 300, 65)

family_history = st.selectbox("Riwayat keluarga dengan obesitas?", ["Yes", "No"])
family_history = 1 if family_history == "Yes" else 0

favc = st.selectbox("Konsumsi makanan tinggi kalori?", ["Yes", "No"])
favc = 1 if favc == "Yes" else 0

fcvc = st.slider("Frekuensi makan sayur (1–3)", 1.0, 3.0, 2.0)
ncp = st.slider("Jumlah makan utama per hari (1–4)", 1.0, 4.0, 3.0)

caec = st.selectbox("Frekuensi ngemil / fast food", ["no", "Sometimes", "Frequently", "Always"])
caec = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}[caec]

smoke = st.selectbox("Merokok?", ["Yes", "No"])
smoke = 1 if smoke == "Yes" else 0

ch2o = st.slider("Minum air harian (liter)", 1.0, 3.0, 2.0)

scc = st.selectbox("Kontrol kalori?", ["Yes", "No"])
scc = 1 if scc == "Yes" else 0

faf = st.slider("Aktivitas fisik mingguan (0–3)", 0.0, 3.0, 1.0)
tue = st.slider("Durasi penggunaan teknologi (0–3 jam)", 0.0, 3.0, 1.0)

calc = st.selectbox("Konsumsi alkohol", ["Never", "Rarely", "Frequently", "Always"])
calc = {"Never": 0, "Rarely": 1, "Frequently": 2, "Always": 3}[calc]

mtrans = st.selectbox("Transportasi", ["Walking", "Public_Transportation", "Automobile", "Bike", "Motorbike"])
mtrans = {"Walking": 0, "Public_Transportation": 1, "Automobile": 2, "Bike": 3, "Motorbike": 4}[mtrans]

# ===============================
# Susun input sesuai urutan fitur saat training
# ===============================
input_data = pd.DataFrame([[
    age, gender, height, weight,
    calc, favc, fcvc, ncp,
    scc, smoke, ch2o, family_history,
    faf, tue, caec, mtrans
]], columns=[
    'Age', 'Gender', 'Height', 'Weight',
    'CALC', 'FAVC', 'FCVC', 'NCP',
    'SCC', 'SMOKE', 'CH2O', 'family_history_with_overweight',
    'FAF', 'TUE', 'CAEC', 'MTRANS'
])

# ===============================
# Prediksi
# ===============================
if st.button("Prediksi"):
    try:
        input_scaled = scaler.transform(input_data.values)
        prediction = model.predict(input_scaled)

        kategori = {
            0: "Berat Badan Kurang (Insufficient Weight)",
            1: "Berat Badan Normal (Normal Weight)",
            2: "Kelebihan Berat Badan Tingkat I (Overweight Level I)",
            3: "Kelebihan Berat Badan Tingkat II (Overweight Level II)",
            4: "Obesitas Tipe I (Obesity Type I)",
            5: "Obesitas Tipe II (Obesity Type II)",
            6: "Obesitas Tipe III (Obesity Type III)"
        }

        st.success(f"Hasil Prediksi: {kategori.get(prediction[0], 'Tidak diketahui')}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi:\n\n{str(e)}")
