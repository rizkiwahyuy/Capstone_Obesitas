import streamlit as st
import joblib
import numpy as np

# Load model dan scaler
model = joblib.load("best_random_forest_model.joblib")  # Model terbaik hasil tuning
scaler = joblib.load("scaler.joblib")  # Scaler yang digunakan saat training

# Judul dan deskripsi
st.title("Prediksi Tingkat Obesitas")
st.markdown("""
Aplikasi ini memprediksi tingkat obesitas berdasarkan data kebiasaan hidup dan faktor pribadi.  
Model yang digunakan adalah **Random Forest (Tuned)** dengan akurasi 91%.
""")

# Input pengguna
st.header("Masukkan Data Anda")

gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
gender = 1 if gender == "Male" else 0

age = st.number_input("Usia", min_value=10, max_value=100, value=25)
height = st.number_input("Tinggi Badan (cm)", min_value=50, max_value=250, value=170) / 100
weight = st.number_input("Berat Badan (kg)", min_value=10, max_value=300, value=70)

family_history = st.selectbox("Riwayat keluarga dengan obesitas?", ["Yes", "No"])
family_history = 1 if family_history == "Yes" else 0

favc = st.selectbox("Sering konsumsi makanan tinggi kalori?", ["Yes", "No"])
favc = 1 if favc == "Yes" else 0

fcvc = st.slider("Seberapa sering makan sayur?", 1.0, 3.0, 2.0)
ncp = st.slider("Frekuensi makan besar (per hari)?", 1.0, 4.0, 3.0)

caec = st.selectbox("Kebiasaan ngemil tidak sehat?", ["no", "Sometimes", "Frequently", "Always"])
caec_dict = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
caec = caec_dict[caec]

smoke = st.selectbox("Apakah merokok?", ["Yes", "No"])
smoke = 1 if smoke == "Yes" else 0

ch2o = st.slider("Konsumsi air per hari (liter)?", 1.0, 3.0, 2.0)

scc = st.selectbox("Apakah memantau asupan kalori?", ["Yes", "No"])
scc = 1 if scc == "Yes" else 0

faf = st.slider("Frekuensi aktivitas fisik?", 0.0, 3.0, 1.0)
tue = st.slider("Waktu dengan perangkat (jam per hari)?", 0.0, 3.0, 1.0)

calc = st.selectbox("Kebiasaan konsumsi alkohol?", ["no", "Sometimes", "Frequently", "Always"])
calc_dict = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
calc = calc_dict[calc]

mtrans = st.selectbox("Transportasi utama?", ["Walking", "Public_Transportation", "Automobile", "Bike", "Motorbike"])
mtrans_dict = {"Walking": 0, "Public_Transportation": 1, "Automobile": 2, "Bike": 3, "Motorbike": 4}
mtrans = mtrans_dict[mtrans]

# Data input ke array
input_data = np.array([[age, gender, height, weight, calc, favc, fcvc, ncp, scc, smoke, ch2o,
                        family_history, faf, tue, caec, mtrans]])

# Scaling input
input_scaled = scaler.transform(input_data)

# Tombol prediksi
if st.button("Prediksi Tingkat Obesitas"):
    prediction = model.predict(input_scaled)[0]

    kelas_obesitas = {
        0: "Insufficient Weight (Berat Badan Kurang)",
        1: "Normal Weight (Berat Badan Normal)",
        2: "Kelebihan Berat Badan Tingkat I (Overweight Level I)",
        3: "Kelebihan Berat Badan Tingkat II (Overweight Level II)",
        4: "Obesitas Tipe I (Obesity Type I)",
        5: "Obesitas Tipe II (Obesity Type II)",
        6: "Obesitas Tipe III (Obesity Type III)"
        }

        st.success(f"Hasil Prediksi: {kategori.get(prediction[0], 'Tidak diketahui')}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi:\n\n{str(e)}")
