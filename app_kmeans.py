import streamlit as st
import joblib
import pandas as pd

# Load model dan scaler
scaler = joblib.load('scaler.pkl')
kmeans = joblib.load('kmeans_model.pkl')

# Mapping cluster
cluster_mapping = {0: 'Kurang', 1: 'Sedang', 2: 'Baik'}

# Fungsi untuk memprediksi cluster
def predict_cluster(data):
    # Konversi data ke DataFrame
    df = pd.DataFrame([data])
    
    # Scaling
    scaled_data = scaler.transform(df)
    
    # Prediksi cluster
    cluster = kmeans.predict(scaled_data)[0]
    return cluster_mapping[cluster]

# Judul aplikasi
st.title('Prediksi Cluster Wine')

# Input Formulir untuk data
st.sidebar.header('Input Data')

# Ambil input dari pengguna
alcohol = st.sidebar.number_input('Alcohol', min_value=0.0, max_value=100.0, value=10.0)
malic_acid = st.sidebar.number_input('Malic Acid', min_value=0.0, max_value=100.0, value=10.0)
ash = st.sidebar.number_input('Ash', min_value=0.0, max_value=100.0, value=10.0)
ash_alcanity = st.sidebar.number_input('Ash Alcanity', min_value=0.0, max_value=100.0, value=10.0)
magnesium = st.sidebar.number_input('Magnesium', min_value=0.0, max_value=100.0, value=10.0)
total_phenols = st.sidebar.number_input('Total Phenols', min_value=0.0, max_value=100.0, value=10.0)
flavanoids = st.sidebar.number_input('Flavanoids', min_value=0.0, max_value=100.0, value=10.0)
nonflavanoid_phenols = st.sidebar.number_input('Nonflavanoid Phenols', min_value=0.0, max_value=100.0, value=10.0)
proanthocyanins = st.sidebar.number_input('Proanthocyanins', min_value=0.0, max_value=100.0, value=10.0)
color_intensity = st.sidebar.number_input('Color Intensity', min_value=0.0, max_value=100.0, value=10.0)
hue = st.sidebar.number_input('Hue', min_value=0.0, max_value=100.0, value=10.0)
od280 = st.sidebar.number_input('OD280', min_value=0.0, max_value=100.0, value=10.0)
proline = st.sidebar.number_input('Proline', min_value=0.0, max_value=100.0, value=10.0)

# Simpan input dalam dictionary
input_data = {
    'Alcohol': alcohol,
    'Malic_Acid': malic_acid,
    'Ash': ash,
    'Ash_Alcanity': ash_alcanity,
    'Magnesium': magnesium,
    'Total_Phenols': total_phenols,
    'Flavanoids': flavanoids,
    'Nonflavanoid_Phenols': nonflavanoid_phenols,
    'Proanthocyanins': proanthocyanins,
    'Color_Intensity': color_intensity,
    'Hue': hue,
    'OD280': od280,
    'Proline': proline
}

# Tombol untuk prediksi
if st.sidebar.button('Prediksi'):
    # Prediksi hasil cluster
    cluster_label = predict_cluster(input_data)
    
    # Tampilkan hasil prediksi
    st.write(f'Hasil prediksi cluster: {cluster_label}')
