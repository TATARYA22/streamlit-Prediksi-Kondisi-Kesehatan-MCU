import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Mapping
tensi_mapping = {'Normal': 0, 'Pre Hipertensi': 1, 'Hipertensi': 2}
jantung_mapping = {'Normal': 0, 'Tidak Normal': 1}

# Judul
st.title("Prediksi Kondisi Kesehatan MCU")

# Input form
with st.form("MCU Form"):
    sistolik = st.number_input("Tekanan Darah Sistolik", min_value=0.0)
    diastolik = st.number_input("Tekanan Darah Diastolik", min_value=0.0)
    denyut_nadi = st.number_input("Denyut Nadi", min_value=0.0)
    suhu = st.number_input("Suhu Tubuh", min_value=0.0, step=0.1)
    kesimpulan_tensi = st.selectbox("Kesimpulan Tensi", list(tensi_mapping.keys()))
    kesimpulan_jantung = st.selectbox("Kesimpulan Jantung", list(jantung_mapping.keys()))

    submit = st.form_submit_button("Prediksi")

# Prediksi
if submit:
    tensi = tensi_mapping.get(kesimpulan_tensi, -1)
    jantung = jantung_mapping.get(kesimpulan_jantung, -1)
    input_data = np.array([[sistolik, diastolik, denyut_nadi, suhu, tensi, jantung]])

    prediction = model.predict(input_data)
    hasil = "Sehat" if prediction[0] == 0 else "Tidak Sehat"

    st.success(f"Hasil Prediksi: {hasil}")

