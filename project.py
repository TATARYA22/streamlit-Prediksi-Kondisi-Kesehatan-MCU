# Import pustaka
import numpy as np
import pandas as pd
import pickle

# Load model MCU
model = pickle.load(open('model_mcu.pkl', 'rb'))  # Pastikan model sudah disimpan dengan nama ini

# Input data (dari user)
sistolik = float(input('Masukkan Tekanan Darah Sistolik: '))
diastolik = float(input('Masukkan Tekanan Darah Diastolik: '))
denyut = float(input('Masukkan Denyut Nadi: '))
suhu = float(input('Masukkan Suhu Tubuh: '))
tensi = int(input('Masukkan Kesimpulan Tensi (0: Normal, 1: Pre Hipertensi, 2: Hipertensi): '))
jantung = int(input('Masukkan Kesimpulan Jantung (0: Normal, 1: Gangguan Irama): '))

# Buat array untuk prediksi
data_input = np.array([[sistolik, diastolik, denyut, suhu, tensi, jantung]])

# Prediksi
prediction = model.predict(data_input)

# Output hasil prediksi
hasil = 'Sehat' if prediction[0] == 0 else 'Tidak Sehat'
print(f"Hasil Prediksi Kondisi Kesehatan: {hasil}")
