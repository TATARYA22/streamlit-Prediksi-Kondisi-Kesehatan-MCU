import numpy as np
import pickle


# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Mapping kategori
tensi_mapping = {'Normal': 0, 'Pre Hipertensi': 1, 'Hipertensi': 2}
jantung_mapping = {'Normal': 0, 'Tidak Normal': 1}

def preprocess_input(sistolik, diastolik, denyut_nadi, suhu, kesimpulan_tensi, kesimpulan_jantung):
    tensi = tensi_mapping.get(kesimpulan_tensi, -1)
    jantung = jantung_mapping.get(kesimpulan_jantung, -1)
    return np.array([[sistolik, diastolik, denyut_nadi, suhu, tensi, jantung]])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sistolik = float(request.form['sistolik'])
    diastolik = float(request.form['diastolik'])
    denyut_nadi = float(request.form['denyut'])
    suhu = float(request.form['suhu'])
    tensi = request.form['tensi']
    jantung = request.form['jantung']

    input_data = preprocess_input(sistolik, diastolik, denyut_nadi, suhu, tensi, jantung)
    prediction = model.predict(input_data)

    hasil = 'Sehat' if prediction[0] == 0 else 'Tidak Sehat'

    return render_template('index.html', prediction_text=f'Hasil Prediksi Kondisi Kesehatan: <b>{hasil}</b>')

if __name__ == '__main__':
    app.run(debug=True)
