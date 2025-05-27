from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model MCU
model = pickle.load(open('model/linear.pkl', 'rb'))  

app = Flask(__name__, template_folder='templates')

tensi_mapping = {'Normal': 0, 'Tinggi': 1, 'Rendah': 2}
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
    denyut_nadi = float(request.form['denyut_nadi'])
    suhu = float(request.form['suhu'])
    kesimpulan_tensi = request.form['kesimpulan_tensi']
    kesimpulan_jantung = request.form['kesimpulan_jantung']

  
    input_data = preprocess_input(sistolik, diastolik, denyut_nadi, suhu, kesimpulan_tensi, kesimpulan_jantung)
    prediction = model.predict(input_data)

    return render_template('index.html', prediction_text=f'Hasil prediksi kondisi kesehatan: <b>{prediction[0]}</b>')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Ambil input dari form
    features = [float(x) for x in request.form.values()]

    # Fitur: [Sistolik, Diastolik, Denyut, Suhu, Tensi, Jantung]
    final_features = np.array(features).reshape(1, -1)

    # Prediksi
    prediction = model.predict(final_features)

    # Format output
    hasil = 'Sehat' if prediction[0] == 0 else 'Tidak Sehat'

    return render_template('index.html', prediction_text=f'Hasil Prediksi Kondisi Kesehatan: <b>{hasil}</b>')

if __name__ == '__main__':
    app.run(debug=True)


