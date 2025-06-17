from flask import Flask, request, render_template
import pandas as pd
import pickle

# تحميل الموديل
pipe = pickle.load(open('LinearRegressionModel.pkl', 'rb'))  # تأكد من اسم الملف

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    name = request.form['model']
    company = request.form['company']
    year = int(request.form['year'])
    kms_driven = int(request.form['kms'])
    fuel_type = request.form['fuel']

    input_df = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    prediction = pipe.predict(input_df)[0]

    return render_template('index.html', prediction_text=f"Estimated Car Price: ₹{int(prediction):,}")

if __name__ == "__main__":
    app.run(debug=True)
