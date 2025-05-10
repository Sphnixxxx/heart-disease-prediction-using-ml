from flask import Flask, request, render_template, redirect, url_for
from ml_model import predict_heart_disease  # Import ML function

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')  



@app.route('/predictt', methods=['POST'])
def predict():
    try:
        # Extract user input from form
        patient_id = int(request.form['patientid'])  # Not used in ML

        features = [
            int(request.form['patientid']),
            int(request.form['age']),           # Age
            int(request.form['sex']),           # Gender (1 = Male, 0 = Female)
            int(request.form['cp']),            # Chest Pain Type
            int(request.form['trestbps']),      # Resting BP
            int(request.form['chol']),          # Serum Cholesterol
            int(request.form['fbs']),           # Fasting Blood Sugar
            int(request.form['restecg']),       # Resting ECG
            int(request.form['thalach']),       # Maximum Heart Rate
            int(request.form['exang']),         # Exercise Induced Angina
            float(request.form['oldpeak']),     # ST Depression
            int(request.form['slope']),         # Slope of Peak ST
            int(request.form['ca'])             # Number of Major Vessels
        ]

        # Send input data to ML model
        result = predict_heart_disease(features)

        # Redirect user to result page
        return redirect(url_for('show_result', result=result))

    except Exception as e:
        return render_template('predictt.html', result=f"Error: {str(e)}")
# Main Page
@app.route('/main')
def main_page():
    return render_template('heart.html') 

@app.route('/result')
def show_result():
    result = request.args.get('result', 'No result available')
    return render_template('predictt.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
