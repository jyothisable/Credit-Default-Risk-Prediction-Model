"""
Flask app for loan default risk prediction
"""
from flask import Flask, render_template, request
import pandas as pd

# Import necessary functions from your data handling module
from Prediction_Model import data_handling

# Load pre-trained pipelines
fe_pipe = data_handling.load_pipeline('fe_pipeline_fitted_final')
model = data_handling.load_pipeline('XBG_model_final')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Renders the homepage and handles loan prediction requests.
    """
    prediction = None
    if request.method == 'POST':
        request_data = dict(request.form)
        data = pd.DataFrame([request_data])
        print(data)
        pred = model.predict(fe_pipe.transform(data))  # Perform feature engineering and make prediction
        print(f"prediction is {pred}")

        if int(pred[0]) == 0:
            prediction = "Congratulations! Your loan application is approved 🎉"
        else:
            prediction = "Sorry! Your loan application is rejected 🚫"
    
    return render_template('homepage.html', prediction=prediction)

@app.errorhandler(500)
def internal_error(error):
    return "500: Something went wrong"

@app.errorhandler(404)
def not_found(error):
    return "404: Page not found", 404

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)