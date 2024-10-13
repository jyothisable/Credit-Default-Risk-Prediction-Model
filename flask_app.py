"""
Flask app for loan default risk prediction
"""
from flask import Flask, render_template, request
import pandas as pd

# Then perform import
from loantap_credit_default_risk_model import data_handling

model = data_handling.load_pipeline('XBG_model')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests from the homepage form, makes a prediction and renders the homepage with the result.
    
    :return: Rendered homepage with the result of the prediction
    """
    if request.method == 'POST':
        request_data = dict(request.form)
        data = pd.DataFrame([request_data])
        print(data)
        pred = model.predict(data)
        print(f"prediction is {pred}")

        if int(pred[0]) == 0:
            result = "Congratulations! Your loan application is approved"
        else:
            result = "Sorry! Your loan application is rejected"
        return render_template('homepage.html', prediction = result)

@app.errorhandler(500)
def internal_error(error):
    return "500: Something went wrong"

@app.errorhandler(404)
def not_found(error):
    return "404: Page not found", 404

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)