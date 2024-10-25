from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import Form, StringField, SelectField, FloatField, IntegerField
from wtforms.validators import DataRequired
import pandas as pd
from Prediction_Model import data_handling

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

# Load pre-trained pipelines
fe_pipe = data_handling.load_pipeline('fe_pipeline_fitted_final')
model = data_handling.load_pipeline('XBG_model_final')

class LoanPredictionForm(FlaskForm):
    # ... (keep your existing form fields)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = LoanPredictionForm()
    prediction = None

    if form.validate_on_submit():
        # Get form data
        data = {
            'loan_amnt': form.loan_amnt.data,
            'term': form.term.data,
            'grade': form.grade.data,
            'sub_grade': form.sub_grade.data,
            'emp_length': form.emp_length.data,
            'home_ownership': form.home_ownership.data,
            'annual_inc': form.annual_inc.data,
            'verification_status': form.verification_status.data,
            'purpose': form.purpose.data,
            'title': form.title.data,
            'dti': form.dti.data,
            'last_pymnt_amnt': form.last_pymnt_amnt.data,
            'initial_list_status': form.initial_list_status.data,
            'addr_state': form.addr_state.data,
            
            # Set default values for non-important features
            'mths_since_last_delinq': None,
            'mths_since_last_major_derog': None,
            'mths_since_last_record': None,
            'issue_d': str(pd.Timestamp.now().date()),
            'emp_title': "Unknown",
            'earliest_cr_line': "06-1997",
            'open_acc': 0.0,
            'pub_rec': 0.0,
            'revol_bal': 0.0,
            'revol_util': 0.0,
            'total_acc': 0.0,
            'zip_code': "00000",
            'delinq_2yrs': 0.0,
            'inq_last_6mths': 0.0,
            'collections_12_mths_ex_med': 0.0,
            'open_acc_6m': 0.0,
            'open_il_6m': 0.0,
            'open_il_12m': 0.0,
            'open_il_24m': 0.0,
            'mths_since_rcnt_il': 0.0,
            'total_bal_il': 0.0,
            'il_util': 0.0,
            'open_rv_12m': 0.0,
            'open_rv_24m': 0.0,
            'max_bal_bc': 0.0,
            'all_util': 0.0,
            'total_rev_hi_lim': 0.0,
            'inq_fi': 0.0,
            'total_cu_tl': 0.0,
            'inq_last_12m': 0.0,
            'tot_coll_amt': 0.0,
            'tot_cur_bal': 0.0,
            'application_type': "Individual",
            'lat': 47.7511,
            'lng': 120.7401
        }
        
        # Convert data to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        pred = model.predict(fe_pipe.transform(df))
        
        if int(pred[0]) == 0:
            prediction = "Congratulations! Your loan application is approved 🎉"
        else:
            prediction = "Sorry! Your loan application is rejected 🚫"
    
    return render_template('index.html', form=form, prediction=prediction)