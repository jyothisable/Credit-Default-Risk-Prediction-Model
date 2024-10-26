from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import uvicorn
from Prediction_Model import data_handling
import datetime

app = FastAPI()

# Define a Pydantic model for the new features, with non-important ones set as optional
class LoanFeatures(BaseModel):
    # Important Features
    loan_amnt: float
    term: str
    grade: str
    sub_grade: str
    emp_length: str
    home_ownership: str
    annual_inc: float
    verification_status: str
    purpose: str
    title: str
    dti: float
    last_pymnt_amnt: float
    initial_list_status: str
    addr_state: str
    
    # Non-Important Features (set as optional with default values)
    mths_since_last_delinq: Optional[float] = None
    mths_since_last_major_derog: Optional[float] = None
    mths_since_last_record: Optional[float] = None
    issue_d : Optional[str] = str(datetime.date.today())
    emp_title: Optional[str] = "Unknown"
    earliest_cr_line: Optional[str] = "06-1997"
    open_acc: Optional[float] = 0.0
    pub_rec: Optional[float] = 0.0
    revol_bal: Optional[float] = 0.0
    revol_util: Optional[float] = 0.0
    total_acc: Optional[float] = 0.0
    zip_code: Optional[str] = "00000"
    delinq_2yrs: Optional[float] = 0.0
    inq_last_6mths: Optional[float] = 0.0
    collections_12_mths_ex_med: Optional[float] = 0.0
    open_acc_6m: Optional[float] = 0.0
    open_il_6m: Optional[float] = 0.0
    open_il_12m: Optional[float] = 0.0
    open_il_24m: Optional[float] = 0.0
    mths_since_rcnt_il: Optional[float] = 0.0
    total_bal_il: Optional[float] = 0.0
    il_util: Optional[float] = 0.0
    open_rv_12m: Optional[float] = 0.0
    open_rv_24m: Optional[float] = 0.0
    max_bal_bc: Optional[float] = 0.0
    all_util: Optional[float] = 0.0
    total_rev_hi_lim: Optional[float] = 0.0
    inq_fi: Optional[float] = 0.0
    total_cu_tl: Optional[float] = 0.0
    inq_last_12m: Optional[float] = 0.0
    tot_coll_amt: Optional[float] = 0.0
    tot_cur_bal: Optional[float] = 0.0
    application_type: Optional[str] = "Individual"
    lat: Optional[float] = 47.7511
    lng: Optional[float] = 120.7401

# Load feature engineering pipeline and model
fe_pipe = data_handling.load_pipeline('fe_pipeline_fitted_final')
model = data_handling.load_pipeline('XBG_model_final')

@app.get('/')
def index():
    return {'message': 'Welcome to Loan Prediction App'}

@app.post('/predict')
async def predict(loan_data: LoanFeatures):
    """
    Predicts whether a loan should be approved or not based on the input data.
    """
    # Extract the data from the input
    data = loan_data.model_dump()

    df = pd.DataFrame([data])
    # Apply the feature engineering pipeline and make predictions
    transformed_data = fe_pipe.transform(df)
    pred = model.predict(transformed_data)
    
    # Return prediction result
    if pred[0] == 0:
        return {'Status of Loan Application': 'Approved'}
    else:
        return {'Status of Loan Application': 'Rejected'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
