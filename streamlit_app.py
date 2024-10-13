
import pandas as pd
import streamlit as st
from loantap_credit_default_risk_model import data_handling

model = data_handling.load_pipeline('XBG_model')

def prediction(data):
    df = pd.DataFrame([data])
    pred = model.predict(df)

    return "Approved" if pred[0] == 0 else "Rejected"

def main():
    st.title("Welcome to Loan Application")
    st.header("Please enter your details to proceed with your loan Application")

    input_fields = {
        'loan_amnt': st.number_input("Loan Amount", min_value=0.0),
        'term': st.selectbox("Term", (" 36 months", " 60 months")),
        'int_rate': st.number_input("Interest Rate", min_value=0.0, max_value=100.0),
        'installment': st.number_input("Installment", min_value=0.0),
        'grade': st.selectbox("Grade", ("A", "B", "C", "D", "E", "F", "G")),
        'sub_grade': st.selectbox("Sub Grade", [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)]),
        'emp_title': st.text_input("Employment Title"),
        'emp_length': st.selectbox("Employment Length", ("< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years")),
        'home_ownership': st.selectbox("Home Ownership", ("RENT", "OWN", "MORTGAGE", "OTHER")),
        'annual_inc': st.number_input("Annual Income", min_value=0.0),
        'verification_status': st.selectbox("Verification Status", ("Verified", "Not Verified", "Source Verified")),
        'issue_d': st.date_input("Issue Date"),
        'purpose': st.selectbox("Purpose", ("debt_consolidation", "credit_card", "home_improvement", "other")),
        'title': st.text_input("Loan Title"),
        'dti': st.number_input("Debt-to-Income Ratio", min_value=0.0),
        'earliest_cr_line': st.date_input("Earliest Credit Line"),
        'open_acc': st.number_input("Number of Open Credit Lines", min_value=0),
        'pub_rec': st.number_input("Number of Derogatory Public Records", min_value=0),
        'revol_bal': st.number_input("Revolving Balance", min_value=0.0),
        'revol_util': st.number_input("Revolving Utilization Rate", min_value=0.0, max_value=100.0),
        'total_acc': st.number_input("Total Number of Credit Lines", min_value=0),
        'initial_list_status': st.selectbox("Initial List Status", ("w", "f")),
        'application_type': st.selectbox("Application Type", ("INDIVIDUAL", "JOINT")),
        'mort_acc': st.number_input("Number of Mortgage Accounts", min_value=0),
        'pub_rec_bankruptcies': st.number_input("Number of Public Record Bankruptcies", min_value=0),
        'address': st.text_input("Address")
    }
    st.info("Not all fields are used for prediction but they are required for the model to run.")
    input_field = {
  "loan_amnt": 20000.0,
  "term": " 36 months",
  "int_rate": 12.99,
  "installment": 673.79,
  "grade": "C",
  "sub_grade": "C1",
  "emp_title": "New Market Analyst",
  "emp_length": "< 1 year",
  "home_ownership": "RENT",
  "annual_inc": 80000.0,
  "verification_status": "Verified",
  "issue_d": "Jul-2014",
  "purpose": "debt_consolidation",
  "title": "Debt consolidation",
  "dti": 19.04,
  "earliest_cr_line": "Sep-2002",
  "open_acc": 11.0,
  "pub_rec": 0.0,
  "revol_bal": 9802.0,
  "revol_util": 73.7,
  "total_acc": 37.0,
  "initial_list_status": "w",
  "application_type": "INDIVIDUAL",
  "mort_acc": 1.0,
  "pub_rec_bankruptcies": 0.0,
  "address": "970 Cox Causeway Suite 030 East Larry, AZ 11650"
}
    if st.button("Predict"):
        result = model.predict(pd.DataFrame([input_fields]))[0]
        if result == 0:
            st.success("Your loan Application is Approved")
        else:
            st.error("Your loan Application is Rejected")

    # show data in a table from input field
    st.info(input_fields)
    
    st.sidebar.header("LoanTap Loan status prediction")
    st.sidebar.text("Created by Athul Jyothis")

if __name__ == "__main__":
    main()
