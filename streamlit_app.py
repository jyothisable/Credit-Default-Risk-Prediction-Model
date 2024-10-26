import pandas as pd
import streamlit as st
from Prediction_Model import data_handling
from datetime import datetime

# Load pre-trained pipelines
fe_pipe = data_handling.load_pipeline('fe_pipeline_fitted_final')
model = data_handling.load_pipeline('XBG_model_final')

# Function to format dates for input compatibility
def format_date(date):
    return date.strftime("%b-%Y")

def main():
    # Set page configuration
    st.set_page_config(page_title="Credit Default Risk Prediction", page_icon="🏦", layout="wide")
    
    # Add header and title
    left_co, cent_co,last_co = st.columns([1,3,1])
    with cent_co:
        st.title("🏦 Credit Default Risk Prediction ML Model")
        st.image('notebooks/Designer.jpeg')
    
    st.markdown("""
    ### 📋 Predict Your Loan Approval Status
    Please provide the required details to check whether your loan application will be **approved** or **rejected**.  
    """)

    # Sidebar Information
    # st.sidebar.image("notebooks/Designer.jpeg", use_column_width=True)
    st.sidebar.title("Credit Default Risk Prediction")
    st.sidebar.markdown("""
    This application uses an End-to-End deployable machine learning model to predict whether your loan application will be approved or not based on input details.
    """)
    # Project github link
    st.sidebar.markdown('Github Repository : [![GitHub](https://img.shields.io/github/stars/jyothisable/Credit-Default-Risk-Prediction-Model)](https://github.com/jyothisable/Credit-Default-Risk-Prediction-Model)')
    st.sidebar.markdown("---")
    st.sidebar.info("Created by Athul Jyothis  \n [![GitHub](https://img.shields.io/github/followers/jyothisable?style=social&label=Follow)](https://github.com/jyothisable)")

    # Input form for essential loan application details
    with st.form(key="loan_form"):
        st.subheader("📝 Please Fill in Your Loan Application Details")

        # Create two columns for input fields
        col1, col2 = st.columns(2)

        with col1:
            loan_amnt = st.number_input("💵 Loan Amount", min_value=0.0, value=10000.0)
            term = st.selectbox("📅 Term", (" 36 months", " 60 months"))
            grade = st.selectbox("🏅 Grade", ("A", "B", "C", "D", "E", "F", "G"))
            sub_grade = st.selectbox("📊 Sub Grade", [f"{g}{i}" for g in "ABCDEFG" for i in range(1, 6)])
            emp_length = st.selectbox("👔 Employment Length", ("< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"))
            home_ownership = st.selectbox("🏠 Home Ownership", ("RENT", "OWN", "MORTGAGE", "OTHER"))
            annual_inc = st.number_input("💸 Annual Income ($)", min_value=0.0, value=50000.0)
        
        with col2:
            verification_status = st.selectbox("🔍 Verification Status", ("Verified", "Not Verified", "Source Verified"))
            purpose = st.selectbox("🎯 Purpose", ("debt_consolidation", "credit_card", "home_improvement", "other"))
            title = st.text_input("📝 Loan Title", value="Debt Consolidation")
            dti = st.number_input("📉 Debt-to-Income Ratio", min_value=0.0, value=15.0)
            last_pymnt_amnt = st.number_input("💳 Last Paid EMI Amount", min_value=0.0, value=200.0)
            initial_list_status = st.selectbox("🔖 Initial List Status", ("Whole Loan","Fractional Loan"))
            addr_state = st.selectbox("📍 State", ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"])
        
        # Form submission button
        left_co, cent_co,last_co = st.columns([1,3,1])
        with cent_co:
            submit_button = st.form_submit_button(label="🔍 Predict Loan Status",use_container_width=True,type="primary")

        if submit_button:
            # Collect input data with most frequent values for optional fields
            input_fields = {
                'loan_amnt': loan_amnt,
                'term': term,
                'grade': grade,
                'sub_grade': sub_grade,
                'emp_length': emp_length,
                'home_ownership': home_ownership,
                'annual_inc': annual_inc,
                'verification_status': verification_status,
                'purpose': purpose,
                'title': title,
                'dti': dti,
                'last_pymnt_amnt': last_pymnt_amnt,
                'initial_list_status': "w" if initial_list_status == "Whole Loan" else "f",
                'addr_state': addr_state,

                # Non-important fields with default values
                'mths_since_last_delinq': None,
                'mths_since_last_major_derog': None,
                'mths_since_last_record': None,
                'issue_d': str(datetime.today().strftime("%b-%Y")),
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

            # Make prediction
            data = pd.DataFrame([input_fields])
            prediction = model.predict(fe_pipe.transform(data))[0]

            # Show result based on prediction
            if prediction == 0:
                st.success("🎉 Congratulations! Your loan application is approved!")
            else:
                st.error("🚫 Sorry! Your loan application is rejected.")

if __name__ == "__main__":
    main()