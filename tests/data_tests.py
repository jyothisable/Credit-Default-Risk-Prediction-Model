import sys
import os
import pandas as pd

# Get the path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Add the parent directory to the system path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from loantap_credit_default_risk_model.data_processing import DataHandler

# Test Data import
data_import = DataHandler(file_path='data/raw/logistic_regression.csv')

df = data_import.load_data()
df = data_import.sanitize(df)

def test_data_import():
    assert df.shape == (396030, 27)


# Range Constrains    
def test_data_range_loan_amnt():
    """
    Test that the range of the loan_amnt is within the expected values
    """
    assert df['loan_amnt'].min() >= 1
    assert df['loan_amnt'].max() <= 140000
    
def test_data_range_int_rate():
    """
    Test that the range of the int_rate is within the expected values
    """
    assert df['int_rate'].min() >= 2
    assert df['int_rate'].max() <= 40
    
# Uniqueness Constrains
def test_data_uniqueness():
    """
    Test that the data is unique
    """
    assert len(df) == len(df.drop_duplicates())
