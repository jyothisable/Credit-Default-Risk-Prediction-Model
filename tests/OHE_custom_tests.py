import sys
import os
import pandas as pd
import numpy as np
import pytest
# Get the path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Add the parent directory to the system path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from loantap_credit_default_risk_model.custom_transformers import OneHotEncoderWithOthers
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'A': ['a', 'b', 'c', 'a', 'b', 'd', 'e', 'a', 'b', 'c'],
        'B': ['1', '2', '3', '1', '2', '3', '4', '1', '2', '3']
    })

def test_initialization():
    encoder = OneHotEncoderWithOthers(min_frequency=0.2)
    assert encoder.min_frequency == 0.2
    assert isinstance(encoder.encoder, OneHotEncoder)
    assert encoder.frequent_categories == {}

def test_fit(sample_data):
    encoder = OneHotEncoderWithOthers(min_frequency=0.2)
    encoder.fit(sample_data)
    
    assert set(encoder.frequent_categories['A']) == {'a', 'b', 'c'}
    assert set(encoder.frequent_categories['B']) == {'1', '2', '3'}

def test_fit_transform(sample_data):
    encoder = OneHotEncoderWithOthers(min_frequency=0.2)
    transformed = encoder.fit_transform(sample_data)

    assert transformed.shape == (10, 7)  # 3 for A (a, b, c), 3 for B (1, 2, 3), 1 for 'few_others'

def test_infrequent_categories():
    data = pd.DataFrame({
        'A': ['a', 'b', 'c', 'd', 'e'] * 20 + ['rare1', 'rare2']
    })
    
    encoder = OneHotEncoderWithOthers(min_frequency=0.1)
    encoder.fit(data)
    
    assert set(encoder.frequent_categories['A']) == {'a', 'b', 'c', 'd', 'e'}
    
    transformed = encoder.transform(data)
    assert transformed.shape[1] == 6  # 5 frequent categories + 1 'few_others'

def test_numeric_input():
    data = pd.DataFrame({
        'A': ['1', '2', '3', '4', '5'] * 20 + ['6', '7']
    })
    
    encoder = OneHotEncoderWithOthers(min_frequency=0.1)
    encoder.fit(data)
    
    assert set(encoder.frequent_categories['A']) == {'1', '2', '3', '4', '5'}
    
    transformed = encoder.transform(data)
    assert transformed.shape[1] == 6  # 5 frequent categories + 1 'few_others'

def test_transform_without_fit():
    encoder = OneHotEncoderWithOthers()
    with pytest.raises(NotFittedError):
        encoder.transform(pd.DataFrame({'A': ['1', '2', '3']}))

def test_non_dataframe_input():
    data = np.array([['a', 'b', 'c'], ['b', 'c', 'd'], ['c', 'd', 'e']])
    encoder = OneHotEncoderWithOthers()
    encoder.fit(data)
    transformed = encoder.transform(data)
    assert transformed.shape == (3, 5)  # 3 rows, 5 unique values

def test_new_category_in_transform():
    train_data = pd.DataFrame({'A': ['a', 'b', 'c', 'a', 'b']})
    test_data = pd.DataFrame({'A': ['a', 'b', 'd', 'e', 'c']})
    
    encoder = OneHotEncoderWithOthers(min_frequency=0.2)
    encoder.fit(train_data)
    
    transformed = encoder.transform(test_data)
    assert transformed.shape[1] == 4  # 'a', 'b', 'c', and 'few_others'