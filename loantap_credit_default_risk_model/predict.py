import logging
import os
import sys

from sklearn.metrics import classification_report

# Get the path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # root dir of project

# Add the parent directory to the system path (can import from anywhere)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from loantap_credit_default_risk_model import config, data_handling, FE_pipeline

model = data_handling.load_pipeline('XBG_model')

def generate_prediction():
    logging.info('Starting prediction')
    test_data = data_handling.load_data_and_sanitize('test_data.csv')
    print(test_data)
    y = test_data[config.TARGET]
    X = test_data.drop(config.TARGET,errors='ignore')
    y_pred_transformed = model.predict(X)
    logging.info('Finished prediction')
    y_pred = FE_pipeline.target_pipeline.inverse_transform(y_pred_transformed)
    print(classification_report(y, y_pred))
    return y_pred

if __name__ == '__main__':
    generate_prediction()