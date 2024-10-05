"""
It includes functions for training an XGBoost model with feature engineering and saving the pipeline and test data.
It also includes functions for tuning the model's threshold adjustment.
"""
import logging
import os
import sys

import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Get the path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # root dir of project

# Add the parent directory to the system path (can import from anywhere)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from loantap_credit_default_risk_model import config, FE_pipeline, data_handling,evaluation

# Model and configuration parameters
XGB_with_FE = Pipeline([
    ('feature_engineering_pipeline', FE_pipeline.selected_FE_with_FS),
    ('base_model', XGBClassifier())
])

XGB_with_FE_CV = GridSearchCV(
    estimator=XGB_with_FE,
    param_grid={
        'base_model__max_depth': [5], 
        'base_model__learning_rate': [0.15],
        'base_model__n_estimators': [300], 
        'base_model__gamma': [0], 
        'base_model__subsample': [0.95], 
        'base_model__colsample_bytree': [0.95], 
        'base_model__lambda': [0.1],
        'base_model__tree_method': ["hist"],
        'base_model__eval_metric': ["aucpr"]
    },
    scoring='f1',
    cv=3,
    n_jobs=config.N_JOBS,
    verbose=True
)

SCORING = 'f1'

def perform_training():
    """
    Train the XGBoost model with feature engineering and save the pipeline and test data
    
    This function will load the data, sanitize it, split it into train and test, 
    perform feature engineering on the train data, and then train an XGBoost model 
    on the engineered train data. Finally, it will save the trained model in 
    the '/loantap_credit_default_risk_model/trained_models' directory, and the test 
    data in the 'data' directory.
    """
    logging.info('Starting training')
    df = data_handling.load_data_and_sanitize(config.FILE_NAME)
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'])
    df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
    X = df.drop(config.TARGET, axis=1)
    y = df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=config.RANDOM_SEED, stratify= y)
    X_test[config.TARGET] = y_test
    data_handling.save_data(X_test, 'test_data.csv')
    logging.info('Split data into train and test. Then, saved test data to test_data.csv')

    # Model training
    y_train_transformed = FE_pipeline.target_pipeline.transform(y_train)
    XBG_model = XGB_with_FE_CV.fit(X_train, y_train_transformed).best_estimator_
    
    # Post tuning of selected best model (threshold adjustment as per business requirements)
    XBG_model_tuned = evaluation.tune_model_threshold_adjustment(XBG_model, 
                                               X_train, 
                                               y_train, 
                                               X_test,
                                               y_test,
                                               scoring=SCORING,
                                               target_pipeline=FE_pipeline.target_pipeline)
    
    data_handling.save_pipeline(XBG_model_tuned, 'XBG_model')
    
    logging.info('Model trained and saved pipeline to trained_models folder')

if __name__ == '__main__':
    perform_training()
