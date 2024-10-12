"""
It includes functions for training an XGBoost model with feature engineering and saving the pipeline and test data.
It also includes functions for tuning the model's threshold adjustment.
"""
import logging
import os
import sys

from xgboost import XGBClassifier
import mlflow
import optuna
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

# Get the path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # root dir of project

# Add the parent directory to the system path (can import from anywhere)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from loantap_credit_default_risk_model import config, FE_pipeline, data_handling,evaluation

SCORING = 'f1'

# Load and split data
df = data_handling.load_data_and_sanitize(config.FILE_NAME)
X = df.drop(config.TARGET, axis=1)
y = df[config.TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=config.RANDOM_SEED, stratify= y)
X_test[config.TARGET] = y_test
data_handling.save_data(X_test, 'test_data.csv')
# Transform the target
y_train_transformed = FE_pipeline.target_pipeline.fit_transform(y_train)
# Transform test to match the target pipeline
y_test_transformed = FE_pipeline.target_pipeline.transform(y_test)
logging.info('Split data into train and test. Then, saved test data to test_data.csv')

# mlflow.sklearn.autolog()

def objective(trial):
    logging.info('Starting objective function for optuna trial')
    # Define the hyperparameters to tune
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'scale_pos_weight': trial.suggest_uniform('scale_pos_weight', 1.0, 4.0),  # Handle class imbalance
        'tree_method': 'hist',  # Fixed parameter
        'eval_metric': 'aucpr'  # Fixed parameter
    }
    # Define the pipeline with Feature Engineering and XGBoost
    XGB_with_FE = Pipeline([
        ('feature_engineering_pipeline', FE_pipeline.selected_FE_with_FS),
        ('base_model', XGBClassifier(**param, use_label_encoder=False))
    ])
    # Start an MLflow run
    with mlflow.start_run(nested=True):
        # Log hyperparameters in MLflow
        mlflow.log_params(param)
        # Train the model
        XGB_with_FE.fit(X_train, y_train_transformed)
        # Predict on the test set
        y_pred = XGB_with_FE.predict(X_test)
        # Calculate the F1 score for class 1 (minority class)
        f1_class_1 = f1_score(y_test_transformed, y_pred, pos_label=1)
        # Log the F1 score in MLflow
        mlflow.log_metric('f1_score', f1_class_1)
        logging.info('Finished objective function for optuna trial')
        return f1_class_1

def perform_training():
    # Model training
    logging.info('Starting training')
    # Create an Optuna study to maximize the F1 score for class 1
    study = optuna.create_study(direction='maximize')
    # Run the optimization with Optuna and log each trial in MLflow
    study.optimize(objective, n_trials=50)
    # Get the best hyperparameters
    best_params = study.best_trial.params
    # Log the best trial
    logging.info("Best trial: %s",best_params)
    # Train the final model with the best hyperparameters
    XGB_with_FE_best = Pipeline([
        ('feature_engineering_pipeline', FE_pipeline.selected_FE_with_FS),
        ('base_model', XGBClassifier(**best_params, use_label_encoder=False))
    ])
    with mlflow.start_run():
        # Log the best hyperparameters in MLflow
        mlflow.log_params(best_params)
        # Train the model
        XBG_model = XGB_with_FE_best.fit(X_train, y_train_transformed)
        # Post tuning of selected best model (threshold adjustment as per business requirements)
        XBG_model_tuned,report = evaluation.tune_model_threshold_adjustment(XBG_model,
                                                X_train,
                                                y_train,
                                                X_test,
                                                y_test,
                                                scoring=SCORING,
                                                target_pipeline=FE_pipeline.target_pipeline)
        mlflow.log_metrics(report['1']) # report is a nested dict for both class metrics => report['1'] gives all metrics for class 1 in dict format which will be logged
        mlflow.log_metric('threshold', XBG_model_tuned.best_threshold_)
        mlflow.sklearn.log_model(XBG_model_tuned, 'model')
        data_handling.save_pipeline(XBG_model_tuned, 'XBG_model')
        data_handling.save_pipeline(FE_pipeline.target_pipeline, 'target_pipeline_fitted')
        logging.info('Model trained and saved pipeline to trained_models folder')

if __name__ == '__main__':
    perform_training()

# todo add optuna for hyperparameter tuning and logs with mlfow each experiment