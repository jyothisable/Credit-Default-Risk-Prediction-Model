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
from sklearn.metrics import f1_score

# Get the path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # root dir of project

# Add the parent directory to the system path (can import from anywhere)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Prediction_Model import config, FE_pipeline, data_handling,evaluation

SCORING = 'f1'

# Load and split data
df_train = data_handling.load_data_and_sanitize('train_data.csv')
X_train, y_train = df_train.drop(config.TARGET, axis=1), df_train[config.TARGET]
df_test = data_handling.load_data_and_sanitize('test_data.csv')
X_test, y_test = df_test.drop(config.TARGET, axis=1), df_test[config.TARGET]
# Do feature engg transformation with fitted feature engineering pipeline (see get_features.py for optuna tuning of the pipeline)
feature_engg = data_handling.load_pipeline('fe_pipeline_fitted')
X_train_transformed = feature_engg.transform(X_train)
X_test_transformed = feature_engg.transform(X_test)
# Transform the target
y_train_transformed = FE_pipeline.target_pipeline.fit_transform(y_train)
# Transform test to match the target pipeline
y_test_transformed = FE_pipeline.target_pipeline.transform(y_test)
logging.info('Split data into train and test. Then, saved test data to test_data.csv')


def objective(trial):
    """
    Objective function for optuna tuning.

    This function takes a trial object from optuna and trains an XGBoost model on top of predefined feature engg pipeline.
    It logs the hyperparameters and the F1 score for the minority class in MLflow.
    """
    logging.info('Starting objective function for optuna trial')
    # Start an MLflow run
    with mlflow.start_run(nested=True,run_name=f"trial_{trial.number+1}"):
        # Define the hyperparameters to tune
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'gamma': trial.suggest_float('gamma', 0.001, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'lambda': trial.suggest_float('lambda', 0.001, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 0.001, 10.0, log=True),
            'tree_method': 'hist',  # Fixed parameter for faster training and better handling of large datasets
            'eval_metric': 'aucpr',  # Fixed parameter
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
            # 'early_stopping_rounds': 100,  # Fixed parameter, but can also be tuned
            # 'max_delta_step': trial.suggest_int('max_delta_step', 0, 10),
            # 'max_leaves': trial.suggest_int('max_leaves', 0, 64),
        }
        # Define the pipeline with Feature Engineering and XGBoost
        XGB_with_FE = XGBClassifier(**params)
        # Log hyperparameters in MLflow
        mlflow.log_params(params)
        # Train the model
        XGB_with_FE.fit(X_train_transformed, y_train_transformed) # TODO implement early stopping
        # Predict on the test set
        y_pred = XGB_with_FE.predict(X_test_transformed)
        # Calculate the F1 score for class 1 (minority class)
        f1_class_1 = f1_score(y_test_transformed, y_pred, pos_label=1)
        # Log the F1 score in MLflow
        mlflow.log_metric('f1_score', f1_class_1)
        logging.info('Finished objective function for optuna trial')
    return f1_class_1

def perform_training():
    """
    Train the model with the best hyperparameters found with Optuna and perform post-tuning threshold adjustment as per business requirements.
    Save the trained model and the target pipeline to the 'trained_models' folder.
    """
    logging.info('Starting training')
    mlflow.set_experiment("Model Optuna Optimization")
    with mlflow.start_run(run_name="Final Optuna Optimized Model"):
        # Set MLflow tags for final model
        mlflow.set_tag("model", "XGBClassifier")
        mlflow.set_tag("objective", "maximize_f1_class_1")
        # Create an Optuna study to maximize the F1 score for class 1
        study = optuna.create_study(direction='maximize')
        # Run the optimization with Optuna and log each trial in MLflow
        study.optimize(objective, n_trials=100, show_progress_bar=True)
        # Get the best hyperparameters
        best_params = study.best_trial.params
        # Log the best trial
        logging.info("Best trial: %s",best_params)
        # Train the final model with the best hyperparameters
        XGB_with_FE_best = XGBClassifier(**best_params)
        # Log the best hyperparameters in MLflow
        mlflow.log_params(best_params)
        # Train the model
        XBG_model = XGB_with_FE_best.fit(X_train_transformed, y_train_transformed)
        # Post tuning of selected best model (threshold adjustment as per business requirements)
        XBG_model_tuned,report = evaluation.tune_model_threshold_adjustment(XBG_model,
                                                X_train_transformed,
                                                y_train,
                                                X_test_transformed,
                                                y_test,
                                                scoring=SCORING,
                                                target_pipeline=FE_pipeline.target_pipeline)
        mlflow.log_metrics(report['1']) # report is a nested dict for both class metrics => report['1'] gives all metrics for class 1 in dict format which will be logged
        mlflow.log_metric('threshold', XBG_model_tuned.best_threshold_)
        mlflow.log_artifact("Prediction_Model/FE_pipeline.py")  # Log the pipeline as an artifact
        mlflow.log_artifact("Prediction_Model/config.py")  # Log the config file
        mlflow.sklearn.log_model(XBG_model_tuned, 'model')

    data_handling.save_pipeline(XBG_model_tuned, 'XBG_model')
    data_handling.save_pipeline(FE_pipeline.target_pipeline, 'target_pipeline_fitted')
    logging.info('Model trained and saved pipeline to trained_models folder')

if __name__ == '__main__':
    perform_training()
