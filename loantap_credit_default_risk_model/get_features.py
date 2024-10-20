# TODO feature engg with optuna and mlflows
"""
It includes functions for training an XGBoost model with feature engineering and saving the pipeline and test data.
It also includes functions for tuning the model's threshold adjustment.
"""
import logging
import os
import sys

# from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
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


def objective(trial):
    """
    Objective function for optuna tuning.

    This function takes a trial object from optuna and trains an XGBoost model with feature engineering.
    It logs the hyperparameters and the F1 score for the minority class in MLflow.

    :param trial: optuna trial object
    :return: F1 score for the minority class
    """
    logging.info('Starting objective function for optuna trial')
    # Start an MLflow run
    with mlflow.start_run(nested=True,run_name=f"trial_{trial.number+1}"):
        # Define the hyperparameters to tune
        params = {
            # feature selection params
            'fe_pipeline__feature_selection_pipeline__k':trial.suggest_int('k', 10, 70),
            'fe_pipeline__feature_engineering_pipeline__categorical_nominal_pipeline__FE_construction_OHE__min_frequency':trial.suggest_float('min_frequency', 0.001, 0.3),
            'fe_pipeline__feature_engineering_pipeline__numerical_combined_pipeline__all_numerical__FE_construction_similarity__FE_construction_distance_to_cluster__n_clusters':trial.suggest_int('n_bins', 7, 15),
            'fe_pipeline__feature_engineering_pipeline__numerical_combined_pipeline__all_numerical__FE_construction_similarity__FE_construction_distance_to_cluster__gamma':trial.suggest_int('gamma', 0.1, 1),
            # model params
            'base_model__n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'base_model__criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'base_model__max_depth': trial.suggest_int('max_depth', 2, 32),
            'base_model__min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'base_model__min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'base_model__max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'base_model__bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        }
        # Passthough logic for kbins
        use_kbin_num =trial.suggest_categorical('use_kbin_num', [True, False]) # this is not stored in params as it is a passthrough and not used for logging explicitly
        if use_kbin_num:
            fe_kbin_params = {
                'fe_pipeline__feature_engineering_pipeline__numerical_combined_pipeline__FE_construction_binning__strategy':trial.suggest_categorical('strategy', ['uniform', 'quantile', 'kmeans']),
                'fe_pipeline__feature_engineering_pipeline__numerical_combined_pipeline__FE_construction_binning__n_bins':trial.suggest_int('n_bins', 3, 10),
            }
            params.update(fe_kbin_params)
        # Define the pipeline with Feature Engineering
        model_with_fe = Pipeline([
            ('fe_pipeline', FE_pipeline.selected_FE_with_FS),
            ('base_model', ExtraTreesClassifier()) # this evaluates the FE pipeline
        ])
        # Set the hyperparameters (both model and FE pipeline)
        model_with_fe.set_params(**params)
        # Log hyperparameters in MLflow
        mlflow.log_params(params)
        # Train the model
        model_with_fe.fit(X_train, y_train_transformed) # TODO implement early stopping
        # Predict on the test set
        y_pred = model_with_fe.predict(X_test)
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
    mlflow.set_experiment("Feature Engineering Optuna Optimization")
    with mlflow.start_run(run_name="Final Optuna Optimized Model"):
        # Set MLflow tags for final model
        mlflow.set_tag("FE_engg", "EXTRA")
        mlflow.set_tag("objective", "maximize_f1_class_1")
        # Create an Optuna study to maximize the F1 score for class 1
        study = optuna.create_study(direction='maximize')
        # Run the optimization with Optuna and log each trial in MLflow
        study.optimize(objective, n_trials=200, show_progress_bar=True)
        # Get the best hyperparameters
        best_params = study.best_trial.params
        # Log the best trial
        logging.info("Best trial: %s",best_params)
        # Define the pipeline with Feature Engineering
        model_with_fe = Pipeline([
            ('fe_pipeline', FE_pipeline.selected_FE_with_FS),
            ('base_model', ExtraTreesClassifier()) # this evaluates the FE pipeline
        ])
        # Set the hyperparameters (both model and FE pipeline)
        model_with_fe.set_params(**best_params)
        # Log the best hyperparameters in MLflow
        mlflow.log_params(best_params)
        # Train the model
        eval_model = model_with_fe.fit(X_train, y_train_transformed)
        # Post tuning of selected best model (threshold adjustment as per business requirements)
        eval_model_tuned,report = evaluation.tune_model_threshold_adjustment(eval_model,
                                                X_train,
                                                y_train,
                                                X_test,
                                                y_test,
                                                scoring=SCORING,
                                                target_pipeline=FE_pipeline.target_pipeline)
        mlflow.log_metrics(report['1']) # report is a nested dict for both class metrics => report['1'] gives all metrics for class 1 in dict format which will be logged
        mlflow.log_metric('threshold', eval_model_tuned.best_threshold_)
        mlflow.log_artifact("loantap_credit_default_risk_model/FE_pipeline.py")  # Log the pipeline as an artifact
        mlflow.log_artifact("loantap_credit_default_risk_model/config.py")  # Log the config file
        mlflow.sklearn.log_model(eval_model_tuned, 'eval_model')
        mlflow.sklearn.log_model(eval_model.named_steps['fe_pipeline'], 'fe_pipeline_fitted')

    data_handling.save_pipeline(eval_model_tuned, 'fe_eval_model')
    data_handling.save_pipeline(eval_model.named_steps['fe_pipeline'], 'fe_pipeline_fitted') # Save best fe pipeline for actual model training in train.py
    data_handling.save_pipeline(FE_pipeline.target_pipeline, 'target_pipeline_fitted')
    logging.info('Model trained and saved pipeline to trained_models folder')

if __name__ == '__main__':
    perform_training()