"""
This module contains all the functions for evaluating the performance of different pipeline or models.
It includes functions for evaluating the performance of the model using different metrics, such as accuracy, precision, recall, confusion matrix and AUC-ROC.
"""
    
import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TunedThresholdClassifierCV

from loantap_credit_default_risk_model.config import N_JOBS, RANDOM_SEED
from loantap_credit_default_risk_model.plotting import plot_threshold_scoring

def feature_engg_evaluator(x_train, y_train, x_test, y_test, feature_engineering_pipeline,target_pipeline):
    """
    Evaluate the given feature engineering pipeline using ExtraTreesClassifier
    and report the feature importances. The hyperparameters of the ExtraTreesClassifier
    are optimized using GridSearchCV over a small set of parameters.
    Additionally, we can also estimate the training time for the FE pipeline.

    Parameters
    ----------
    x_train : array-like, shape (n_samples, n_features)
        The training data.
    y_train : array-like, shape (n_samples,)
        The target values.
    x_test : array-like, shape (n_samples, n_features)
        The test data.
    y_test : array-like, shape (n_samples,)
        The target values for the test data.
    feature_engineering_pipeline : Pipeline
        The feature engineering pipeline to evaluate.
    target_pipeline : Pipeline
        The target pipeline.

    Returns
    -------
    best_model : Pipeline
        The best model found by GridSearchCV.
    """

    params = {  # some simple parameters to grid search
        'base_model__max_depth': [None],
        'base_model__n_estimators': [50],
        'base_model__criterion': ['gini'],
        }

    base_model = ExtraTreesClassifier(n_jobs=N_JOBS,random_state=RANDOM_SEED) # this evaluates our feature engineering pipeline
    
    model_with_fe = Pipeline([
        ('feature_engineering_pipeline', feature_engineering_pipeline),
        ('base_model', base_model)
    ])
    

    model_grid_search = GridSearchCV(model_with_fe, param_grid=params, cv=3,n_jobs=6,verbose=True,scoring='f1')
    start_time = time.time()  # capture the start time

    parse_time = time.time()
    print(f"Parsing took {(parse_time - start_time):.2f} seconds")

    model_grid_search.fit(x_train,target_pipeline.fit_transform(y_train))
    fit_time = time.time()
    print(f"Training took {(fit_time - start_time):.2f} seconds")

    best_model = model_grid_search.best_estimator_

    y_pred=best_model.predict(x_test)
    print(classification_report(target_pipeline.transform(y_test), y_pred))
    
    end_time = time.time()
    print(f"Overall took {(end_time - start_time):.2f} seconds")
    
    return best_model


# Post training tuning of selected best model
def tune_model_threshold_adjustment(tuned_model, X_train, y_train_transformed, X_test, y_test_transformed,scoring='f1'):
    """
    Perform a grid search on a given model to find the optimal threshold for prediction model (classification).
    
    Parameters
    ----------
    model: object
        The model to be optimized.
    X_train: array-like
        The training data.
    y_train_transformed: array-like
        The transformed target variable.
    X_test: array-like
        The testing data.
    y_test_transformed: array-like
        The transformed target variable for the test set.
    scoring: str, optional
        The scoring metric to use for optimization. Default is 'f1'.
    
    Returns
    -------
    The optimized model with the best threshold.
    """
    tuned_model = TunedThresholdClassifierCV(tuned_model, cv=3, scoring='f1',store_cv_results=True, n_jobs=N_JOBS)
    tuned_model.fit(X_train,y_train_transformed)
    y_pred=tuned_model.predict(X_test)
    print('Classification report: Training set')
    print(classification_report(y_train_transformed, tuned_model.predict(X_train)))
    
    print('Classification report: Testing set')
    print(classification_report(y_test_transformed, y_pred))
    print(f'Best threshold = {tuned_model.best_threshold_:.2f} with {scoring} score = {tuned_model.best_score_:.2f}')
    
    plot_threshold_scoring(tuned_model,scoring)
    return tuned_model

