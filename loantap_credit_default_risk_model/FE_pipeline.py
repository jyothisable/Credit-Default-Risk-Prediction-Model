"""
This module contains all the Pipelines for feature engineering.
"""

from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, OneHotEncoder, OrdinalEncoder,KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.feature_selection import SelectKBest,chi2
    
import loantap_credit_default_risk_model.config as config

numerical_skewed_pipeline = Pipeline([
    ('select_numerical_skewed_features', FunctionTransformer(lambda X: X[config.NUM_SKEWED_FEATURES])),
    ('FE_improvement_impute', SimpleImputer(strategy='median'))
])

numerical_features_pipeline = Pipeline([
    ('select_numerical_features', FunctionTransformer(lambda X: X[config.NUM_FEATURES])),
    ('FE_improvement_impute', SimpleImputer(strategy='mean'))
])

numerical_features_combined_pipeline = Pipeline([
    ('all_numerical',FeatureUnion([
        ('numerical_skewed_pipeline', numerical_skewed_pipeline),
        ('numerical_features_pipeline', numerical_features_pipeline)
        ])),
    ('FE_construction_binning', KBinsDiscretizer(n_bins=5,encode='ordinal',strategy='kmeans')),
    ('FE_improvement_scaling',MinMaxScaler())
])


categorical_ordinal_pipeline = Pipeline([
    ('select_categorical_ordinal_features', FunctionTransformer(lambda X: X[config.CAT_ORDINAL_FEATURES])),
    ('FE_improvement_impute', SimpleImputer(strategy='most_frequent')),
    ('FE_construction_ODE', OrdinalEncoder(categories=config.ORDER_MATRIX))
])


all_nominal_cat = FeatureUnion([
            ('select_categorical_nominal_features', FunctionTransformer(lambda X: X[config.CAT_NOMINAL_FEATURES].applymap(lambda x: str(x).strip().lower()))),
            ('FE_construction_zipcode', FunctionTransformer(lambda X: X['address'].str.strip().str.slice(-5).to_frame('zipcode'))),
            ('FE_construction_state', FunctionTransformer(lambda X: X['address'].str.strip().str.slice(-8,-6).to_frame('state'))),
            ('FE_construction_age_of_credit', FunctionTransformer(lambda X: (X['issue_d'].dt.year - X['earliest_cr_line'].dt.year).to_frame('age_of_credit')))
        ])

categorical_nominal_pipeline = Pipeline([
    ('all_nominal_cat',all_nominal_cat),
    ('FE_improvement_impute', SimpleImputer(strategy='most_frequent')),
    ('FE_construction_OHE', OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=0.01,sparse_output=False))
])

selected_FE = FeatureUnion([
        ('numerical_combined_pipeline',numerical_features_combined_pipeline),
        ('categorical_ordinal_pipeline', categorical_ordinal_pipeline),
        ('categorical_nominal_pipeline', categorical_nominal_pipeline)
    ])

# Target pipeline for target variable encoding
target_pipeline = Pipeline([
    ('target_ohe',FunctionTransformer(lambda x : x.map({'Fully Paid':0,'Charged Off':1})))
])

# Final pipeline
selected_FE_with_FS = Pipeline([
    ('feature_engineering_pipeline', selected_FE),
    ('feature_selection_pipeline',SelectKBest(k=30,score_func=chi2))
])