import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class OneHotEncoderWithOthers(BaseEstimator, TransformerMixin):
    def __init__(self, min_frequency=0.01):
        self.min_frequency = min_frequency
        self.encoder = OneHotEncoder()
        self.frequent_categories = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        for column in X.columns:
            value_counts = X[column].value_counts(normalize=True)
            frequent_cats = value_counts[value_counts >= self.min_frequency].index.tolist()
            self.frequent_categories[column] = frequent_cats
        
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Transform X to replace infrequent categories with 'Others'
        X_transformed = self._replace_infrequent(X)
        
        # Apply one-hot encoding
        encoded = self.encoder.fit_transform(X_transformed) # always fit_transform because we already learned the frequent categories and replaced infrequent ones
        
        return encoded

    def _replace_infrequent(self, X):
        X_copy = X.copy()
        for column in X.columns:
            mask = ~X_copy[column].isin(self.frequent_categories[column])
            X_copy.loc[mask, column] = 'few_others'
        return X_copy
    
    
# class OneHotEncoderWithOthers(BaseEstimator, TransformerMixin):
#     def __init__(self, min_frequency=0.01, handle_unknown='ignore'):
#         self.min_frequency = min_frequency
#         self.handle_unknown = handle_unknown
#         self.encoder = OneHotEncoder(handle_unknown=self.handle_unknown)
#         self.frequent_categories = {}

#     def fit(self, X, y=None):
#         if not isinstance(X, pd.DataFrame):
#             X = pd.DataFrame(X)
        
#         for column in X.columns:
#             value_counts = X[column].value_counts(normalize=True)
#             frequent_cats = value_counts[value_counts >= self.min_frequency].index.tolist()
#             self.frequent_categories[column] = frequent_cats
        
#         # Transform X to replace infrequent categories with 'Others'
#         X_transformed = self._replace_infrequent(X)
        
#         # Fit the OneHotEncoder on the transformed data
#         self.encoder.fit(X_transformed)
        
#         return self

#     def transform(self, X):
#         if not isinstance(X, pd.DataFrame):
#             X = pd.DataFrame(X)
        
#         # Transform X to replace infrequent categories with 'Others'
#         X_transformed = self._replace_infrequent(X)
        
#         # Apply one-hot encoding
#         encoded = self.encoder.transform(X_transformed)
        
#         # Get feature names
#         feature_names = self.encoder.get_feature_names_out(X.columns)
        
#         # Convert to DataFrame
#         encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
        
#         return encoded_df

#     def _replace_infrequent(self, X):
#         X_copy = X.copy()
#         for column in X.columns:
#             mask = ~X_copy[column].isin(self.frequent_categories[column])
#             X_copy.loc[mask, column] = 'Others'
#         return X_copy