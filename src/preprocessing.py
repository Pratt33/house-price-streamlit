"""
Data preprocessing and feature engineering for the house price prediction project.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering."""
    
    def __init__(self):
        self.numeric_features = None
        self.categorical_features = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Apply feature engineering transformations."""
        X = X.copy()
        
        # Total square footage
        X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
        
        # Age of house when sold
        X['HouseAge'] = X['YrSold'] - X['YearBuilt']
        
        # Age of garage when sold
        X['GarageAge'] = X['YrSold'] - X['GarageYrBlt']
        
        # Total bathrooms
        X['TotalBath'] = X['FullBath'] + (0.5 * X['HalfBath'])
        
        # Total porch square footage
        X['TotalPorchSF'] = X['OpenPorchSF'] + X['EnclosedPorch'] + X['3SsnPorch'] + X['ScreenPorch']
        
        # Has features
        X['HasPool'] = X['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
        X['Has2ndFloor'] = X['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
        X['HasGarage'] = X['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
        X['HasBsmt'] = X['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
        X['HasFireplace'] = X['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
        
        return X

class DataCleaner(BaseEstimator, TransformerMixin):
    """Custom transformer for data cleaning."""
    
    def __init__(self):
        self.numeric_imputer = None
        self.categorical_imputer = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Apply data cleaning transformations."""
        X = X.copy()
        
        # Handle missing values in numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Handle missing values in categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        X[categorical_cols] = X[categorical_cols].fillna('Missing')
        
        return X

def create_preprocessing_pipeline():
    """
    Create a preprocessing pipeline for the house price prediction model.
    
    Returns:
        sklearn Pipeline object
    """
    # Define numeric and categorical features
    numeric_features = [
        'LotArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
        'GarageArea', 'TotalSF', 'HouseAge', 'GarageAge', 'TotalBath',
        'TotalPorchSF', 'HasPool', 'Has2ndFloor', 'HasGarage', 'HasBsmt',
        'HasFireplace'
    ]
    
    categorical_features = [
        'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
        'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',
        'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition'
    ]
    
    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create full pipeline
    pipeline = Pipeline(steps=[
        ('cleaner', DataCleaner()),
        ('engineer', FeatureEngineer()),
        ('preprocessor', preprocessor)
    ])
    
    return pipeline

def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Prepare data for model training and prediction.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        
    Returns:
        Tuple of (X_train, y_train, X_test)
    """
    # Separate features and target
    y_train = train_df['SalePrice']
    X_train = train_df.drop(['SalePrice', 'Id'], axis=1)
    X_test = test_df.drop(['Id'], axis=1)
    
    # Create and fit preprocessing pipeline
    pipeline = create_preprocessing_pipeline()
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    # Log transform target variable
    y_train_log = np.log1p(y_train)
    
    return X_train_processed, y_train_log, X_test_processed 