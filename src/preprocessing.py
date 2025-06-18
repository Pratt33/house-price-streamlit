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
from sklearn.impute import SimpleImputer
from typing import Tuple, List

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

def create_preprocessing_pipeline() -> Pipeline:
    """
    Create a preprocessing pipeline for the house price prediction project.
    
    Returns:
        Pipeline: Preprocessing pipeline
    """
    # Define numeric and categorical features
    numeric_features = [
        'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
        '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
        'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
        'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
        'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
        'MiscVal', 'TotalSF', 'HouseAge', 'GarageAge', 'TotalBath', 'TotalPorchSF',
        'HasPool', 'Has2ndFloor', 'HasGarage', 'HasBsmt', 'HasFireplace'
    ]
    
    categorical_features = [
        'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
        'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond',
        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
        'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',
        'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',
        'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',
        'SaleType', 'SaleCondition'
    ]
    
    # Create preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Create full pipeline with feature engineering
    full_pipeline = Pipeline([
        ('feature_engineering', FeatureEngineer()),
        ('preprocessing', preprocessor)
    ])
    
    return full_pipeline

def prepare_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Prepare data for model training.
    
    Args:
        train_df: Training data DataFrame
        test_df: Test data DataFrame
        
    Returns:
        Tuple of (X_train_df, y_train, X_test_df)
    """
    # Prepare target variable
    y_train = np.log1p(train_df['SalePrice'])
    
    # Prepare features
    X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
    X_test = test_df.drop(['Id'], axis=1)
    
    return X_train, y_train, X_test 