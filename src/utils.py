"""
Utility functions for the house price prediction project.
"""
import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.base import BaseEstimator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_model(model: BaseEstimator, path: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        path: Path where the model should be saved
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"Model saved successfully at {path}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(path: str) -> BaseEstimator:
    """
    Load a trained model from disk.
    
    Args:
        path: Path to the saved model
        
    Returns:
        Loaded model object
    """
    try:
        model = joblib.load(path)
        logger.info(f"Model loaded successfully from {path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def save_predictions(predictions: np.ndarray, path: str) -> None:
    """
    Save model predictions to a CSV file.
    
    Args:
        predictions: Array of predictions
        path: Path where predictions should be saved
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(predictions, columns=['SalePrice']).to_csv(path, index=False)
        logger.info(f"Predictions saved successfully at {path}")
    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")
        raise

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test datasets.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        
    Returns:
        Tuple of (training data, test data)
    """
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info(f"Data loaded successfully. Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate various regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary containing evaluation metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    logger.info("Model evaluation metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric.upper()}: {value:.4f}")
    
    return metrics 