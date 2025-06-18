"""
Model training and evaluation for the house price prediction project.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
from typing import Tuple, Dict, Any

from .preprocessing import prepare_data
from .utils import save_model, evaluate_model, load_data

logger = logging.getLogger(__name__)

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Any, Dict[str, float]]:
    """
    Train the house price prediction model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Tuple of (trained model, evaluation metrics)
    """
    # Split data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Initialize models
    models = {
        'rf': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        'gb': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    }
    
    best_model = None
    best_score = float('-inf')
    best_metrics = None
    
    # Train and evaluate each model
    for name, model in models.items():
        logger.info(f"Training {name} model...")
        
        # Train model
        model.fit(X_train_split, y_train_split)
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        metrics = evaluate_model(y_val, y_pred)
        
        # Update best model if current model is better
        if metrics['r2'] > best_score:
            best_score = metrics['r2']
            best_model = model
            best_metrics = metrics
            
        logger.info(f"{name} model R2 score: {metrics['r2']:.4f}")
    
    # Train best model on full dataset
    logger.info("Training best model on full dataset...")
    best_model.fit(X_train, y_train)
    
    return best_model, best_metrics

def main():
    """Main function to run the training pipeline."""
    try:
        # Load data
        train_df, test_df = load_data(
            train_path='../data/train.csv',
            test_path='../data/test.csv'
        )
        
        # Prepare data
        X_train, y_train, X_test = prepare_data(train_df, test_df)
        
        # Train model
        model, metrics = train_model(X_train, y_train)
        
        # Save model
        save_model(model, '../output/model.joblib')
        
        # Make predictions on test set
        test_predictions = np.expm1(model.predict(X_test))
        
        # Save predictions
        pd.DataFrame({
            'Id': test_df['Id'],
            'SalePrice': test_predictions
        }).to_csv('../output/predictions.csv', index=False)
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 