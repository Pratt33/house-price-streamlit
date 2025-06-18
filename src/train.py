"""
Model training and evaluation for the house price prediction project.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
from typing import Tuple, Dict, Any, List
import os

from .preprocessing import prepare_data, create_preprocessing_pipeline
from .utils import save_model, evaluate_model, load_data
from .visualization import generate_report, plot_model_comparison

logger = logging.getLogger(__name__)

def train_model(X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]) -> Tuple[Any, Dict[str, float]]:
    """
    Train and compare multiple house price prediction models.
    
    Args:
        X_train: Training features
        y_train: Training target
        feature_names: List of feature names
        
    Returns:
        Tuple of (best model, evaluation metrics)
    """
    # Split data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    }
    
    # Store metrics for each model
    metrics_list = []
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
        metrics['model'] = name
        metrics_list.append(metrics)
        
        # Generate visualization report for each model
        generate_report(
            model=model,
            X=X_val,
            y=y_val,
            feature_names=feature_names,
            metrics=metrics,
            output_dir=f'../output/model_reports/{name}'
        )
        
        # Update best model if current model is better
        if metrics['r2'] > best_score:
            best_score = metrics['r2']
            best_model = model
            best_metrics = metrics
            
        logger.info(f"{name} model R2 score: {metrics['r2']:.4f}")
    
    # Create metrics DataFrame and plot comparisons
    metrics_df = pd.DataFrame(metrics_list)
    plot_model_comparison(metrics_df, '../output/model_comparison')
    
    # Save metrics to CSV
    metrics_df.to_csv('../output/model_metrics.csv', index=False)
    
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
        
        # Get feature names from preprocessing pipeline
        pipeline = create_preprocessing_pipeline()
        feature_names = (
            pipeline.named_steps['preprocessor']
            .transformers_[0][2] +  # numeric features
            pipeline.named_steps['preprocessor']
            .transformers_[1][2]    # categorical features
        )
        
        # Train model
        model, metrics = train_model(X_train, y_train, feature_names)
        
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