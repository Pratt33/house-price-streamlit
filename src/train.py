"""
Model training and evaluation for the house price prediction project.
"""
import logging
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple, Dict, Any, List
from sklearn.pipeline import Pipeline

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.preprocessing import prepare_data, create_preprocessing_pipeline
from src.utils import save_model, evaluate_model, load_data
from src.visualization import generate_report, plot_model_comparison

logger = logging.getLogger(__name__)

def train_model(X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]) -> Tuple[Any, Dict[str, float], Pipeline]:
    """
    Train and compare multiple house price prediction models.
    
    Args:
        X_train: Training features
        y_train: Training target
        feature_names: List of feature names
        
    Returns:
        Tuple of (best model, evaluation metrics, fitted pipeline)
    """
    # Split data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create preprocessing pipeline and fit on training split
    preprocessor = create_preprocessing_pipeline()
    X_train_processed = preprocessor.fit_transform(X_train_split)
    X_val_processed = preprocessor.transform(X_val)
    
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
        model.fit(X_train_processed, y_train_split)
        
        # Make predictions
        y_pred = model.predict(X_val_processed)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        
        metrics = {
            'model': name,
            'rmse': rmse,
            'r2': r2
        }
        metrics_list.append(metrics)
        
        # Generate visualization report for each model
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'model_reports', name)
        generate_report(
            model=model,
            X=X_val_processed,
            y=y_val,
            feature_names=feature_names,
            metrics=metrics,
            output_dir=output_dir
        )
        
        # Update best model if current model is better
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_metrics = metrics
            
        logger.info(f"{name} model - RMSE: {rmse:.2f}, R²: {r2:.2f}")
    
    # Create metrics DataFrame and plot comparisons
    metrics_df = pd.DataFrame(metrics_list)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'model_comparison')
    plot_model_comparison(metrics_df, output_dir)
    
    # Save metrics to CSV
    metrics_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'model_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    
    # Refit pipeline on full training data and retrain best model
    logger.info("Refitting pipeline on full training data and retraining best model...")
    preprocessor_full = create_preprocessing_pipeline()
    X_train_full_processed = preprocessor_full.fit_transform(X_train)
    best_model.fit(X_train_full_processed, y_train)
    
    # Save the fitted pipeline
    pipeline_path = os.path.join('output', 'preprocessing_pipeline.joblib')
    joblib.dump(preprocessor_full, pipeline_path)
    logger.info(f"Saved preprocessing pipeline to {pipeline_path}")
    
    return best_model, best_metrics, preprocessor_full

def main():
    """Main function to run the training pipeline."""
    try:
        # Get base directory
        base_dir = os.path.dirname(os.path.dirname(__file__))
        
        # Load data
        train_df, test_df = load_data(
            train_path=os.path.join(base_dir, 'data', 'train.csv'),
            test_path=os.path.join(base_dir, 'data', 'test.csv')
        )
        
        # Prepare data
        X_train, y_train, X_test = prepare_data(train_df, test_df)
        
        # Get feature names from preprocessing pipeline (from training split)
        # We'll use the initial pipeline for this
        temp_preprocessor = create_preprocessing_pipeline()
        temp_preprocessor.fit(X_train)
        col_transformer = temp_preprocessor.named_steps['preprocessing']
        feature_names = (
            col_transformer.transformers[0][2] +  # numeric features
            col_transformer.transformers[1][2]    # categorical features
        )
        
        # Train model
        model, metrics, preprocessor = train_model(X_train, y_train, feature_names)
        
        # Save model
        model_path = os.path.join(base_dir, 'output', 'model.joblib')
        save_model(model, model_path)
        
        # Transform test data using the final preprocessing pipeline
        X_test_processed = preprocessor.transform(X_test)
        
        # Make predictions on test set
        test_predictions = np.expm1(model.predict(X_test_processed))
        
        # Save predictions
        predictions_path = os.path.join(base_dir, 'output', 'predictions.csv')
        pd.DataFrame({
            'Id': test_df['Id'],
            'SalePrice': test_predictions
        }).to_csv(predictions_path, index=False)
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 