"""
Visualization utilities for the house price prediction project.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.regressor import ResidualsPlot, PredictionError
from yellowbrick.features import FeatureImportances
from typing import List, Dict, Any
import os
import logging

logger = logging.getLogger(__name__)

def plot_feature_importance(model: Any, feature_names: List[str], output_dir: str) -> None:
    """
    Plot feature importances for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot using matplotlib
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    plt.close()
    
    # Plot using yellowbrick
    visualizer = FeatureImportances(model, labels=feature_names)
    visualizer.fit(model, feature_names)
    visualizer.show(outpath=os.path.join(output_dir, 'feature_importance_yb.png'))
    plt.close()

def plot_residuals(model: Any, X: np.ndarray, y: np.ndarray, output_dir: str) -> None:
    """
    Plot residuals analysis using yellowbrick.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target values
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Residuals plot
    visualizer = ResidualsPlot(model)
    visualizer.fit(X, y)
    visualizer.score(X, y)
    visualizer.show(outpath=os.path.join(output_dir, 'residuals_plot.png'))
    plt.close()
    
    # Prediction error plot
    visualizer = PredictionError(model)
    visualizer.fit(X, y)
    visualizer.score(X, y)
    visualizer.show(outpath=os.path.join(output_dir, 'prediction_error.png'))
    plt.close()

def plot_model_comparison(metrics_df: pd.DataFrame, output_dir: str) -> None:
    """
    Create comprehensive model comparison visualizations.
    
    Args:
        metrics_df: DataFrame containing model metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn')
    
    # RMSE Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=metrics_df, x='model', y='rmse')
    plt.title('RMSE Comparison Across Models')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmse_comparison.png'))
    plt.close()
    
    # R2 Score Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=metrics_df, x='model', y='r2')
    plt.title('R² Score Comparison Across Models')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_comparison.png'))
    plt.close()
    
    # Combined metrics plot
    plt.figure(figsize=(12, 6))
    metrics_df_melted = pd.melt(
        metrics_df,
        id_vars=['model'],
        value_vars=['rmse', 'r2'],
        var_name='metric',
        value_name='score'
    )
    sns.barplot(data=metrics_df_melted, x='model', y='score', hue='metric')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_metrics.png'))
    plt.close()

def plot_prediction_analysis(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, output_dir: str) -> None:
    """
    Create comprehensive prediction analysis plots.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate errors
    errors = y_true - y_pred
    
    # Error distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(errors, kde=True)
    plt.title(f'Error Distribution - {model_name}')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_error_dist.png'))
    plt.close()
    
    # Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(f'Actual vs Predicted - {model_name}')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_actual_vs_pred.png'))
    plt.close()
    
    # Residuals vs Predicted
    plt.figure(figsize=(12, 6))
    plt.scatter(y_pred, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Residuals vs Predicted - {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_residuals_vs_pred.png'))
    plt.close()

def generate_report(model: Any, X: np.ndarray, y: np.ndarray, 
                   feature_names: List[str], metrics: Dict[str, float],
                   output_dir: str) -> None:
    """
    Generate a comprehensive visualization report.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target values
        feature_names: List of feature names
        metrics: Dictionary of model metrics
        output_dir: Directory to save plots
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'feature_importance'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'residuals'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)
    
    # Generate all plots
    plot_feature_importance(model, feature_names, os.path.join(output_dir, 'feature_importance'))
    plot_residuals(model, X, y, os.path.join(output_dir, 'residuals'))
    
    # Get predictions
    y_pred = model.predict(X)
    plot_prediction_analysis(y, y_pred, model.__class__.__name__, 
                           os.path.join(output_dir, 'predictions'))
    
    # Save metrics
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'model_metrics.csv'), index=False)
    
    logger.info(f"Visualization report generated in {output_dir}") 