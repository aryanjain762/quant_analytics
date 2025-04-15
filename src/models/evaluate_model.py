"""
Module: evaluate_model.py
Description: Functions for evaluating model performance

Functions:
- calculate_regression_metrics: Calculate standard regression metrics
- calculate_timeseries_metrics: Calculate metrics specific to time series models
- cross_validate_model: Perform cross-validation for a model
- evaluate_feature_importance: Evaluate and visualize feature importance
- compare_models: Compare multiple models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import statsmodels.api as sm
from scipy.stats import pearsonr

def calculate_regression_metrics(y_true, y_pred, prefix=''):
    """
    Calculate standard regression metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    prefix : str, optional
        Prefix to add to metric names
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate mean absolute percentage error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Calculate correlation coefficient
    corr, _ = pearsonr(y_true, y_pred)
    
    # Create metrics dictionary
    metrics = {
        f'{prefix}MSE': mse,
        f'{prefix}RMSE': rmse,
        f'{prefix}MAE': mae,
        f'{prefix}MAPE': mape,
        f'{prefix}R2': r2,
        f'{prefix}Correlation': corr
    }
    
    return metrics

def calculate_timeseries_metrics(y_true, y_pred, prefix=''):
    """
    Calculate metrics specific to time series models.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    prefix : str, optional
        Prefix to add to metric names
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    # Get standard regression metrics first
    metrics = calculate_regression_metrics(y_true, y_pred, prefix)
    
    # Calculate direction accuracy
    # How often the model predicts the correct direction of change
    actual_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    
    direction_accuracy = np.mean((actual_diff > 0) == (pred_diff > 0)) * 100
    metrics[f'{prefix}Direction_Accuracy'] = direction_accuracy
    
    # Calculate Theil's U statistic (version 2)
    # Compares model to a naive model (predict no change)
    y_naive = np.roll(y_true, 1)[1:]  # Shift by 1 as the naive forecast
    theil_u = np.sqrt(np.mean((y_true[1:] - y_pred[1:])**2)) / np.sqrt(np.mean((y_true[1:] - y_naive)**2))
    metrics[f'{prefix}Theil_U'] = theil_u
    
    # Calculate autocorrelation of residuals
    # If significant, model may be missing important patterns
    residuals = y_true - y_pred
    acf = sm.tsa.stattools.acf(residuals, nlags=5)
    metrics[f'{prefix}Residual_Autocorr'] = acf[1]  # First lag autocorrelation
    
    return metrics

def cross_validate_model(model, X, y, cv=5, scoring='neg_mean_squared_error', is_timeseries=False):
    """
    Perform cross-validation for a model.
    
    Parameters:
    -----------
    model : object
        Model with fit and predict methods
    X : pandas.DataFrame
        Features
    y : pandas.Series
        Target variable
    cv : int or cross-validation generator, optional
        Number of folds or CV generator
    scoring : str, optional
        Scoring metric
    is_timeseries : bool, optional
        Whether to use time series cross validation
        
    Returns:
    --------
    dict
        Dictionary containing CV results
    """
    # Use time series split if specified
    if is_timeseries:
        cv = TimeSeriesSplit(n_splits=cv)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    # For negative metrics (like neg_mean_squared_error), convert back to positive
    if scoring.startswith('neg_'):
        cv_scores = -cv_scores
    
    # Calculate statistics
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    
    # Create results dictionary
    results = {
        'cv_scores': cv_scores,
        'mean': cv_mean,
        'std': cv_std,
        'min': np.min(cv_scores),
        'max': np.max(cv_scores)
    }
    
    return results

def evaluate_feature_importance(model, X_train, feature_names=None, top_n=10, plot=True, save_path=None):
    """
    Evaluate and visualize feature importance.
    
    Parameters:
    -----------
    model : object
        Trained model with feature_importances_ attribute
    X_train : pandas.DataFrame
        Training features
    feature_names : list, optional
        List of feature names
    top_n : int, optional
        Number of top features to show
    plot : bool, optional
        Whether to create a plot
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    pandas.DataFrame
        Feature importances
    """
    # Get feature names if not provided
    if feature_names is None:
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns
        else:
            feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
    
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        raise ValueError("Model doesn't have feature_importances_ or coef_ attribute")
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot if requested
    if plot:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(top_n))
        plt.title('Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    return importance_df

def compare_models(models, X_test, y_test, model_names=None, is_timeseries=False, plot=True, save_path=None):
    """
    Compare multiple models.
    
    Parameters:
    -----------
    models : list
        List of trained models
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
    model_names : list, optional
        List of model names
    is_timeseries : bool, optional
        Whether to calculate time series metrics
    plot : bool, optional
        Whether to create a plot
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    pandas.DataFrame
        Comparison of model metrics
    """
    # Generate default model names if not provided
    if model_names is None:
        model_names = [f'Model_{i+1}' for i in range(len(models))]
    
    # Dictionary to store results
    results = {}
    
    # Evaluate each model
    for i, model in enumerate(models):
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        if is_timeseries:
            metrics = calculate_timeseries_metrics(y_test, y_pred)
        else:
            metrics = calculate_regression_metrics(y_test, y_pred)
        
        # Store in results
        results[model_names[i]] = metrics
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results).T
    
    # Plot if requested
    if plot:
        # Select key metrics to plot
        plot_metrics = ['RMSE', 'MAE', 'R2']
        plot_df = results_df[plot_metrics].copy()
        
        # Create plot
        plt.figure(figsize=(14, 8))
        
        # For each metric, create a subplot
        for i, metric in enumerate(plot_metrics):
            plt.subplot(1, len(plot_metrics), i+1)
            sns.barplot(x=plot_df.index, y=metric, data=plot_df)
            plt.title(f'Comparison of {metric}')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    return results_df

def plot_residuals(y_true, y_pred, bins=30, save_path=None):
    """
    Plot residuals to diagnose model performance.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    bins : int, optional
        Number of bins for histogram
    save_path : str, optional
        Path to save the plot
    """
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Residuals vs. Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='-')
    axes[0, 0].set_title('Residuals vs. Predicted')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    
    # Plot 2: Histogram of Residuals
    axes[0, 1].hist(residuals, bins=bins, alpha=0.7)
    axes[0, 1].set_title('Histogram of Residuals')
    axes[0, 1].set_xlabel('Residual Value')
    axes[0, 1].set_ylabel('Frequency')
    
    # Plot 3: QQ Plot of Residuals
    sm.qqplot(residuals, line='45', ax=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot of Residuals')
    
    # Plot 4: Actual vs. Predicted
    axes[1, 1].scatter(y_true, y_pred, alpha=0.6)
    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r-')
    axes[1, 1].set_title('Actual vs. Predicted')
    axes[1, 1].set_xlabel('Actual Values')
    axes[1, 1].set_ylabel('Predicted Values')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show() 