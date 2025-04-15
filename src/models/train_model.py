"""
Module: train_model.py
Description: Functions for training different housing price prediction models

Functions:
- train_linear_regression: Trains a multiple linear regression model
- train_ridge_regression: Trains a ridge regression model
- train_lasso_regression: Trains a lasso regression model
- train_arimax: Trains an ARIMAX time series model
- train_var_model: Trains a Vector Autoregression model
- train_random_forest: Trains a Random Forest regression model
- train_gbm: Trains a Gradient Boosting regression model
- train_lstm: Trains an LSTM neural network model
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt

def train_linear_regression(X_train, y_train, save_path=None):
    """
    Trains a multiple linear regression model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    save_path : str, optional
        Path to save the trained model
        
    Returns:
    --------
    sklearn.linear_model.LinearRegression
        Trained linear regression model
    """
    # Initialize and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print model coefficients
    coefficients = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_
    })
    print("Linear Regression Coefficients:")
    print(coefficients.sort_values('Coefficient', ascending=False))
    
    # Save model if path provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {save_path}")
    
    return model

def train_ridge_regression(X_train, y_train, alpha=1.0, save_path=None):
    """
    Trains a ridge regression model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    alpha : float
        Regularization strength
    save_path : str, optional
        Path to save the trained model
        
    Returns:
    --------
    sklearn.linear_model.Ridge
        Trained ridge regression model
    """
    # Initialize and train model
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    
    # Print model coefficients
    coefficients = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_
    })
    print(f"Ridge Regression Coefficients (alpha={alpha}):")
    print(coefficients.sort_values('Coefficient', ascending=False))
    
    # Save model if path provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {save_path}")
    
    return model

def train_lasso_regression(X_train, y_train, alpha=1.0, save_path=None):
    """
    Trains a lasso regression model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    alpha : float
        Regularization strength
    save_path : str, optional
        Path to save the trained model
        
    Returns:
    --------
    sklearn.linear_model.Lasso
        Trained lasso regression model
    """
    # Initialize and train model
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    
    # Print model coefficients
    coefficients = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_
    })
    # Filter out zero coefficients (features dropped by Lasso)
    non_zero_coefs = coefficients[coefficients['Coefficient'] != 0]
    print(f"Lasso Regression Coefficients (alpha={alpha}):")
    print(non_zero_coefs.sort_values('Coefficient', ascending=False))
    print(f"Number of features selected by Lasso: {len(non_zero_coefs)}")
    
    # Save model if path provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {save_path}")
    
    return model

def train_arimax(endog, exog=None, order=(1,1,1), seasonal_order=None, save_path=None):
    """
    Trains an ARIMAX time series model.
    
    Parameters:
    -----------
    endog : pandas.Series
        Endogenous variable (target)
    exog : pandas.DataFrame, optional
        Exogenous variables (features)
    order : tuple, optional
        (p,d,q) order of the model for AR, I, and MA components
    seasonal_order : tuple, optional
        (P,D,Q,s) order of the seasonal component
    save_path : str, optional
        Path to save the trained model
        
    Returns:
    --------
    statsmodels.tsa.arima.model.ARIMAResults
        Fitted ARIMAX model
    """
    if seasonal_order:
        model = ARIMA(endog, exog=exog, order=order, seasonal_order=seasonal_order)
    else:
        model = ARIMA(endog, exog=exog, order=order)
    
    # Fit model
    results = model.fit()
    
    # Print model summary
    print(results.summary())
    
    # Save model if path provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Model saved to {save_path}")
    
    return results

def train_var_model(data, lag=4, save_path=None):
    """
    Trains a Vector Autoregression model.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataframe containing all variables for VAR model
    lag : int, optional
        Number of lags to include
    save_path : str, optional
        Path to save the trained model
        
    Returns:
    --------
    statsmodels.tsa.api.VARResults
        Fitted VAR model
    """
    # Initialize model
    model = VAR(data)
    
    # Fit model
    results = model.fit(lag)
    
    # Print model summary
    print(results.summary())
    
    # Save model if path provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Model saved to {save_path}")
    
    return results

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, save_path=None):
    """
    Trains a Random Forest regression model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    n_estimators : int, optional
        Number of trees in the forest
    max_depth : int, optional
        Maximum depth of trees
    save_path : str, optional
        Path to save the trained model
        
    Returns:
    --------
    sklearn.ensemble.RandomForestRegressor
        Trained Random Forest model
    """
    # Initialize and train model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Get feature importances
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Random Forest Feature Importances:")
    print(importances.head(10))
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    importances.head(10).plot(kind='barh', x='Feature', y='Importance')
    plt.title('Random Forest Feature Importances')
    plt.tight_layout()
    plt.savefig('random_forest_importances.png')
    
    # Save model if path provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {save_path}")
    
    return model

def train_gbm(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3, save_path=None):
    """
    Trains a Gradient Boosting regression model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    n_estimators : int, optional
        Number of boosting stages
    learning_rate : float, optional
        Learning rate for the model
    max_depth : int, optional
        Maximum depth of trees
    save_path : str, optional
        Path to save the trained model
        
    Returns:
    --------
    sklearn.ensemble.GradientBoostingRegressor
        Trained Gradient Boosting model
    """
    # Initialize and train model
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Get feature importances
    importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Gradient Boosting Feature Importances:")
    print(importances.head(10))
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    importances.head(10).plot(kind='barh', x='Feature', y='Importance')
    plt.title('Gradient Boosting Feature Importances')
    plt.tight_layout()
    plt.savefig('gradient_boosting_importances.png')
    
    # Save model if path provided
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {save_path}")
    
    return model

def train_lstm(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, save_path=None):
    """
    Trains an LSTM neural network model.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features, should be shaped as (samples, timesteps, features)
    y_train : numpy.ndarray
        Training target
    epochs : int, optional
        Number of training epochs
    batch_size : int, optional
        Batch size for training
    validation_split : float, optional
        Fraction of training data to use for validation
    save_path : str, optional
        Path to save the trained model
        
    Returns:
    --------
    tensorflow.keras.models.Sequential
        Trained LSTM model
    """
    # Ensure X_train is in the right shape
    if len(X_train.shape) != 3:
        raise ValueError("X_train must be 3D: (samples, timesteps, features)")
    
    # Get dimensions
    samples, timesteps, features = X_train.shape
    
    # Define model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(timesteps, features)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='mse')
    
    # Print model summary
    model.summary()
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('lstm_training_history.png')
    
    # Save model if path provided
    if save_path:
        model.save(save_path)
        print(f"Model saved to {save_path}")
    
    return model 