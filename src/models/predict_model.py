"""
Module: predict_model.py
Description: Functions for making predictions with trained models

Functions:
- predict_with_model: General function to make predictions with any model
- predict_with_timeseries: Make predictions with time series models (ARIMAX, VAR)
- predict_with_nn: Make predictions with neural network models (LSTM)
- create_forecast: Generate n-step ahead forecasts
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

def predict_with_model(model, X_test, model_type='sklearn'):
    """
    Make predictions with a trained model.
    
    Parameters:
    -----------
    model : object
        Trained model
    X_test : pandas.DataFrame
        Test features
    model_type : str, optional
        Type of model ('sklearn', 'statsmodels', 'keras')
        
    Returns:
    --------
    numpy.ndarray
        Predictions
    """
    if model_type == 'sklearn':
        # For scikit-learn models (LinearRegression, Ridge, Lasso, RandomForest, GBM)
        predictions = model.predict(X_test)
    elif model_type == 'statsmodels':
        # For statsmodels models (ARIMAX)
        predictions = model.forecast(steps=len(X_test), exog=X_test)
    elif model_type == 'keras':
        # For Keras models (LSTM)
        predictions = model.predict(X_test)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return predictions

def predict_with_timeseries(model, steps=12, exog=None):
    """
    Make predictions with time series models.
    
    Parameters:
    -----------
    model : object
        Trained time series model (ARIMAX or VAR)
    steps : int, optional
        Number of steps ahead to forecast
    exog : pandas.DataFrame, optional
        Exogenous variables for future periods (required for ARIMAX)
        
    Returns:
    --------
    pandas.DataFrame
        Forecasted values
    """
    if hasattr(model, 'forecast'):  # ARIMAX
        # For ARIMA/ARIMAX models
        forecast = model.forecast(steps=steps, exog=exog)
        
        # Convert to DataFrame if it's not already
        if not isinstance(forecast, pd.DataFrame):
            forecast = pd.DataFrame(forecast, columns=['forecast'])
            
    elif hasattr(model, 'get_forecast'):  # SARIMAX  
        forecast = model.get_forecast(steps=steps, exog=exog)
        forecast = forecast.predicted_mean
        
    elif hasattr(model, 'forecast'):  # VAR
        # For VAR models
        forecast = model.forecast(model.y, steps=steps)
        forecast = pd.DataFrame(forecast, columns=model.names)
    
    return forecast

def predict_with_nn(model, X_test):
    """
    Make predictions with neural network models.
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained neural network model
    X_test : numpy.ndarray
        Test data
        
    Returns:
    --------
    numpy.ndarray
        Predictions
    """
    # Make predictions
    predictions = model.predict(X_test)
    
    return predictions

def create_forecast(model, last_observed_data, n_steps=12, exog=None, model_type='sklearn'):
    """
    Create a multi-step forecast.
    
    Parameters:
    -----------
    model : object
        Trained model
    last_observed_data : pandas.DataFrame or numpy.ndarray
        Last observed data points
    n_steps : int, optional
        Number of steps ahead to forecast
    exog : pandas.DataFrame, optional
        Exogenous variables for the forecast period
    model_type : str, optional
        Type of model ('sklearn', 'timeseries', 'nn')
        
    Returns:
    --------
    pandas.DataFrame
        Forecasted values
    """
    if model_type == 'timeseries':
        # For time series models (ARIMAX, VAR)
        forecast = predict_with_timeseries(model, steps=n_steps, exog=exog)
    
    elif model_type == 'nn':
        # For neural networks (requires recursive forecasting for multi-step)
        forecast = []
        current_input = last_observed_data.copy()
        
        for i in range(n_steps):
            # Make one-step forecast
            next_step = predict_with_nn(model, current_input)
            forecast.append(next_step[0, 0])
            
            # Update input for next prediction
            if len(current_input.shape) == 3:  # For LSTM with (samples, timesteps, features)
                # Shift data and add new prediction
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1, 0] = next_step[0, 0]
        
        forecast = pd.DataFrame(forecast, columns=['forecast'])
    
    elif model_type == 'sklearn':
        # For traditional ML models (requires feature generation for multi-step)
        forecast = []
        
        # This is a simplified approach that assumes we have exogenous variables
        # In a real application, you'd need to generate features for each future step
        if exog is not None:
            forecast = predict_with_model(model, exog, model_type='sklearn')
            forecast = pd.DataFrame(forecast, columns=['forecast'])
        else:
            raise ValueError("Exogenous variables required for multi-step forecast with sklearn models")
    
    return forecast

def plot_forecast(historical_data, forecast_data, title='Housing Price Forecast', save_path=None):
    """
    Plot historical data and forecast.
    
    Parameters:
    -----------
    historical_data : pandas.Series or pandas.DataFrame
        Historical data
    forecast_data : pandas.Series or pandas.DataFrame
        Forecasted data
    title : str, optional
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(historical_data.index, historical_data.values, 'b-', label='Historical Data')
    
    # If forecast_data has a proper index, use it
    if hasattr(forecast_data, 'index') and hasattr(forecast_data.index, 'date'):
        plt.plot(forecast_data.index, forecast_data.values, 'r--', label='Forecast')
    else:
        # Create a date index extending from the end of historical data
        last_date = historical_data.index[-1]
        if hasattr(last_date, 'date'):  # If it's a datetime
            freq = pd.infer_freq(historical_data.index)
            future_index = pd.date_range(start=historical_data.index[-1] + pd.Timedelta(days=1), 
                                       periods=len(forecast_data), 
                                       freq=freq)
            plt.plot(future_index, forecast_data.values, 'r--', label='Forecast')
        else:
            # If we don't have dates, just use sequential numbers
            plt.plot(range(len(historical_data), len(historical_data) + len(forecast_data)),
                   forecast_data.values, 'r--', label='Forecast')
    
    # Add shaded confidence intervals if available
    if hasattr(forecast_data, 'conf_int') and callable(getattr(forecast_data, 'conf_int')):
        conf_int = forecast_data.conf_int()
        plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='r', alpha=0.1)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Housing Price')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show() 