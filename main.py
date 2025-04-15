#!/usr/bin/env python3
"""
Housing Price Analysis using CSUSHPISA Data
------------------------------------------
This script analyzes the Case-Shiller U.S. National Home Price Index (CSUSHPISA).

The script:
1. Loads the raw CSUSHPISA data
2. Performs feature engineering 
3. Creates visualizations of housing price trends
4. Performs time series analysis and forecasting

Usage:
    python main.py [--data_path DATA_PATH] [--models MODEL1,MODEL2,...] [--output_dir OUTPUT_DIR]

Options:
    --data_path: Path to housing data CSV file
    --models: Comma-separated list of models to train (linear,ridge,lasso,rf,gbm,arimax,var,lstm)
    --output_dir: Directory to save results and reports
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Import modules
from src.data import make_dataset, preprocess
from src.features import build_features
from src.models import train_model, predict_model, evaluate_model
from src.visualization import visualize

# Import our feature engineering module
from src.features.build_features import (
    create_price_change_features,
    create_seasonal_features,
    create_market_indicators,
    create_rolling_statistics,
    create_polynomial_features,
    prepare_csushpisa_data
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('housing_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run housing price analysis pipeline')
    
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/raw/housing_data.csv',
        help='Path to housing data CSV file'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        default='linear,ridge,rf',
        help='Comma-separated list of models to train'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='reports',
        help='Directory to save results and reports'
    )
    
    return parser.parse_args()

def create_directories(base_dir):
    """Create necessary directories."""
    # Create directories if they don't exist
    dirs = [
        os.path.join(base_dir, 'figures'),
        os.path.join(base_dir, 'models'),
        'data/processed'
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return dirs

def run_data_pipeline(data_path):
    """Run data preprocessing pipeline."""
    logger.info("Starting data preprocessing pipeline")
    
    # Check if data exists, if not create synthetic data
    if not os.path.exists(data_path):
        logger.info(f"Data not found at {data_path}, creating synthetic data")
        data = make_dataset.create_synthetic_data(
            n_samples=5000,
            save_path=data_path
        )
    else:
        # Load data
        data = preprocess.load_data(data_path)
    
    # Clean data
    data = preprocess.clean_data(data)
    
    # Handle missing values
    data = preprocess.handle_missing_values(data, strategy='simple')
    
    # Remove outliers
    data = preprocess.remove_outliers(data, method='zscore', threshold=3)
    
    # Create time features if date column exists
    if 'date' in data.columns:
        data = preprocess.create_time_features(data, 'date')
    
    # Save processed data
    processed_path = os.path.join('data', 'processed', 'processed_housing_data.csv')
    preprocess.save_processed_data(data, processed_path)
    
    logger.info(f"Data preprocessing completed. Processed data saved to {processed_path}")
    
    return data

def run_feature_engineering(data):
    """Run feature engineering pipeline."""
    logger.info("Starting feature engineering pipeline")
    
    # Create price per square foot if both columns exist
    if 'price' in data.columns and 'sqft' in data.columns:
        data = build_features.create_price_per_sqft(data)
    
    # Create age features if year_built exists
    if 'year_built' in data.columns:
        data = build_features.create_age_features(data)
    
    # Create location features if location columns exist
    location_cols = ['latitude', 'longitude', 'zip_code', 'zipcode', 'postal_code']
    if any(col in data.columns for col in location_cols):
        data = build_features.create_location_features(data)
    
    # Create seasonal features if date column exists
    if 'date' in data.columns:
        data = build_features.create_seasonal_features(data)
        
        # Create market indicators
        if 'price' in data.columns:
            data = build_features.create_market_indicators(data)
    
    # Select columns for polynomial features (only numeric columns with high correlation to price)
    if 'price' in data.columns:
        numeric_cols = data.select_dtypes(include=np.number).columns
        corr = data[numeric_cols].corr()['price'].abs().sort_values(ascending=False)
        poly_columns = corr[corr > 0.3].index.tolist()[:5]  # Top 5 correlated features
        
        if len(poly_columns) >= 2:  # Need at least 2 columns for interactions
            logger.info(f"Creating polynomial features for columns: {poly_columns}")
            data = build_features.create_polynomial_features(data, poly_columns, degree=2)
    
    # Save engineered features
    engineered_path = os.path.join('data', 'processed', 'engineered_housing_data.csv')
    preprocess.save_processed_data(data, engineered_path)
    
    logger.info(f"Feature engineering completed. Data with engineered features saved to {engineered_path}")
    
    return data

def run_model_training(data, models_list, output_dir):
    """Run model training and evaluation pipeline."""
    logger.info(f"Starting model training pipeline with models: {models_list}")
    
    # Ensure price column exists
    if 'price' not in data.columns:
        logger.error("Target column 'price' not found in data")
        return None
    
    # Split data into features and target
    target_col = 'price'
    y = data[target_col]
    
    # Define feature columns (exclude target and non-numeric columns)
    X = data.select_dtypes(include=np.number).drop(columns=[target_col], errors='ignore')
    logger.info(f"Using {X.shape[1]} features for modeling")
    
    # Train-test split (time-based if date column exists)
    if 'date' in data.columns:
        train_data, test_data = preprocess.train_test_split_time(data, date_column='date')
        X_train = train_data.select_dtypes(include=np.number).drop(columns=[target_col], errors='ignore')
        y_train = train_data[target_col]
        X_test = test_data.select_dtypes(include=np.number).drop(columns=[target_col], errors='ignore')
        y_test = test_data[target_col]
    else:
        # Use regular train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Ensure X_train and X_test have the same columns
    X_test = X_test[X_train.columns]
    
    # Train models based on requested list
    trained_models = {}
    model_metrics = {}
    
    if 'linear' in models_list:
        logger.info("Training Linear Regression model")
        linear_model = train_model.train_linear_regression(
            X_train, y_train, 
            save_path=os.path.join(output_dir, 'models', 'linear_regression.pkl')
        )
        trained_models['Linear Regression'] = linear_model
    
    if 'ridge' in models_list:
        logger.info("Training Ridge Regression model")
        ridge_model = train_model.train_ridge_regression(
            X_train, y_train, alpha=1.0,
            save_path=os.path.join(output_dir, 'models', 'ridge_regression.pkl')
        )
        trained_models['Ridge Regression'] = ridge_model
    
    if 'lasso' in models_list:
        logger.info("Training Lasso Regression model")
        lasso_model = train_model.train_lasso_regression(
            X_train, y_train, alpha=0.01,
            save_path=os.path.join(output_dir, 'models', 'lasso_regression.pkl')
        )
        trained_models['Lasso Regression'] = lasso_model
    
    if 'rf' in models_list:
        logger.info("Training Random Forest model")
        rf_model = train_model.train_random_forest(
            X_train, y_train, n_estimators=100,
            save_path=os.path.join(output_dir, 'models', 'random_forest.pkl')
        )
        trained_models['Random Forest'] = rf_model
    
    if 'gbm' in models_list:
        logger.info("Training Gradient Boosting model")
        gbm_model = train_model.train_gbm(
            X_train, y_train, n_estimators=100, learning_rate=0.1,
            save_path=os.path.join(output_dir, 'models', 'gradient_boosting.pkl')
        )
        trained_models['Gradient Boosting'] = gbm_model
    
    # For time series models (ARIMAX, VAR, LSTM), need to check if we have time data
    if 'date' in data.columns and any(m in models_list for m in ['arimax', 'var', 'lstm']):
        # Prepare time series data
        ts_data = data.sort_values('date')
        
        if 'arimax' in models_list:
            logger.info("Training ARIMAX model")
            # Prepare ARIMAX data
            endog = ts_data[target_col]
            exog = ts_data[['month_sin', 'month_cos']] if 'month_sin' in ts_data.columns else None
            
            try:
                arimax_model = train_model.train_arimax(
                    endog, exog, order=(1,1,1),
                    save_path=os.path.join(output_dir, 'models', 'arimax_model.pkl')
                )
                trained_models['ARIMAX'] = arimax_model
            except Exception as e:
                logger.error(f"Error training ARIMAX model: {e}")
        
        if 'var' in models_list:
            logger.info("Training VAR model")
            # Prepare VAR data (need multiple time series)
            try:
                # Select a few key numeric variables
                var_cols = [target_col] + [col for col in ['sqft', 'bedrooms', 'bathrooms', 'year_built'] 
                                          if col in ts_data.columns][:3]  # Use up to 3 additional columns
                
                var_data = ts_data[var_cols].resample('M', on='date').mean().dropna()
                
                if len(var_data) > 10:  # Need sufficient time periods
                    var_model = train_model.train_var_model(
                        var_data, lag=3,
                        save_path=os.path.join(output_dir, 'models', 'var_model.pkl')
                    )
                    trained_models['VAR'] = var_model
                else:
                    logger.warning("Not enough time periods for VAR model")
            except Exception as e:
                logger.error(f"Error training VAR model: {e}")
                
        if 'lstm' in models_list:
            logger.info("Training LSTM model")
            try:
                # Create sequences for LSTM (simplified)
                # This is a basic implementation - would need more preprocessing for real data
                from sklearn.preprocessing import MinMaxScaler
                
                # Scale data
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(ts_data[[target_col]])
                
                # Create sequences
                seq_length = 10
                X_lstm, y_lstm = [], []
                
                for i in range(len(scaled_data) - seq_length):
                    X_lstm.append(scaled_data[i:i+seq_length])
                    y_lstm.append(scaled_data[i+seq_length])
                
                X_lstm = np.array(X_lstm)
                y_lstm = np.array(y_lstm)
                
                # Split data
                split_idx = int(len(X_lstm) * 0.8)
                X_train_lstm = X_lstm[:split_idx]
                y_train_lstm = y_lstm[:split_idx]
                X_test_lstm = X_lstm[split_idx:]
                y_test_lstm = y_lstm[split_idx:]
                
                # Train LSTM model
                lstm_model = train_model.train_lstm(
                    X_train_lstm, y_train_lstm, epochs=50,
                    save_path=os.path.join(output_dir, 'models', 'lstm_model.h5')
                )
                trained_models['LSTM'] = lstm_model
                
            except Exception as e:
                logger.error(f"Error training LSTM model: {e}")
    
    # Evaluate models
    for name, model in trained_models.items():
        logger.info(f"Evaluating {name} model")
        
        try:
            # Make predictions
            if name in ['ARIMAX', 'VAR', 'LSTM']:
                # Time series models require special handling
                if name == 'LSTM':
                    y_pred = predict_model.predict_with_nn(model, X_test_lstm)
                    # Inverse transform predictions
                    y_pred = scaler.inverse_transform(y_pred)
                    y_test_actual = scaler.inverse_transform(y_test_lstm)
                    
                    # Evaluate on last part of the data
                    metrics = evaluate_model.calculate_regression_metrics(
                        y_test_actual.flatten(), y_pred.flatten()
                    )
                else:
                    # For ARIMAX, VAR
                    # This is simplified - would need proper forecast for these models
                    metrics = {'Note': f"{name} evaluation requires time series forecast"}
            else:
                # Standard models
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = evaluate_model.calculate_regression_metrics(y_test, y_pred)
                
                # Plot predictions vs actual
                fig_path = os.path.join(output_dir, 'figures', f'{name.lower().replace(" ", "_")}_predictions.png')
                visualize.plot_model_predictions(y_test, y_pred, model_name=name, save_path=fig_path)
                
                # Plot residuals
                fig_path = os.path.join(output_dir, 'figures', f'{name.lower().replace(" ", "_")}_residuals.png')
                evaluate_model.plot_residuals(y_test, y_pred, save_path=fig_path)
                
                # Plot feature importance if available
                if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                    fig_path = os.path.join(output_dir, 'figures', f'{name.lower().replace(" ", "_")}_feature_importance.png')
                    evaluate_model.evaluate_feature_importance(model, X_train, feature_names=X_train.columns, 
                                                              save_path=fig_path)
            
            model_metrics[name] = metrics
            logger.info(f"{name} evaluation metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Error evaluating {name} model: {e}")
    
    # Compare models
    if len(model_metrics) > 1:
        try:
            # Filter out time series models that don't have comparable metrics
            comparable_metrics = {k: v for k, v in model_metrics.items() if 'RMSE' in v}
            
            if comparable_metrics:
                fig_path = os.path.join(output_dir, 'figures', 'model_comparison.png')
                visualize.plot_model_comparison(comparable_metrics, metric='RMSE', save_path=fig_path)
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
    
    return trained_models, model_metrics

def create_visualizations(data, output_dir):
    """Create data visualizations."""
    logger.info("Creating data visualizations")
    
    figures_dir = os.path.join(output_dir, 'figures')
    
    # Price distribution
    visualize.plot_price_distribution(
        data, price_col='price',
        save_path=os.path.join(figures_dir, 'price_distribution.png')
    )
    
    # If we have location data, create a map visualization
    if 'latitude' in data.columns and 'longitude' in data.columns:
        visualize.plot_map_visualization(
            data, color_col='price',
            save_path=os.path.join(figures_dir, 'price_map.html')
        )
    
    # If we have time data, create time series visualizations
    if 'date' in data.columns:
        visualize.plot_price_trends(
            data, date_col='date', price_col='price',
            save_path=os.path.join(figures_dir, 'price_trends.png')
        )
        
        # If we have region data, create regional trends
        region_cols = ['region', 'zip_region', 'location_cluster']
        region_col = next((col for col in region_cols if col in data.columns), None)
        
        if region_col:
            visualize.plot_price_trends(
                data, date_col='date', price_col='price', by=region_col,
                save_path=os.path.join(figures_dir, f'price_trends_by_{region_col}.png')
            )
    
    # Create correlation heatmap
    visualize.plot_correlation_heatmap(
        data, target_col='price',
        save_path=os.path.join(figures_dir, 'correlation_heatmap.png')
    )
    
    # Create feature correlation plot
    visualize.plot_feature_correlation(
        data, target_col='price',
        save_path=os.path.join(figures_dir, 'feature_correlation.png')
    )
    
    # Create interactive dashboard
    visualize.create_dashboard(
        data, output_path=os.path.join(output_dir, 'housing_dashboard.html')
    )
    
    logger.info(f"Visualizations saved to {figures_dir}")

def load_data(file_path='CSUSHPISA.csv'):
    """
    Load the CSUSHPISA dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSUSHPISA.csv file
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the CSUSHPISA data
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Convert date column to datetime
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        
        logger.info(f"Loaded CSUSHPISA data with {len(df)} records from {df['observation_date'].min()} to {df['observation_date'].max()}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def explore_data(df):
    """
    Perform exploratory data analysis on CSUSHPISA data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the CSUSHPISA data
    """
    # Basic statistics
    logger.info("Basic statistics of CSUSHPISA data:")
    logger.info(f"\nShape: {df.shape}")
    logger.info(f"\nData types: {df.dtypes}")
    logger.info(f"\nSummary statistics:\n{df.describe()}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    logger.info(f"\nMissing values:\n{missing_values}")
    
    # Create EDA visualizations directory if it doesn't exist
    os.makedirs('reports/figures', exist_ok=True)
    
    # Plot the time series
    plt.figure(figsize=(12, 6))
    plt.plot(df['observation_date'], df['CSUSHPISA'])
    plt.title('Case-Shiller U.S. National Home Price Index (1987-present)')
    plt.xlabel('Date')
    plt.ylabel('Index Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/figures/csushpisa_timeseries.png')
    
    # Plot YoY percentage changes
    df_changes = create_price_change_features(df)
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_changes['observation_date'], df_changes['price_yoy_pct_change'])
    plt.title('Year-over-Year Percentage Change in Home Prices')
    plt.xlabel('Date')
    plt.ylabel('YoY Change (%)')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/figures/yoy_price_changes.png')
    
    # Seasonal patterns (monthly averages)
    df_seasonal = create_seasonal_features(df)
    monthly_avg = df_seasonal.groupby('month')['CSUSHPISA'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.bar(monthly_avg['month'], monthly_avg['CSUSHPISA'])
    plt.title('Average Home Price Index by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Index Value')
    plt.xticks(range(1, 13))
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig('reports/figures/monthly_averages.png')
    
    # Distribution of values
    plt.figure(figsize=(10, 6))
    sns.histplot(df['CSUSHPISA'], kde=True)
    plt.title('Distribution of Home Price Index Values')
    plt.xlabel('Index Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/figures/value_distribution.png')
    
    logger.info("Completed exploratory data analysis and saved visualizations")

def time_series_analysis(df):
    """
    Perform time series analysis on the CSUSHPISA data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the CSUSHPISA data
    """
    logger.info("Performing time series analysis")
    
    # Make a copy of the data
    ts_data = df.copy()
    ts_data.set_index('observation_date', inplace=True)
    
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(ts_data['CSUSHPISA'], model='additive', period=12)
    
    # Plot decomposition
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonality')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residuals')
    plt.tight_layout()
    plt.savefig('reports/figures/seasonal_decomposition.png')
    
    # Check for stationarity
    adf_result = adfuller(ts_data['CSUSHPISA'])
    logger.info(f"ADF Statistic: {adf_result[0]}")
    logger.info(f"p-value: {adf_result[1]}")
    
    # Make data stationary if needed (first difference)
    ts_data['CSUSHPISA_diff'] = ts_data['CSUSHPISA'].diff()
    
    # Check stationarity of differenced data
    adf_result_diff = adfuller(ts_data['CSUSHPISA_diff'].dropna())
    logger.info(f"ADF Statistic (differenced): {adf_result_diff[0]}")
    logger.info(f"p-value (differenced): {adf_result_diff[1]}")
    
    # Plot differenced data
    plt.figure(figsize=(12, 6))
    ts_data['CSUSHPISA_diff'].plot()
    plt.title('First Difference of Home Price Index')
    plt.xlabel('Date')
    plt.ylabel('First Difference')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/figures/differenced_series.png')
    
    # ACF and PACF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    sm.graphics.tsa.plot_acf(ts_data['CSUSHPISA_diff'].dropna(), lags=36, ax=ax1)
    ax1.set_title('Autocorrelation Function (ACF)')
    sm.graphics.tsa.plot_pacf(ts_data['CSUSHPISA_diff'].dropna(), lags=36, ax=ax2)
    ax2.set_title('Partial Autocorrelation Function (PACF)')
    plt.tight_layout()
    plt.savefig('reports/figures/acf_pacf.png')
    
    logger.info("Completed time series analysis")

def build_forecast_model(df, forecast_periods=24):
    """
    Build an ARIMA forecast model for CSUSHPISA data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the CSUSHPISA data
    forecast_periods : int
        Number of periods to forecast into the future
        
    Returns:
    --------
    tuple
        (forecast_results, model)
    """
    logger.info(f"Building forecast model to predict {forecast_periods} months ahead")
    
    # Prepare data for forecasting
    ts_data = df.copy()
    ts_data.set_index('observation_date', inplace=True)
    
    # Split data into train and test sets (use last 24 months as test)
    train_data = ts_data[:-24]
    test_data = ts_data[-24:]
    
    # Fit ARIMA model (p,d,q) - example parameters, can be optimized
    # p=1: Autoregressive lag order
    # d=1: Differencing order
    # q=1: Moving average window size
    model = ARIMA(train_data['CSUSHPISA'], order=(1, 1, 1))
    model_fit = model.fit()
    
    logger.info(f"ARIMA Model Summary:\n{model_fit.summary()}")
    
    # Forecast
    forecast_result = model_fit.forecast(steps=len(test_data))
    
    # Calculate forecast accuracy
    mae = mean_absolute_error(test_data['CSUSHPISA'], forecast_result)
    rmse = np.sqrt(mean_squared_error(test_data['CSUSHPISA'], forecast_result))
    
    logger.info(f"Forecast accuracy - MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    # Plot forecast vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data['CSUSHPISA'], label='Training Data')
    plt.plot(test_data.index, test_data['CSUSHPISA'], label='Actual Values')
    plt.plot(test_data.index, forecast_result, label='Forecast')
    plt.title('ARIMA Forecast vs Actual Home Price Index')
    plt.xlabel('Date')
    plt.ylabel('Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/figures/arima_forecast.png')
    
    # Now forecast future periods beyond the data
    future_forecast = model_fit.forecast(steps=forecast_periods)
    
    # Create future date range
    last_date = ts_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_periods, freq='MS')
    
    # Create a DataFrame for the future forecast
    future_forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast': future_forecast
    })
    
    # Plot future forecast
    plt.figure(figsize=(12, 6))
    plt.plot(ts_data.index, ts_data['CSUSHPISA'], label='Historical Data')
    plt.plot(future_forecast_df['date'], future_forecast_df['forecast'], label='Future Forecast', linestyle='--')
    plt.title(f'Home Price Index Forecast for Next {forecast_periods} Months')
    plt.xlabel('Date')
    plt.ylabel('Index Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/figures/future_forecast.png')
    
    logger.info(f"Completed forecast for next {forecast_periods} months")
    
    return future_forecast_df, model_fit

def build_regression_model(df):
    """
    Build regression models to predict housing prices using engineered features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the CSUSHPISA data with engineered features
    """
    logger.info("Building regression models")
    
    # Prepare data with all features
    df_features = prepare_csushpisa_data(df)
    
    # Drop rows with NaN values (common in time series with lagged features)
    df_features.dropna(inplace=True)
    
    # Define features to use (excluding date column and target)
    features = [col for col in df_features.columns if col not in ['observation_date', 'CSUSHPISA']]
    
    # Prepare X (features) and y (target)
    X = df_features[features]
    y = df_features['CSUSHPISA']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate models
    lr_preds = lr_model.predict(X_test)
    rf_preds = rf_model.predict(X_test)
    
    # Print metrics
    logger.info("\nLinear Regression Metrics:")
    logger.info(f"R² Score: {r2_score(y_test, lr_preds):.4f}")
    logger.info(f"MAE: {mean_absolute_error(y_test, lr_preds):.4f}")
    logger.info(f"RMSE: {np.sqrt(mean_squared_error(y_test, lr_preds)):.4f}")
    
    logger.info("\nRandom Forest Metrics:")
    logger.info(f"R² Score: {r2_score(y_test, rf_preds):.4f}")
    logger.info(f"MAE: {mean_absolute_error(y_test, rf_preds):.4f}")
    logger.info(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_preds)):.4f}")
    
    # Plot feature importance for Random Forest
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Feature Importances (Random Forest)')
    plt.tight_layout()
    plt.savefig('reports/figures/feature_importance.png')
    
    # Plot actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, rf_preds, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Actual vs Predicted Home Price Index (Random Forest)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/figures/actual_vs_predicted.png')
    
    logger.info("Completed regression modeling")

def main():
    """Main function to run the housing price analysis."""
    logger.info("Starting housing price analysis")
    
    try:
        # Load data
        df = load_data()
        
        # Exploratory data analysis
        explore_data(df)
        
        # Time series analysis
        time_series_analysis(df)
        
        # Build forecast model
        future_forecast, model = build_forecast_model(df, forecast_periods=36)
        
        # Build regression model with engineered features
        build_regression_model(df)
        
        logger.info("Housing price analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in housing price analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 