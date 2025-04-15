"""
Module: build_features.py
Description: Functions for building features for housing price models using CSUSHPISA data

Functions:
- create_price_change_features: Calculate price changes and growth rates
- create_seasonal_features: Create seasonal features from dates
- create_market_indicators: Create housing market indicator features
- create_rolling_statistics: Create rolling window statistics
- create_polynomial_features: Create polynomial features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_price_change_features(data, price_col='CSUSHPISA', date_col='observation_date'):
    """
    Calculate price changes and growth rates for housing index.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing index data
    price_col : str, optional
        Name of the price column
    date_col : str, optional
        Name of the date column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added price change features
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Check if required columns exist
    if price_col not in df.columns or date_col not in df.columns:
        logger.error(f"Required columns {price_col} and/or {date_col} not found in data")
        return df
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            logger.error(f"Could not convert {date_col} to datetime")
            return df
    
    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Calculate month-over-month changes
    df['price_mom_change'] = df[price_col].diff()
    df['price_mom_pct_change'] = df[price_col].pct_change() * 100
    
    # Calculate quarter-over-quarter changes (3 months)
    df['price_qoq_change'] = df[price_col].diff(3)
    df['price_qoq_pct_change'] = df[price_col].pct_change(3) * 100
    
    # Calculate year-over-year changes (12 months)
    df['price_yoy_change'] = df[price_col].diff(12)
    df['price_yoy_pct_change'] = df[price_col].pct_change(12) * 100
    
    # Calculate cumulative growth from start
    df['price_cumulative_growth'] = (df[price_col] / df[price_col].iloc[0] - 1) * 100
    
    logger.info(f"Created price change features: mom, qoq, yoy and cumulative growth")
    
    return df

def create_seasonal_features(data, date_col='observation_date'):
    """
    Create seasonal features from dates.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing data
    date_col : str, optional
        Name of the date column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added seasonal features
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Check if required column exists
    if date_col not in df.columns:
        logger.error(f"Required column {date_col} not found in data")
        return df
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            logger.error(f"Could not convert {date_col} to datetime")
            return df
    
    # Extract base date components
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    
    # Create season feature
    # Northern Hemisphere seasons
    season_map = {
        1: 'Winter', 2: 'Winter', 3: 'Spring',
        4: 'Spring', 5: 'Spring', 6: 'Summer',
        7: 'Summer', 8: 'Summer', 9: 'Fall',
        10: 'Fall', 11: 'Fall', 12: 'Winter'
    }
    df['season'] = df['month'].map(season_map)
    
    # Create housing market season feature (U.S. housing market specific)
    market_season_map = {
        1: 'Off-Peak', 2: 'Off-Peak', 3: 'Spring Rush',
        4: 'Spring Rush', 5: 'Spring Rush', 6: 'Summer Peak',
        7: 'Summer Peak', 8: 'Summer Peak', 9: 'Fall Slow',
        10: 'Fall Slow', 11: 'Holiday Slowdown', 12: 'Holiday Slowdown'
    }
    df['market_season'] = df['month'].map(market_season_map)
    
    # Create month-related cyclical features (to handle circular nature of months)
    # This is better than using raw month numbers in models
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Create decade features
    df['decade'] = (df['year'] // 10) * 10
    
    # Create pre/post recession features (housing bubbles)
    recession_years = [2001, 2008, 2020]
    for year in recession_years:
        df[f'post_recession_{year}'] = (df['year'] >= year).astype(int)
    
    logger.info(f"Created seasonal features: year, month, quarter, season, market_season, cyclical features")
    
    return df

def create_market_indicators(data, price_col='CSUSHPISA', date_col='observation_date', window=12):
    """
    Create housing market indicator features.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing index data
    price_col : str, optional
        Name of the price column
    date_col : str, optional
        Name of the date column  
    window : int, optional
        Window size for rolling calculations
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added market indicator features
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Check if required columns exist
    if price_col not in df.columns or date_col not in df.columns:
        logger.error(f"Required columns {price_col} and/or {date_col} not found in data")
        return df
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            logger.error(f"Could not convert {date_col} to datetime")
            return df
    
    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Calculate price trend (percent change over window)
    df['price_trend'] = df[price_col].pct_change(window) * 100
    
    # Calculate price volatility (standard deviation over mean in window)
    df['price_volatility'] = df[price_col].rolling(window).std() / df[price_col].rolling(window).mean() * 100
    
    # Fill NaN values in price_volatility with median
    df['price_volatility'].fillna(df['price_volatility'].median(), inplace=True)
    
    # Calculate momentum (difference between short and long term changes)
    short_window = 3
    long_window = 12
    df['price_momentum'] = df[price_col].pct_change(short_window) - df[price_col].pct_change(long_window)
    
    # Create market condition indicator based on trend and volatility
    conditions = [
        (df['price_trend'] > 5) & (df['price_volatility'] < 5),  # Strong market
        (df['price_trend'] > 2) & (df['price_volatility'] < 7),  # Healthy market
        (df['price_trend'] < -2),  # Declining market
        (df['price_volatility'] > 10)  # Volatile market
    ]
    choices = ['Strong', 'Healthy', 'Declining', 'Volatile']
    df['market_condition'] = np.select(conditions, choices, default='Stable')
    
    logger.info(f"Created market indicator features: price_trend, price_volatility, price_momentum, and market_condition")
    
    return df

def create_rolling_statistics(data, price_col='CSUSHPISA', windows=[3, 6, 12, 24]):
    """
    Create rolling statistics features.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing data
    price_col : str, optional
        Name of the price column
    windows : list, optional
        List of window sizes for rolling calculations
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added rolling statistics features
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Check if required column exists
    if price_col not in df.columns:
        logger.error(f"Required column {price_col} not found in data")
        return df
    
    # Create rolling statistics for each window
    for window in windows:
        # Rolling mean
        df[f'rolling_mean_{window}m'] = df[price_col].rolling(window=window).mean()
        
        # Rolling standard deviation
        df[f'rolling_std_{window}m'] = df[price_col].rolling(window=window).std()
        
        # Rolling min/max
        df[f'rolling_min_{window}m'] = df[price_col].rolling(window=window).min()
        df[f'rolling_max_{window}m'] = df[price_col].rolling(window=window).max()
        
        # Distance from rolling mean (as percentage)
        df[f'dist_from_mean_{window}m'] = (df[price_col] - df[f'rolling_mean_{window}m']) / df[f'rolling_mean_{window}m'] * 100
    
    # Fill NaN values with appropriate methods
    # For means, use forward filling
    mean_cols = [col for col in df.columns if 'mean' in col]
    df[mean_cols] = df[mean_cols].fillna(method='ffill')
    
    # For other stats, use median of the column
    stat_cols = [col for col in df.columns if any(x in col for x in ['std', 'min', 'max', 'dist'])]
    for col in stat_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    logger.info(f"Created rolling statistics features for windows: {windows}")
    
    return df

def create_polynomial_features(data, columns, degree=2, include_bias=False, interaction_only=False):
    """
    Create polynomial features from existing features.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing data
    columns : list
        List of column names to create polynomial features from
    degree : int, optional
        Degree of polynomial features
    include_bias : bool, optional
        Whether to include a bias column
    interaction_only : bool, optional
        Whether to include only interaction terms
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added polynomial features
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Check if all required columns exist
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Columns {missing_cols} not found in data")
        return df
    
    # Check for NaN values in the columns
    has_nan = df[columns].isna().any().any()
    if has_nan:
        logger.warning(f"NaN values found in columns for polynomial features. Filling with median values.")
        # Fill NaN values with median for each column
        for col in columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.info(f"Filled NaN values in column {col} with median: {median_val}")
                else:
                    # For non-numeric columns, drop from polynomial feature creation
                    logger.warning(f"Column {col} is non-numeric with NaN values. Removing from polynomial features.")
                    columns.remove(col)
    
    # If no columns left after removing non-numeric ones with NaNs
    if not columns:
        logger.error("No valid columns left for polynomial feature creation")
        return df
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=interaction_only)
    
    try:
        poly_features = poly.fit_transform(df[columns])
        
        # Get feature names
        feature_names = poly.get_feature_names_out(columns)
        
        # Add polynomial features to dataframe
        poly_df = pd.DataFrame(poly_features, columns=feature_names)
        
        # Remove the original features (already in the dataframe)
        for col in columns:
            if col in poly_df.columns:
                poly_df = poly_df.drop(columns=[col])
        
        # Concatenate with original dataframe
        df = pd.concat([df, poly_df], axis=1)
        
        logger.info(f"Created {len(poly_df.columns)} polynomial features of degree {degree}")
        
    except Exception as e:
        logger.error(f"Error creating polynomial features: {str(e)}")
    
    return df

def prepare_csushpisa_data(data, date_col='observation_date', price_col='CSUSHPISA', create_all_features=True):
    """
    Prepare CSUSHPISA data with all relevant features for housing price analysis.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw CSUSHPISA data
    date_col : str, optional
        Name of the date column
    price_col : str, optional 
        Name of the price column
    create_all_features : bool, optional
        Whether to create all features or just the basic ones
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with all features created
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Check if required columns exist
    if price_col not in df.columns or date_col not in df.columns:
        logger.error(f"Required columns {price_col} and/or {date_col} not found in data")
        return df
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            logger.error(f"Could not convert {date_col} to datetime")
            return df
    
    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # 1. Create price change features
    df = create_price_change_features(df, price_col, date_col)
    
    # 2. Create seasonal features
    df = create_seasonal_features(df, date_col)
    
    if create_all_features:
        # 3. Create market indicators
        df = create_market_indicators(df, price_col, date_col)
        
        # 4. Create rolling statistics
        df = create_rolling_statistics(df, price_col)
        
        # 5. Create polynomial features for key numeric columns
        numeric_cols = ['price_mom_pct_change', 'price_yoy_pct_change', 'month_sin', 'month_cos']
        df = create_polynomial_features(df, numeric_cols, degree=2)
    
    logger.info(f"Created full feature set for CSUSHPISA data with {len(df.columns)} features")
    
    return df 