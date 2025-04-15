"""
Module: preprocess.py
Description: Functions for preprocessing housing price data

Functions:
- load_data: Load housing data from CSV file
- clean_data: Clean the raw housing data
- handle_missing_values: Deal with missing values in the dataset
- remove_outliers: Remove outliers from the dataset
- normalize_data: Normalize/standardize numeric features
- encode_categorical: Encode categorical variables
- create_time_features: Create time-based features from date columns
- create_lagged_features: Create lagged features for time series analysis
- train_test_split_time: Split data into train/test sets respecting time order
- save_processed_data: Save the processed data to disk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from scipy import stats
import os
from pathlib import Path
import pickle

def load_data(file_path):
    """
    Load housing data from a CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    # Check file extension and read accordingly
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    print(f"Loaded data with shape {data.shape}")
    
    # Display basic info
    print("\nData Types:")
    print(data.dtypes)
    
    print("\nSample Data:")
    print(data.head())
    
    return data

def clean_data(data, drop_columns=None):
    """
    Clean the raw housing data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw housing data
    drop_columns : list, optional
        List of columns to drop
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned data
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Drop specified columns
    if drop_columns is not None:
        df = df.drop(columns=drop_columns, errors='ignore')
        print(f"Dropped columns: {drop_columns}")
    
    # Convert date columns to datetime
    date_columns = df.select_dtypes(include=['object']).columns
    for col in date_columns:
        # Try to infer if column is a date
        try:
            df[col] = pd.to_datetime(df[col])
            print(f"Converted {col} to datetime")
        except:
            pass
    
    # Check for duplicate rows
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        print(f"Found {n_duplicates} duplicate rows. Removing...")
        df = df.drop_duplicates().reset_index(drop=True)
    
    return df

def handle_missing_values(data, strategy='simple', categorical_strategy='most_frequent', numeric_strategy='mean'):
    """
    Deal with missing values in the dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing data
    strategy : str, optional
        Strategy for handling missing values ('simple', 'knn', 'drop')
    categorical_strategy : str, optional
        Strategy for categorical variables ('most_frequent', 'constant')
    numeric_strategy : str, optional
        Strategy for numeric variables ('mean', 'median', 'constant')
        
    Returns:
    --------
    pandas.DataFrame
        Data with missing values handled
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Check missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    
    if len(missing) == 0:
        print("No missing values found")
        return df
    
    print("\nMissing Values:")
    print(missing)
    
    if strategy == 'drop':
        # Drop rows with missing values
        df = df.dropna().reset_index(drop=True)
        print(f"Dropped {len(data) - len(df)} rows with missing values")
        
    elif strategy == 'simple':
        # Split data into numeric and categorical columns
        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(exclude=np.number).columns
        
        if len(numeric_cols) > 0:
            # Impute numeric columns
            numeric_imputer = SimpleImputer(strategy=numeric_strategy)
            df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
        
        if len(categorical_cols) > 0:
            # Impute categorical columns
            cat_imputer = SimpleImputer(strategy=categorical_strategy)
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            
    elif strategy == 'knn':
        # KNN imputation works best on normalized data
        # Split data into numeric and categorical
        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(exclude=np.number).columns
        
        # Handle categorical data first with simple imputation
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy=categorical_strategy)
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            
            # Encode categorical for KNN
            for col in categorical_cols:
                if df[col].dtype == 'object':
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
        
        # Now use KNN imputation
        knn_imputer = KNNImputer(n_neighbors=5)
        df_imputed = knn_imputer.fit_transform(df)
        
        # Convert back to DataFrame with original columns
        df = pd.DataFrame(df_imputed, columns=df.columns)
    
    return df

def remove_outliers(data, columns=None, method='zscore', threshold=3):
    """
    Remove outliers from the dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing data
    columns : list, optional
        List of columns to check for outliers, defaults to all numeric columns
    method : str, optional
        Method for detecting outliers ('zscore', 'iqr')
    threshold : float, optional
        Threshold for outlier detection
        
    Returns:
    --------
    pandas.DataFrame
        Data with outliers removed
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns
    
    # Track outliers
    outlier_indices = set()
    
    for col in columns:
        if method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(df[col]))
            outliers = np.where(z_scores > threshold)[0]
        elif method == 'iqr':
            # IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        
        print(f"Found {len(outliers)} outliers in column '{col}'")
        outlier_indices.update(outliers)
    
    # Remove outliers
    df_clean = df.drop(index=outlier_indices).reset_index(drop=True)
    print(f"Removed {len(outlier_indices)} rows with outliers")
    
    return df_clean

def normalize_data(data, columns=None, method='standard', return_scaler=False):
    """
    Normalize/standardize numeric features.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing data
    columns : list, optional
        List of columns to normalize, defaults to all numeric columns
    method : str, optional
        Method for normalization ('standard', 'minmax')
    return_scaler : bool, optional
        Whether to return the scaler object
        
    Returns:
    --------
    pandas.DataFrame or tuple
        Normalized data and optionally the scaler object
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # If no columns specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns
    
    # Initialize scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Fit and transform
    df[columns] = scaler.fit_transform(df[columns])
    
    if return_scaler:
        return df, scaler
    else:
        return df

def encode_categorical(data, columns=None, method='onehot', drop_first=True):
    """
    Encode categorical variables.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing data
    columns : list, optional
        List of categorical columns to encode
    method : str, optional
        Method for encoding ('onehot', 'label')
    drop_first : bool, optional
        Whether to drop the first category in one-hot encoding
        
    Returns:
    --------
    pandas.DataFrame
        Data with encoded categorical variables
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # If no columns specified, use all object and category columns
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns
    
    # Convert datetime columns to string to avoid encoding error
    for col in columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    
    if method == 'onehot':
        # One-hot encoding
        for col in columns:
            # Get dummies and add prefix to column names
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
            # Concat to original dataframe
            df = pd.concat([df, dummies], axis=1)
            # Drop original column
            df = df.drop(columns=[col])
    
    elif method == 'label':
        # Label encoding
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    return df

def create_time_features(data, date_column):
    """
    Create time-based features from date columns.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing data
    date_column : str
        Name of the date column
        
    Returns:
    --------
    pandas.DataFrame
        Data with additional time features
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract date components
    df[f'{date_column}_year'] = df[date_column].dt.year
    df[f'{date_column}_month'] = df[date_column].dt.month
    df[f'{date_column}_quarter'] = df[date_column].dt.quarter
    
    # Add cyclical encoding for month (sine and cosine transformations)
    # This captures the cyclical nature of months better than plain integers
    df[f'{date_column}_month_sin'] = np.sin(2 * np.pi * df[date_column].dt.month / 12)
    df[f'{date_column}_month_cos'] = np.cos(2 * np.pi * df[date_column].dt.month / 12)
    
    # Add day of week for daily data if applicable
    if df[date_column].dt.normalize().nunique() > 100:  # More than 100 unique days suggests daily data
        df[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
        
        # Add cyclical encoding for day of week
        df[f'{date_column}_dayofweek_sin'] = np.sin(2 * np.pi * df[date_column].dt.dayofweek / 7)
        df[f'{date_column}_dayofweek_cos'] = np.cos(2 * np.pi * df[date_column].dt.dayofweek / 7)
    
    return df

def create_lagged_features(data, target_column, lags=3, drop_na=True):
    """
    Create lagged features for time series analysis.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing data, should be sorted by date
    target_column : str
        Column to create lags for
    lags : int or list, optional
        Number of lags or list of lag periods
    drop_na : bool, optional
        Whether to drop rows with NaN values created by lagging
        
    Returns:
    --------
    pandas.DataFrame
        Data with lagged features
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Create lag features
    if isinstance(lags, int):
        lags = range(1, lags + 1)
    
    for lag in lags:
        df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
    
    # Drop rows with NaN values if requested
    if drop_na:
        df = df.dropna().reset_index(drop=True)
    
    return df

def train_test_split_time(data, test_size=0.2, date_column=None):
    """
    Split data into train/test sets respecting time order.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing data
    test_size : float, optional
        Fraction of data to use for testing
    date_column : str, optional
        Name of date column to sort by
        
    Returns:
    --------
    tuple
        (train_data, test_data)
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Sort by date if date column provided
    if date_column is not None:
        df = df.sort_values(date_column).reset_index(drop=True)
    
    # Calculate split point
    split_idx = int(len(df) * (1 - test_size))
    
    # Split data
    train_data = df.iloc[:split_idx].reset_index(drop=True)
    test_data = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    return train_data, test_data

def save_processed_data(data, file_path, create_dir=True):
    """
    Save the processed data to disk.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Processed data
    file_path : str
        Path to save the data
    create_dir : bool, optional
        Whether to create directory if it doesn't exist
        
    Returns:
    --------
    bool
        True if successful
    """
    # Create directory if it doesn't exist
    if create_dir:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Determine file format from extension
    if file_path.endswith('.csv'):
        data.to_csv(file_path, index=False)
    elif file_path.endswith('.pkl'):
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    elif file_path.endswith('.parquet'):
        data.to_parquet(file_path, index=False)
    else:
        data.to_csv(file_path, index=False)  # Default to CSV
    
    print(f"Data saved to {file_path}")
    
    return True

def exploratory_data_analysis(data, target_column=None, save_dir=None):
    """
    Perform exploratory data analysis on housing data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing data
    target_column : str, optional
        Name of the target column
    save_dir : str, optional
        Directory to save plots
        
    Returns:
    --------
    dict
        Dictionary with EDA results
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Create save directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Basic statistics
    print("Basic Statistics:")
    print(df.describe().T)
    
    # Correlation analysis
    numeric_cols = df.select_dtypes(include=np.number).columns
    corr = df[numeric_cols].corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'correlation_matrix.png'))
    
    # Distribution of target variable if provided
    if target_column and target_column in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[target_column], kde=True)
        plt.title(f'Distribution of {target_column}')
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'{target_column}_distribution.png'))
    
    # Return EDA results
    eda_results = {
        'basic_stats': df.describe().T,
        'correlation': corr,
        'missing_values': df.isnull().sum(),
        'data_types': df.dtypes
    }
    
    return eda_results 