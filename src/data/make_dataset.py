"""
Module: make_dataset.py
Description: Functions for downloading and creating datasets

Functions:
- download_dataset: Download housing data from online sources
- create_synthetic_data: Create synthetic housing data for testing
- merge_datasets: Merge multiple housing datasets
- resample_timeseries: Resample time series data to different frequencies
- split_by_region: Split dataset into regional subsets
- create_dataset_metadata: Create metadata for dataset
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import requests
from urllib.parse import urlparse
import zipfile
import shutil
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_dataset(url, save_path, extract=False, extract_path=None):
    """
    Download housing data from online sources.
    
    Parameters:
    -----------
    url : str
        URL to download the dataset from
    save_path : str
        Path to save the downloaded file
    extract : bool, optional
        Whether to extract the downloaded file if it's an archive
    extract_path : str, optional
        Path to extract the archive to
        
    Returns:
    --------
    str
        Path to the downloaded/extracted data
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Parse URL to get filename
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    logger.info(f"Downloading data from {url}")
    
    # Download the file
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded data saved to {save_path}")
        
        # Extract if requested
        if extract:
            if extract_path is None:
                extract_path = os.path.dirname(save_path)
            
            if save_path.endswith('.zip'):
                with zipfile.ZipFile(save_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                logger.info(f"Extracted ZIP archive to {extract_path}")
                return extract_path
            else:
                logger.warning("Extraction requested but file is not a supported archive format")
        
        return save_path
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading data: {e}")
        return None

def create_synthetic_data(n_samples=1000, start_date='2010-01-01', 
                         end_date='2023-12-31', include_features=True,
                         save_path=None, seed=42):
    """
    Create synthetic housing data for testing.
    
    Parameters:
    -----------
    n_samples : int, optional
        Number of samples to generate
    start_date : str, optional
        Start date for time series
    end_date : str, optional
        End date for time series
    include_features : bool, optional
        Whether to include additional features
    save_path : str, optional
        Path to save the generated data
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        Synthetic housing data
    """
    # Set random seed
    np.random.seed(seed)
    
    # Generate dates
    date_range = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    # Base price with upward trend and seasonal pattern
    time_index = np.arange(n_samples)
    trend = 100000 + time_index * 100  # Upward trend
    seasonal = 10000 * np.sin(2 * np.pi * time_index / 365)  # Yearly seasonality
    
    # Create prices with trend, seasonality, and noise
    base_price = trend + seasonal
    noise = np.random.normal(0, 5000, n_samples)
    price = base_price + noise
    
    # Create dataframe
    df = pd.DataFrame({
        'date': date_range,
        'price': price
    })
    
    # Add additional features if requested
    if include_features:
        # House size (square feet)
        df['sqft'] = np.random.normal(1800, 400, n_samples)
        df['sqft'] = df['sqft'].clip(lower=600)  # Ensure no negative sizes
        
        # Number of bedrooms
        df['bedrooms'] = np.random.choice([1, 2, 3, 4, 5], size=n_samples, 
                                          p=[0.1, 0.2, 0.4, 0.2, 0.1])
        
        # Number of bathrooms
        df['bathrooms'] = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], size=n_samples,
                                           p=[0.1, 0.15, 0.3, 0.2, 0.15, 0.05, 0.05])
        
        # Property age
        current_year = pd.Timestamp.now().year
        df['year_built'] = np.random.randint(1950, current_year, n_samples)
        
        # Location (region)
        regions = ['North', 'South', 'East', 'West', 'Central']
        df['region'] = np.random.choice(regions, size=n_samples)
        
        # Property type
        property_types = ['Single Family', 'Townhouse', 'Condo', 'Multi-Family']
        df['property_type'] = np.random.choice(property_types, size=n_samples, 
                                              p=[0.6, 0.2, 0.15, 0.05])
        
        # Add property condition
        conditions = ['Excellent', 'Good', 'Fair', 'Poor']
        df['condition'] = np.random.choice(conditions, size=n_samples, 
                                           p=[0.3, 0.5, 0.15, 0.05])
        
        # Add price adjustment based on features
        df['price'] += df['sqft'] * 100  # $100 per square foot
        df['price'] += df['bedrooms'] * 15000  # $15k per bedroom
        df['price'] += (df['bathrooms'] * 10000).astype(int)  # $10k per bathroom
        
        # Adjust price based on age
        age_factor = (current_year - df['year_built']) / 10  # Decades old
        df['price'] -= age_factor * 5000  # $5k less per decade of age
        
        # Adjust price based on condition
        condition_map = {'Excellent': 50000, 'Good': 25000, 'Fair': 0, 'Poor': -25000}
        df['price'] += df['condition'].map(condition_map)
        
        # Regional price adjustments
        region_map = {'North': 25000, 'South': -10000, 'East': 15000, 'West': 40000, 'Central': 0}
        df['price'] += df['region'].map(region_map)
        
        # Property type adjustments
        type_map = {'Single Family': 25000, 'Townhouse': 0, 'Condo': -15000, 'Multi-Family': 50000}
        df['price'] += df['property_type'].map(type_map)
    
    # Ensure prices are positive and round to nearest dollar
    df['price'] = np.round(df['price'].clip(lower=50000))
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Created synthetic dataset with {n_samples} samples")
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved synthetic data to {save_path}")
    
    return df

def merge_datasets(datasets, on=None, how='inner', save_path=None):
    """
    Merge multiple housing datasets.
    
    Parameters:
    -----------
    datasets : list of pandas.DataFrame or list of str
        List of dataframes or paths to CSV files
    on : str or list, optional
        Column(s) to merge on
    how : str, optional
        Type of merge to perform
    save_path : str, optional
        Path to save the merged data
        
    Returns:
    --------
    pandas.DataFrame
        Merged dataset
    """
    # Load datasets if paths are provided
    dataframes = []
    for ds in datasets:
        if isinstance(ds, str):
            try:
                df = pd.read_csv(ds)
                dataframes.append(df)
                logger.info(f"Loaded dataset from {ds}")
            except Exception as e:
                logger.error(f"Error loading dataset {ds}: {e}")
                continue
        else:
            dataframes.append(ds)
    
    if len(dataframes) < 2:
        logger.error("Need at least two datasets to merge")
        return None
    
    # Perform merge
    merged_df = dataframes[0]
    for i, df in enumerate(dataframes[1:], 1):
        try:
            merged_df = pd.merge(merged_df, df, on=on, how=how)
            logger.info(f"Merged dataset {i}")
        except Exception as e:
            logger.error(f"Error merging dataset {i}: {e}")
            continue
    
    logger.info(f"Merged dataset has shape {merged_df.shape}")
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        merged_df.to_csv(save_path, index=False)
        logger.info(f"Saved merged data to {save_path}")
    
    return merged_df

def resample_timeseries(data, date_column, target_column, frequency='M', 
                        aggregation='mean', save_path=None):
    """
    Resample time series data to different frequencies.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Time series housing data
    date_column : str
        Name of the date column
    target_column : str
        Name of the target column to aggregate
    frequency : str, optional
        Target frequency ('D' for daily, 'W' for weekly, 'M' for monthly, 'Q' for quarterly, 'Y' for yearly)
    aggregation : str or dict, optional
        Aggregation method or dictionary mapping columns to methods
    save_path : str, optional
        Path to save the resampled data
        
    Returns:
    --------
    pandas.DataFrame
        Resampled time series data
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Set date as index
    df = df.set_index(date_column)
    
    # Determine aggregation method
    if isinstance(aggregation, str):
        # If a single method is provided, apply to target column only
        agg_dict = {target_column: aggregation}
        
        # For numeric columns other than target, use mean by default
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            if col != target_column:
                agg_dict[col] = 'mean'
    else:
        # Use provided aggregation dictionary
        agg_dict = aggregation
    
    # Resample data
    resampled = df.resample(frequency).agg(agg_dict)
    
    logger.info(f"Resampled data to {frequency} frequency")
    
    # Reset index to make date a column again
    resampled = resampled.reset_index()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        resampled.to_csv(save_path, index=False)
        logger.info(f"Saved resampled data to {save_path}")
    
    return resampled

def split_by_region(data, region_column, save_dir=None):
    """
    Split dataset into regional subsets.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing data
    region_column : str
        Name of the column containing region information
    save_dir : str, optional
        Directory to save the regional datasets
        
    Returns:
    --------
    dict
        Dictionary mapping region names to DataFrames
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Get unique regions
    regions = df[region_column].unique()
    
    # Create dictionary to store regional dataframes
    regional_dfs = {}
    
    # Split data by region
    for region in regions:
        region_df = df[df[region_column] == region].reset_index(drop=True)
        regional_dfs[region] = region_df
        
        logger.info(f"Created dataset for region '{region}' with {len(region_df)} samples")
        
        # Save if directory provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            # Create safe filename from region name
            safe_name = ''.join(c if c.isalnum() else '_' for c in str(region))
            file_path = os.path.join(save_dir, f"{safe_name}.csv")
            region_df.to_csv(file_path, index=False)
            logger.info(f"Saved dataset for region '{region}' to {file_path}")
    
    return regional_dfs

def create_dataset_metadata(data, description="Housing Price Dataset", 
                           target_column=None, save_path=None):
    """
    Create metadata for dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing data
    description : str, optional
        Description of the dataset
    target_column : str, optional
        Name of the target column
    save_path : str, optional
        Path to save the metadata
        
    Returns:
    --------
    dict
        Metadata dictionary
    """
    # Create metadata dictionary
    metadata = {
        "description": description,
        "dimensions": data.shape,
        "columns": {},
        "statistics": {},
        "created_at": pd.Timestamp.now().isoformat()
    }
    
    # Add target column if provided
    if target_column:
        metadata["target_column"] = target_column
    
    # Add column information
    for col in data.columns:
        col_type = str(data[col].dtype)
        unique_values = data[col].nunique()
        missing_values = data[col].isnull().sum()
        
        metadata["columns"][col] = {
            "type": col_type,
            "unique_values": unique_values,
            "missing_values": missing_values
        }
        
        # Add basic statistics for numeric columns
        if pd.api.types.is_numeric_dtype(data[col]):
            metadata["statistics"][col] = {
                "min": float(data[col].min()),
                "max": float(data[col].max()),
                "mean": float(data[col].mean()),
                "median": float(data[col].median()),
                "std": float(data[col].std())
            }
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved metadata to {save_path}")
    
    return metadata

def main():
    """
    Main function to demonstrate dataset creation.
    """
    # Example usage
    logger.info("Creating synthetic housing dataset")
    
    # Create project directories
    raw_dir = os.path.join('data', 'raw')
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Create synthetic data
    synthetic_data_path = os.path.join(raw_dir, 'housing_data.csv')
    data = create_synthetic_data(n_samples=5000, 
                                save_path=synthetic_data_path)
    
    # Create metadata
    metadata_path = os.path.join(raw_dir, 'housing_data_metadata.json')
    create_dataset_metadata(data, 
                           description="Synthetic Housing Price Dataset",
                           target_column="price",
                           save_path=metadata_path)
    
    logger.info("Dataset creation complete")

if __name__ == "__main__":
    main() 