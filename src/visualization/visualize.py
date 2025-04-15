"""
Module: visualize.py
Description: Functions for visualizing housing price data and model results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_price_distribution(data, price_col='price', by=None, bins=50, save_path=None):
    """
    Plot distribution of housing prices.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing price data
    price_col : str, optional
        Name of the price column
    by : str, optional
        Column to group by (e.g., 'region')
    bins : int, optional
        Number of bins for histogram
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    if by is None:
        # Simple histogram
        sns.histplot(data[price_col], bins=bins, kde=True)
        plt.title(f'Distribution of Housing Prices')
    else:
        # Grouped histogram
        if by in data.columns:
            sns.histplot(data=data, x=price_col, hue=by, bins=bins, kde=True)
            plt.title(f'Distribution of Housing Prices by {by}')
        else:
            logger.error(f"Column {by} not found in data")
            return
    
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def plot_price_trends(data, date_col='date', price_col='price', by=None, freq='M', save_path=None):
    """
    Plot time series trends of housing prices.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing price data
    date_col : str, optional
        Name of the date column
    price_col : str, optional
        Name of the price column
    by : str, optional
        Column to group by (e.g., 'region')
    freq : str, optional
        Frequency for resampling ('D', 'W', 'M', 'Q', 'Y')
    save_path : str, optional
        Path to save the plot
    """
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        try:
            data[date_col] = pd.to_datetime(data[date_col])
        except:
            logger.error(f"Could not convert {date_col} to datetime")
            return
    
    plt.figure(figsize=(12, 6))
    
    if by is None:
        # Resample and plot overall trend
        ts_data = data.set_index(date_col)[price_col].resample(freq).mean()
        plt.plot(ts_data.index, ts_data.values, marker='o', linestyle='-')
        plt.title(f'Housing Price Trends ({freq} frequency)')
    else:
        # Plot trends by group
        if by in data.columns:
            groups = data.groupby(by)
            for name, group in groups:
                ts_data = group.set_index(date_col)[price_col].resample(freq).mean()
                plt.plot(ts_data.index, ts_data.values, marker='o', linestyle='-', label=name)
            plt.title(f'Housing Price Trends by {by} ({freq} frequency)')
            plt.legend()
        else:
            logger.error(f"Column {by} not found in data")
            return
    
    plt.xlabel('Date')
    plt.ylabel('Average Price')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def plot_feature_correlation(data, target_col='price', top_n=10, save_path=None):
    """
    Plot correlation of features with target variable.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing price data
    target_col : str, optional
        Name of the target column
    top_n : int, optional
        Number of top correlations to show
    save_path : str, optional
        Path to save the plot
    """
    # Get numeric columns
    numeric_cols = data.select_dtypes(include=np.number).columns
    
    # Calculate correlation with target
    correlations = data[numeric_cols].corrwith(data[target_col]).sort_values(ascending=False)
    
    # Drop target column from the correlations
    correlations = correlations.drop(target_col, errors='ignore')
    
    # Select top N correlations
    top_correlations = correlations.abs().nlargest(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_correlations.values, y=top_correlations.index)
    plt.title(f'Top {top_n} Feature Correlations with {target_col}')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def plot_correlation_heatmap(data, target_col='price', save_path=None):
    """
    Plot correlation heatmap for all numeric features.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing price data
    target_col : str, optional
        Name of the target column (will be highlighted)
    save_path : str, optional
        Path to save the plot
    """
    # Get numeric columns
    numeric_cols = data.select_dtypes(include=np.number).columns
    
    # Calculate correlation matrix
    corr = data[numeric_cols].corr()
    
    # Create mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))
    
    # Draw the heatmap
    sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0,
               square=True, linewidths=.5, annot=False, cbar_kws={"shrink": .5})
    
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def plot_map_visualization(data, lat_col='latitude', lon_col='longitude', 
                         color_col='price', size_col=None, save_path=None):
    """
    Create an interactive map visualization of housing prices.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing price data
    lat_col : str, optional
        Name of the latitude column
    lon_col : str, optional
        Name of the longitude column
    color_col : str, optional
        Name of the column to use for color
    size_col : str, optional
        Name of the column to use for point size
    save_path : str, optional
        Path to save the plot
    """
    # Check if required columns exist
    required_cols = [lat_col, lon_col, color_col]
    if size_col:
        required_cols.append(size_col)
    
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        logger.error(f"Columns {missing_cols} not found in data")
        return
    
    # Create map
    if size_col:
        fig = px.scatter_mapbox(data, lat=lat_col, lon=lon_col, color=color_col, 
                              size=size_col, hover_name=color_col,
                              color_continuous_scale="Viridis", zoom=10)
    else:
        fig = px.scatter_mapbox(data, lat=lat_col, lon=lon_col, color=color_col, 
                              hover_name=color_col, color_continuous_scale="Viridis", 
                              zoom=10)
    
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Interactive map saved to {save_path}")
    
    fig.show()

def plot_model_predictions(y_true, y_pred, model_name='Model', save_path=None):
    """
    Plot actual vs predicted values.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted values
    model_name : str, optional
        Name of the model for the plot title
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'Actual vs Predicted Values ({model_name})')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def plot_forecast(historical_data, forecast_data, model_name='Model', intervals=None, save_path=None):
    """
    Plot historical data and forecast.
    
    Parameters:
    -----------
    historical_data : pandas.Series or pandas.DataFrame
        Historical time series data
    forecast_data : pandas.Series or pandas.DataFrame
        Forecasted data
    model_name : str, optional
        Name of the model for the plot title
    intervals : tuple, optional
        Tuple of (lower, upper) prediction intervals
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(historical_data.index, historical_data.values, 'b-', label='Historical Data')
    
    # Plot forecast
    plt.plot(forecast_data.index, forecast_data.values, 'r--', label='Forecast')
    
    # Add prediction intervals if provided
    if intervals:
        lower, upper = intervals
        plt.fill_between(forecast_data.index, lower, upper, color='r', alpha=0.1, label='Prediction Interval')
    
    plt.title(f'Time Series Forecast ({model_name})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def plot_model_comparison(models_data, metric='RMSE', save_path=None):
    """
    Plot comparison of multiple models.
    
    Parameters:
    -----------
    models_data : dict
        Dictionary with model names as keys and performance metrics as values
    metric : str, optional
        Metric to compare models on
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Extract model names and metric values
    models = list(models_data.keys())
    values = [models_data[model][metric] for model in models]
    
    # Create bar plot
    bars = plt.bar(models, values)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.title(f'Model Comparison by {metric}')
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def plot_feature_importance(feature_importance, top_n=10, model_name='Model', save_path=None):
    """
    Plot feature importance for a model.
    
    Parameters:
    -----------
    feature_importance : dict or pandas.Series
        Feature names and their importance scores
    top_n : int, optional
        Number of top features to show
    model_name : str, optional
        Name of the model for the plot title
    save_path : str, optional
        Path to save the plot
    """
    # Convert to Series if dict
    if isinstance(feature_importance, dict):
        importance = pd.Series(feature_importance)
    else:
        importance = feature_importance
    
    # Sort and get top features
    importance = importance.sort_values(ascending=False)
    top_features = importance.head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x=top_features.values, y=top_features.index)
    plt.title(f'Top {top_n} Feature Importance ({model_name})')
    plt.xlabel('Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()

def create_dashboard(data, output_path, title='Housing Price Analysis Dashboard'):
    """
    Create an interactive HTML dashboard for housing price analysis.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Housing price data
    output_path : str
        Path to save the HTML dashboard
    title : str, optional
        Title for the dashboard
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Price Distribution', 
            'Price Trends', 
            'Price by Property Type', 
            'Price by Bedrooms'
        ),
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "box"}, {"type": "bar"}]
        ]
    )
    
    # Add price distribution histogram
    fig.add_trace(
        go.Histogram(x=data['price'], nbinsx=50, name='Price Distribution'),
        row=1, col=1
    )
    
    # Add price trends over time
    if 'date' in data.columns:
        # Ensure date is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            try:
                date_col = pd.to_datetime(data['date'])
            except:
                logger.error("Could not convert date column to datetime")
                date_col = None
        else:
            date_col = data['date']
        
        if date_col is not None:
            # Group by month and calculate average price
            monthly_data = data.groupby(date_col.dt.to_period('M'))['price'].mean()
            monthly_data.index = monthly_data.index.to_timestamp()
            
            fig.add_trace(
                go.Scatter(
                    x=monthly_data.index, 
                    y=monthly_data.values, 
                    mode='lines+markers',
                    name='Monthly Avg Price'
                ),
                row=1, col=2
            )
    
    # Add price by property type box plot
    if 'property_type' in data.columns:
        for prop_type in data['property_type'].unique():
            fig.add_trace(
                go.Box(
                    y=data[data['property_type'] == prop_type]['price'],
                    name=prop_type
                ),
                row=2, col=1
            )
    
    # Add price by bedrooms bar chart
    if 'bedrooms' in data.columns:
        bedroom_data = data.groupby('bedrooms')['price'].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=bedroom_data['bedrooms'],
                y=bedroom_data['price'],
                name='Avg Price by Bedrooms'
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text=title,
        height=800,
        showlegend=False
    )
    
    # Save dashboard
    fig.write_html(output_path)
    logger.info(f"Dashboard saved to {output_path}")
    
    # Return figure for display
    return fig