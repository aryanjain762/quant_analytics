# Housing Price Analysis

A comprehensive analysis and prediction toolkit for housing prices using various statistical and machine learning approaches.

## Project Overview

This project provides a complete pipeline for analyzing housing market data and building predictive models. It includes data preprocessing, feature engineering, model training, evaluation, and visualization components.

## Features

- **Data Processing**: Clean, transform, and prepare housing data for analysis
- **Feature Engineering**: Create advanced features for better model performance
- **Multiple Models**:
  - Linear Regression
  - Ridge and Lasso Regression
  - Random Forest
  - Gradient Boosting
  - ARIMAX (time series)
  - VAR (vector autoregression)
  - LSTM (deep learning)
- **Evaluation**: Comprehensive model evaluation metrics
- **Visualization**: Interactive and static visualizations of housing data and model results
- **Regional Analysis**: Support for analyzing housing markets by region

## Project Structure

```
housing_price_analysis/
├── data/
│   ├── raw/                      # Original unprocessed data
│   │   └── housing_data.csv      # The dataset
│   ├── processed/                # Cleaned and preprocessed data
│   └── external/                 # Data from external sources if needed
├── notebooks/
│   ├── 01_data_exploration.ipynb # Initial EDA
│   ├── 02_data_preprocessing.ipynb  # Data cleaning and feature engineering
│   ├── 03_model_development.ipynb   # Building different models
│   ├── 04_model_evaluation.ipynb    # Comparing model performance
│   └── 05_regional_analysis.ipynb   # Analysis by housing market types
├── src/
│   ├── data/
│   │   ├── make_dataset.py       # Script to download or generate data
│   │   └── preprocess.py         # Data preprocessing code
│   ├── features/
│   │   └── build_features.py     # Feature engineering code
│   ├── models/
│   │   ├── train_model.py        # Training code for models
│   │   ├── predict_model.py      # Making predictions with trained models
│   │   └── evaluate_model.py     # Model evaluation metrics
│   └── visualization/
│       └── visualize.py          # Visualization functions
├── models/                       # Saved model files
├── reports/
│   └── figures/                  # Generated graphics and figures
├── requirements.txt              # pip requirements
├── README.md                     # Project description
└── main.py                       # Entry point for running the full pipeline
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/housing_price_analysis.git
   cd housing_price_analysis
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Full Pipeline

The main.py script provides a complete pipeline from data processing to model evaluation:

```
python main.py --data_path data/raw/housing_data.csv --models linear,ridge,rf --output_dir reports
```

Options:
- `--data_path`: Path to housing data CSV file (if file doesn't exist, synthetic data will be created)
- `--models`: Comma-separated list of models to train (options: linear, ridge, lasso, rf, gbm, arimax, var, lstm)
- `--output_dir`: Directory to save results and reports

### Using Individual Components

You can also use individual components from the library:

```python
from src.data import preprocess
from src.features import build_features
from src.models import train_model, predict_model, evaluate_model
from src.visualization import visualize

# Load and preprocess data
data = preprocess.load_data('data/raw/housing_data.csv')
data = preprocess.clean_data(data)

# Build features
data = build_features.create_price_per_sqft(data)
data = build_features.create_seasonal_features(data, date_col='date')

# Train a model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = train_model.train_random_forest(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
metrics = evaluate_model.calculate_regression_metrics(y_test, y_pred)

# Create visualizations
visualize.plot_model_predictions(y_test, y_pred, model_name='Random Forest')
```

## Data

The project can work with any housing dataset that includes price information. The code is designed to handle:
- Property characteristics (size, bedrooms, etc.)
- Location information (coordinates, regions, zip codes)
- Time series data (with date columns)

If no dataset is provided, the system can generate synthetic data for testing and demonstration.

## Models

### Regression Models
- **Linear Regression**: Simple baseline model
- **Ridge Regression**: Linear model with L2 regularization
- **Lasso Regression**: Linear model with L1 regularization and feature selection
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Boosted trees with gradient descent

### Time Series Models
- **ARIMAX**: Autoregressive Integrated Moving Average with eXogenous variables
- **VAR**: Vector Autoregression for multivariate time series

### Deep Learning
- **LSTM**: Long Short-Term Memory network for sequential data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- TensorFlow
- Statsmodels
- Plotly 

## Output on Local Machine

INFO - Loaded CSUSHPISA data with 457 records from 1987-01-01 to 2025-01-01
INFO - Basic statistics of CSUSHPISA data:
INFO - Shape: (457, 2)
INFO - Data types: observation_date    datetime64[ns]
                  CSUSHPISA           float64
                  dtype: object
INFO - Summary statistics:
                    observation_date   CSUSHPISA
count                            457  457.000000
mean   2005-12-31 04:15:13.785558016  148.157418
min              1987-01-01 00:00:00   63.963000
25%              1996-07-01 00:00:00   82.930000
50%              2006-01-01 00:00:00  142.905000
75%              2015-07-01 00:00:00  182.797000
max              2025-01-01 00:00:00  329.441000
std                              NaN   68.504037
INFO - Missing values:
observation_date    0
CSUSHPISA           0
dtype: int64
INFO - Created price change features: mom, qoq, yoy and cumulative growth
INFO - Created seasonal features: year, month, quarter, season, market_season, cyclical features
INFO - Completed exploratory data analysis and saved visualizations
INFO - Performing time series analysis
INFO - ADF Statistic: 1.150343979280158
INFO - p-value: 0.9956241926342921
INFO - ADF Statistic (differenced): -3.1730461697685985
INFO - p-value (differenced): 0.02159885241536847
INFO - Completed time series analysis
INFO - Building forecast model to predict 36 months ahead
INFO - ARIMA Model Summary: [ARIMA details displayed]
INFO - Forecast accuracy - MAE: [value], RMSE: [value]
INFO - Completed forecast for next 36 months
INFO - Building regression models
INFO - Created full feature set for CSUSHPISA data with [number] features
INFO - Linear Regression Metrics:
INFO - R² Score: [value]
INFO - MAE: [value]
INFO - RMSE: [value]
INFO - Random Forest Metrics:
INFO - R² Score: [value]
INFO - MAE: [value]
INFO - RMSE: [value]
INFO - Completed regression modeling
INFO - Housing price analysis completed successfully 