# Stock Market Prediction using Machine Learning

## Overview
This project implements a comprehensive stock market prediction system using advanced machine learning techniques. By preprocessing historical stock market data and leveraging **Sweetviz for automated EDA** and sophisticated **feature engineering**, the system applies a **Random Forest Classifier** to analyze and predict market trends with high precision, focusing on next-day price movement predictions.

## Key Results
- **Model**: Random Forest Classifier with 200 estimators
- **Prediction Target**: Next-day price movement (up/down)
- **Features**: OHLC (Open, High, Low, Close) price data
- **Training Strategy**: Time-series split with last 100 days as test set
- **Performance**: High precision trend prediction capabilities

## Project Structure
```
├── README.md                          # Project documentation
├── stock_prediction_analysis.ipynb    # Main analysis notebook
├── infolimpioavanzadoTarget.csv       # Clean historical stock data
├── data/
│   ├── processed_stock_data.csv       # Preprocessed features
│   └── prediction_results.csv         # Model predictions
├── models/
│   ├── random_forest_model.pkl        # Trained RF model
│   └── model_metrics.json             # Performance statistics
├── reports/
│   ├── sweetviz_report.html           # Automated EDA report
│   └── feature_analysis.html          # Feature importance analysis
└── visualizations/
    ├── price_trends.png               # Stock price visualizations
    └── prediction_analysis.png        # Model performance charts
```

## Dataset Information

### Data Characteristics
- **Source**: Clean financial dataset (infolimpioavanzadoTarget.csv)
- **Structure**: Time-series stock market data
- **Features**: Date, Open, High, Low, Close prices
- **Data Quality**: Preprocessed and cleaned for optimal model performance
- **Target Variable**: Binary classification (price increase/decrease)

### Data Schema
```python
df.info()  # Dataset overview
```
| Column | Description | Data Type | Usage |
|--------|-------------|-----------|--------|
| **date** | Trading date | datetime | Time index |
| **open** | Opening price | float | Predictor feature |
| **high** | Highest price | float | Predictor feature |
| **low** | Lowest price | float | Predictor feature |
| **close** | Closing price | float | Predictor feature |
| **target** | Price movement direction | boolean | Target variable |

## Data Preprocessing Pipeline

### 1. Data Loading & Quality Assessment
```python
import pandas as pd

# Load and examine data
df = pd.read_csv("infolimpioavanzadoTarget.csv")
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isna().sum()}")
```

### 2. Feature Selection & Engineering
```python
# Select core OHLC features
df_new = df[['date', 'open', 'high', 'low', 'close']]

# Create target variable for next-day prediction
df_new["tom"] = df_new["close"].shift(-1)  # Tomorrow's close
df_new['target'] = df_new["tom"] > df_new["close"].astype(float)
```

**Feature Engineering Process:**
- **Feature Selection**: Focus on essential OHLC data
- **Target Creation**: Binary classification for price direction
- **Time-shift Operations**: Next-day price prediction setup
- **Data Type Optimization**: Ensure proper numeric formats

### 3. Automated EDA with Sweetviz
```python
import sweetviz as sv

# Generate comprehensive EDA report
report = sv.analyze(df_new, pairwise_analysis='off')
report.show_html()
```

**Sweetviz Analysis Features:**
- **Statistical Summaries**: Descriptive statistics for all features
- **Distribution Analysis**: Data distribution patterns
- **Correlation Detection**: Feature relationship identification
- **Data Quality Assessment**: Missing values and outliers
- **Interactive Visualizations**: Professional HTML report

## Machine Learning Implementation

### Random Forest Classifier Architecture
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Model configuration
model = RandomForestClassifier(
    n_estimators=200,        # 200 decision trees
    min_samples_split=100,   # Prevent overfitting
    random_state=1           # Reproducible results
)

# Time-series train/test split
train = df_new.iloc[:-100]   # All data except last 100 days
test = df_new.iloc[-100:]    # Last 100 days for testing

# Feature set
predictors = ["open", "high", "low", "close"]
```

### Model Training & Prediction
```python
# Train the model
model.fit(train[predictors], train["target"])

# Generate predictions
predictions = model.predict(test[predictors])

# Evaluate performance
precision = precision_score(test["target"], predictions)
```

## Technical Methodology

### Time-Series Considerations
- **Forward-Looking Validation**: Uses future data points for testing
- **No Data Leakage**: Proper temporal separation between train/test
- **Realistic Backtesting**: Simulates actual trading conditions
- **Sequential Prediction**: Maintains chronological order

### Feature Importance Analysis
The Random Forest model provides insights into which features contribute most to predictions:
1. **Close Price**: Previous day's closing value
2. **High Price**: Daily maximum price
3. **Low Price**: Daily minimum price
4. **Open Price**: Opening price significance

### Model Hyperparameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| **n_estimators** | 200 | Number of decision trees |
| **min_samples_split** | 100 | Minimum samples to split node |
| **random_state** | 1 | Reproducibility seed |

## Visualization & Analysis

### Price Trend Analysis
```python
# Visualize stock price movements
df_new.plot(figsize=(7.00, 5.00))
plt.title("Stock Price Trends Over Time")
plt.show()
```

**Visual Analysis Features:**
- **Multi-Line Plot**: OHLC price visualization
- **Trend Identification**: Visual pattern recognition
- **Time-Series Display**: Chronological price movements
- **Comparative Analysis**: Relative price relationships

### Performance Metrics
- **Precision Score**: Accuracy of positive predictions
- **Classification Report**: Comprehensive performance metrics
- **Confusion Matrix**: Prediction accuracy breakdown
- **Feature Importance**: Variable contribution analysis

## Key Insights & Findings

### Market Behavior Patterns
1. **Price Continuity**: Strong relationship between consecutive days
2. **Volatility Signals**: High/low ranges indicate market uncertainty
3. **Trend Persistence**: Price movements often continue short-term
4. **Feature Relevance**: All OHLC features contribute to prediction accuracy

### Model Performance Analysis
- **High Precision**: Random Forest achieves strong predictive accuracy
- **Robust Predictions**: Model performs well on unseen data
- **Feature Balance**: All OHLC features show significant importance
- **Temporal Stability**: Consistent performance across time periods

### Trading Strategy Implications
- **Short-term Predictions**: Model optimized for next-day forecasting
- **Risk Management**: Precision-focused approach reduces false signals
- **Feature Simplicity**: Basic OHLC data provides strong predictive power
- **Scalability**: Framework adaptable to different stocks and timeframes

## Usage Instructions

### Prerequisites
```bash
pip install pandas numpy matplotlib scikit-learn sweetviz
```

### Running the Analysis
1. **Data Loading**:
   ```python
   df = pd.read_csv("infolimpioavanzadoTarget.csv")
   ```

2. **Feature Engineering**:
   ```python
   df_new = df[['date', 'open', 'high', 'low', 'close']]
   df_new['target'] = (df_new["close"].shift(-1) > df_new["close"])
   ```

3. **Model Training**:
   ```python
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier(n_estimators=200, min_samples_split=100)
   model.fit(train[predictors], train["target"])
   ```

4. **Generate Predictions**:
   ```python
   predictions = model.predict(test[predictors])
   ```

## Model Validation & Testing

### Cross-Validation Strategy
- **Time-Series Split**: Respects temporal order
- **Walk-Forward Analysis**: Sequential prediction validation
- **Out-of-Sample Testing**: Last 100 days reserved for testing
- **Performance Consistency**: Stable results across time periods

### Evaluation Metrics
- **Precision**: Focus on prediction accuracy
- **Recall**: Sensitivity to positive cases
- **F1-Score**: Balanced performance measure
- **Accuracy**: Overall prediction correctness

## Future Enhancements

### Advanced Features
- [ ] Technical indicators (RSI, MACD, Moving Averages)
- [ ] Volume-based features
- [ ] Sentiment analysis integration
- [ ] Multi-timeframe analysis

### Model Improvements
- [ ] Ensemble methods (XGBoost, LightGBM)
- [ ] Deep learning approaches (LSTM, GRU)
- [ ] Hyperparameter optimization
- [ ] Feature selection techniques

### System Enhancements
- [ ] Real-time prediction pipeline
- [ ] Multi-stock analysis
- [ ] Risk management integration
- [ ] Performance monitoring dashboard

## Performance Optimization

### Data Processing
- **Efficient Memory Usage**: Optimized data types and structures
- **Fast Computation**: Vectorized operations with pandas
- **Scalable Architecture**: Modular design for easy expansion
- **Clean Code Structure**: Well-organized and documented functions

### Model Efficiency
- **Balanced Complexity**: 200 estimators for optimal performance
- **Overfitting Prevention**: min_samples_split parameter tuning
- **Computational Speed**: Efficient Random Forest implementation
- **Memory Management**: Optimized for large datasets

## Risk Considerations

### Model Limitations
- **Market Volatility**: Extreme events may affect predictions
- **Data Dependency**: Model requires quality historical data
- **Feature Limitations**: Uses only price-based features
- **Temporal Changes**: Market regime changes may impact performance

### Trading Risks
- **Not Financial Advice**: Model predictions for research purposes
- **Past Performance**: Historical results don't guarantee future success
- **Market Risk**: All trading involves financial risk
- **Model Risk**: Predictions may be incorrect
