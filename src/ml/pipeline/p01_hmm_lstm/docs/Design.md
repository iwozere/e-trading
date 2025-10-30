# HMM-LSTM Pipeline Design Document

## Overview

The HMM-LSTM pipeline is a comprehensive machine learning system for financial time series analysis and prediction. It combines Hidden Markov Models (HMM) for market regime detection with Long Short-Term Memory (LSTM) networks for price prediction, using Optuna for hyperparameter optimization of both technical indicators and model architecture.

## Pipeline Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│ Data Loader │───▶│ HMM Training │───▶│ HMM Apply   │───▶│ Indicators  │
│ (x_01)      │    │ (x_02)       │    │ (x_03)      │    │ Optimization│
└─────────────┘    └──────────────┘    └─────────────┘    │ (x_04)      │
                                                          └─────────────┘
                                                                    │
┌─────────────┐    ┌──────────────┐    ┌─────────────┐            │
│ Validation  │◀───│ LSTM Training│◀───│ LSTM Opt.   │◀───────────┘
│ (x_07)      │    │ (x_06)       │    │ (x_05)      │
└─────────────┘    └──────────────┘    └─────────────┘
```

## Stage 1: Data Loading (x_01_data_loader.py)

### Purpose
Downloads historical OHLCV data for multiple symbols and timeframes from various data providers.

### Input
- **Configuration**: `config/pipeline/p01.yaml`
- **Data Sources**: Multi-provider configuration (Binance, Yahoo Finance, etc.)

### Output
**File Format**: CSV files in `data/raw/`
**Naming Convention**: `{provider}_{symbol}_{timeframe}_{start_date}_{end_date}.csv`

**Columns**:
- `timestamp`: ISO format datetime
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price  
- `close`: Closing price
- `volume`: Trading volume
- `log_return`: Natural logarithm of price returns (computed if not present)

### Data Providers
- **Binance**: Crypto pairs (BTCUSDT, LTCUSDT, etc.)
- **Yahoo Finance**: Stocks (VT, PSNY, etc.)
- **Alpha Vantage**: Forex and stocks
- **Finnhub**: Market data
- **Polygon**: Financial data
- **Twelve Data**: Market data
- **CoinGecko**: Crypto data

### Rate Limiting
- **Binance**: 1200 requests/minute
- **Yahoo Finance**: 1 request/second
- **Alpha Vantage**: 5 requests/minute (free tier)

## Stage 2: HMM Training (x_02_train_hmm.py)

### Purpose
Trains Hidden Markov Models to detect market regimes using technical indicators and price data.

### Input
- **Raw Data**: CSV files from Stage 1
- **Configuration**: HMM parameters from config

### Technical Indicators Used for HMM

#### Core Indicators (Fixed Parameters)
- **RSI**: Relative Strength Index
  - Period: 14 (scaled by timeframe)
  - Range: 0-100
- **ATR**: Average True Range
  - Period: 14 (scaled by timeframe)
  - Measures volatility
- **Bollinger Bands**
  - Period: 20 (scaled by timeframe)
  - Standard deviation: 2
  - Components: Upper, Middle, Lower bands
- **Volume SMA**: Simple Moving Average of Volume
  - Period: 14 (scaled by timeframe)

#### Dynamic Feature Scaling
The pipeline scales indicator periods based on timeframe:
```python
scale_factor = (timeframe_minutes / multiplier) ** 0.5
scaled_period = max(2, int(round(base_period * scale_factor)))
```

### HMM Configuration
- **Components**: 3 (configurable)
- **Covariance Type**: "diag" (diagonal covariance matrices)
- **Algorithm**: "viterbi" (for state sequence decoding)
- **Iterations**: 100
- **Training Window**: 730 days (2 years)

### Output
**Model Files**: `src/ml/pipeline/p01_hmm_lstm/models/hmm/hmm_{symbol}_{timeframe}_{timestamp}.pkl`

**Model Package Contents**:
- `model`: Trained GaussianHMM object
- `scaler`: StandardScaler for feature normalization
- `features`: List of feature names used
- `config`: HMM configuration parameters
- `state_mapping`: Mapping from HMM states to regime numbers
- `training_stats`: Training statistics and regime analysis

**Regime Mapping**:
- States are mapped to numeric regimes (0-7) based on mean returns
- Higher regime numbers typically correspond to higher returns
- Regime 0: Lowest returns (bearish)
- Regime 1-6: Intermediate returns (sideways/bullish)
- Regime 7: Highest returns (bullish)

## Stage 3: HMM Application (x_03_apply_hmm.py)

### Purpose
Applies trained HMM models to generate regime labels for each time step.

### Input
- **Raw Data**: CSV files from Stage 1
- **HMM Models**: Pickle files from Stage 2

### Process
1. Loads trained HMM model
2. Computes technical indicators using same parameters as training
3. Applies HMM to predict regime states
4. Generates regime labels and confidence scores

### Output
**Labeled Data Files**: `data/labeled/{provider}_{symbol}_{timeframe}_{start_date}_{end_date}_labeled.csv`

**Columns**:
- All original OHLCV columns
- `log_return`: Natural logarithm of price returns
- `regime`: Numeric regime label (0-7)
- `regime_confidence`: Confidence score for regime prediction
- `regime_duration`: Duration of current regime
- Technical indicators used for HMM training

## Stage 4: Indicator Optimization (x_04_optuna_indicators.py)

### Purpose
Optimizes technical indicator parameters using Optuna to maximize trading performance.

### Input
- **Labeled Data**: CSV files from Stage 3
- **Configuration**: Optuna parameters from config

### Technical Indicators Optimized

#### RSI (Relative Strength Index)
- **Parameter**: `rsi_period`
- **Range**: 5-50
- **Default**: 14
- **Purpose**: Momentum oscillator

#### Bollinger Bands
- **Parameters**: 
  - `bb_period`: 10-50 (default: 20)
  - `bb_std`: 1.0-3.0 (default: 2.0)
- **Components**: Upper, Middle, Lower bands
- **Derived Features**:
  - `bb_position`: Position within bands (0-1)
  - `bb_width`: Band width relative to price

### Optimization Objective
Multi-objective optimization using:
- **Total Return**: Primary metric (60% weight)
- **Profit Factor**: Risk-adjusted returns (30% weight)
- **Maximum Drawdown**: Risk management (10% weight)

### Trading Strategy
Simple rule-based strategy:
- **Buy Signal**: RSI < 30 OR BB position < 0.2
- **Sell Signal**: RSI > 70 OR BB position > 0.8
- **Hold**: No clear signal

### Output
**Parameter Files**: `src/ml/pipeline/p01_hmm_lstm/models/lstm/indicators_{symbol}_{timeframe}_{timestamp}.json`

**JSON Structure**:
```json
{
  "best_params": {
    "rsi_period": 14,
    "bb_period": 20,
    "bb_std": 2.0
  },
  "best_value": 0.156,
  "optimization_history": [...],
  "performance_metrics": {
    "sharpe_ratio": 1.23,
    "profit_factor": 1.45,
    "max_drawdown": 0.12,
    "total_return": 0.156
  }
}
```

## Stage 5: LSTM Optimization (x_05_optuna_lstm.py)

### Purpose
Optimizes LSTM model hyperparameters using Optuna for time series prediction.

### Input
- **Labeled Data**: CSV files from Stage 3
- **Optimized Indicators**: JSON files from Stage 4
- **Configuration**: LSTM and Optuna parameters

### Technical Indicators Applied (Optimized Parameters)

#### Core Optimized Indicators
- **RSI Optimized**: `rsi_optimized` (using optimized period)
- **Bollinger Bands Optimized**:
  - `bb_upper_opt`, `bb_middle_opt`, `bb_lower_opt`
  - `bb_position_opt`, `bb_width_opt`
- **MACD Optimized**:
  - `macd_opt`, `macd_signal_opt`, `macd_histogram_opt`
  - Parameters: `macd_fast`, `macd_slow`, `macd_signal`
- **EMA Optimized**:
  - `ema_fast_opt`, `ema_slow_opt`, `ema_spread_opt`
  - Parameters: `ema_fast`, `ema_slow`
- **ATR Optimized**: `atr_opt` (using optimized period)
- **Stochastic Optimized**:
  - `stoch_k_opt`, `stoch_d_opt`
  - Parameters: `stoch_k`, `stoch_d`
- **Williams %R Optimized**: `williams_r_opt`
- **MFI Optimized**: `mfi_opt` (Money Flow Index)
- **SMA Optimized**: `sma_opt`

### LSTM Hyperparameters Optimized

#### Architecture Parameters
- **Sequence Length**: 20-120 (default: 60)
- **Hidden Size**: 32-256 (default: 64)
- **Number of Layers**: 1-4 (default: 2)
- **Dropout**: 0.1-0.5 (default: 0.2)

#### Training Parameters
- **Batch Size**: 16-128 (default: 32)
- **Learning Rate**: 0.0001-0.01 (default: 0.001)
- **Epochs**: 20-100 (default: 50)

### Feature Selection
**Priority Order**:
1. Base OHLCV features: `open`, `high`, `low`, `close`, `volume`, `log_return`
2. Regime features: `regime`, `regime_confidence`, `regime_duration`
3. Optimized indicators (suffix `_opt`)
4. Time features: `hour_sin`, `hour_cos`, `day_of_week_sin`, `day_of_week_cos`
5. Additional technical features (fallback)

### Optimization Objective
Multi-objective optimization:
- **MSE**: Mean Squared Error (primary)
- **Directional Accuracy**: Percentage of correct direction predictions
- **Sharpe Ratio**: Risk-adjusted returns

### Output
**Parameter Files**: `src/ml/pipeline/p01_hmm_lstm/models/lstm/lstm_params_{symbol}_{timeframe}_{timestamp}.json`

**JSON Structure**:
```json
{
  "best_params": {
    "sequence_length": 60,
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 50
  },
  "best_value": 0.000234,
  "optimization_history": [...],
  "feature_importance": {...}
}
```

## Stage 6: LSTM Training (x_06_train_lstm.py)

### Purpose
Trains the final LSTM model using optimized hyperparameters and indicators.

### Input
- **Labeled Data**: CSV files from Stage 3
- **Optimized Indicators**: JSON files from Stage 4
- **Optimized LSTM Parameters**: JSON files from Stage 5

### LSTM Architecture

#### Model Structure
```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, n_regimes=3):
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        # Output layer with regime conditioning
        self.linear = nn.Linear(hidden_size + n_regimes, output_size)
```

#### Regime Integration
- Regime information is one-hot encoded
- Concatenated with LSTM output before final prediction
- Enables regime-aware predictions

### Training Process
1. **Data Preparation**: Apply optimized indicators, create sequences
2. **Feature Scaling**: StandardScaler for features, MinMaxScaler for targets
3. **Sequence Creation**: Sliding window approach
4. **Train/Validation Split**: 80/20 split
5. **Training**: Adam optimizer with early stopping
6. **Learning Rate Scheduling**: Reduce on plateau

### Output
**Model Files**: `src/ml/pipeline/p01_hmm_lstm/models/lstm/lstm_{symbol}_{timeframe}_{timestamp}.pkl`

**Model Package Contents**:
- `model_state_dict`: PyTorch model weights
- `model_architecture`: Model configuration
- `hyperparameters`: Training parameters
- `features`: List of feature names
- `scalers`: Feature and target scalers
- `training_results`: Training metrics and history
- `optimization_results`: Best parameters from optimization

## Stage 7: Model Validation (x_07_validate_lstm.py)

### Purpose
Validates trained LSTM models against naive baselines and generates comprehensive reports with detailed visualizations.

### Input
- **Labeled Data**: CSV files from Stage 3
- **Trained LSTM Models**: Pickle files from Stage 6
- **Optimization Parameters**: JSON files from Stages 4 and 5

### Validation Process
1. **Data Loading**: Load labeled data and trained models
2. **Feature Preparation**: Apply optimized indicators and same preprocessing as training
3. **Prediction Generation**: Generate LSTM predictions on test data
4. **Baseline Comparison**: Compare against naive baseline (previous log return)
5. **Performance Analysis**: Calculate comprehensive metrics
6. **Regime Analysis**: Analyze performance by market regime
7. **Visualization**: Create detailed charts and plots
8. **Report Generation**: Generate PNG files and JSON reports

### Performance Metrics
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination
- **Directional Accuracy**: Percentage of correct direction predictions
- **Hit Rate**: Percentage of predictions within tolerance
- **Sharpe Ratio**: Risk-adjusted returns
- **Performance Improvements**: Percentage improvements over baseline

### Output

#### PNG Visualization Files
**Location**: `reports/lstm_validation_{symbol}_{timeframe}_{timestamp}/`

**File Naming Convention**: `{chart_type}_{symbol}_{timeframe}_{timestamp}.png`

**Available Charts**:

1. **`predictions_time_series.png`** - Time Series Comparison
   - **Purpose**: Shows how well the LSTM predictions track actual log returns over time
   - **Colors**: Red (Actual), Blue (LSTM Prediction), Orange (Naive Baseline)
   - **Font Sizes**: Legend 16pt, Titles 24pt, Axis labels 20pt
   - **What to Look For**:
     - **Good**: LSTM line (blue) closely follows actual line (red) with minimal lag
     - **Concerning**: Large gaps between predicted and actual values, or systematic bias
     - **Baseline Comparison**: Naive baseline (orange) should generally be worse than LSTM
     - **Critical Issue**: If blue line (LSTM) is horizontal/flat, the model is not learning
   - **Interpretation**: If LSTM predictions are consistently far from actual values, the model may be overfitting or using inappropriate features. A horizontal blue line indicates the LSTM is predicting constant values and needs retraining.

2. **`predictions_scatter.png`** - Scatter Plot Analysis
   - **Purpose**: Shows the correlation between predicted and actual log returns
   - **Colors**: Blue dots (LSTM), Red dots (Baseline), Black dashed line (Perfect Prediction)
   - **Font Sizes**: Legend 16pt, Titles 24pt, Axis labels 20pt
   - **What to Look For**:
     - **Good**: Points cluster tightly around the diagonal line (perfect prediction)
     - **Concerning**: Points scattered widely, or clustering in specific regions
     - **R² Correlation**: Higher values indicate better predictive power
   - **Interpretation**: If points are widely scattered, the model has poor predictive accuracy. If points cluster in specific regions, the model may have bias.

3. **`error_distribution.png`** - Error Distribution Analysis
   - **Purpose**: Shows the distribution of prediction errors (predicted - actual)
   - **Colors**: Blue histogram (LSTM Errors), Red histogram (Baseline Errors)
   - **Font Sizes**: Legend 16pt, Titles 24pt, Axis labels 20pt
   - **What to Look For**:
     - **Good**: Bell-shaped curve centered near zero with small standard deviation
     - **Concerning**: Skewed distribution, multiple peaks, or wide spread
     - **Comparison**: LSTM errors should be smaller and more centered than baseline errors
   - **Interpretation**: Skewed distributions indicate systematic bias. Wide spreads indicate poor precision.

4. **`error_over_time.png`** - Error Tracking Over Time
   - **Purpose**: Shows how prediction accuracy varies over the test period
   - **Colors**: Blue line (LSTM |Error|), Red line (Baseline |Error|)
   - **Font Sizes**: Legend 16pt, Titles 24pt, Axis labels 20pt
   - **What to Look For**:
     - **Good**: Consistent, low error levels with occasional spikes
     - **Concerning**: Increasing error trends, or periods of consistently high errors
     - **Regime Patterns**: Errors may be higher during certain market regimes
   - **Interpretation**: Increasing errors suggest model degradation or changing market conditions

5. **`cumulative_error.png`** - Cumulative Squared Error
   - **Purpose**: Shows the cumulative prediction error over time
   - **Colors**: Blue line (LSTM Cumulative SE), Red line (Baseline Cumulative SE)
   - **Font Sizes**: Legend 16pt, Titles 24pt, Axis labels 20pt
   - **What to Look For**:
     - **Good**: LSTM line grows more slowly than baseline line
     - **Concerning**: LSTM line growing as fast or faster than baseline
     - **Trend Analysis**: Steeper slopes indicate periods of poor performance
   - **Interpretation**: If LSTM cumulative error grows as fast as baseline, the model provides no value

6. **`predictions_by_regime.png`** - Regime-Specific Performance
   - **Purpose**: Shows prediction accuracy broken down by market regime
   - **Colors**: Multi-color scheme (Red, Green, Blue, Orange, Purple, Brown, Pink, Gray) for different regimes
   - **Font Sizes**: Legend 16pt, Titles 24pt, Axis labels 20pt
   - **What to Look For**:
     - **Good**: Consistent accuracy across all regimes, or better performance in volatile regimes
     - **Concerning**: Poor performance in specific regimes, or regime-dependent bias
     - **Color Coding**: Different colors represent different market regimes
   - **Interpretation**: Poor performance in specific regimes suggests the model doesn't adapt well to different market conditions

7. **`rolling_performance.png`** - Rolling Performance Metrics
   - **Purpose**: Shows how model performance varies over rolling windows
   - **Colors**: Blue line (LSTM Rolling MSE), Red line (Baseline Rolling MSE)
   - **Font Sizes**: Legend 16pt, Titles 24pt, Axis labels 20pt
   - **What to Look For**:
     - **Good**: Consistent performance with occasional dips
     - **Concerning**: Declining performance trends, or high volatility in performance
     - **Window Size**: Uses 50-period rolling window (adjustable)
   - **Interpretation**: Declining trends suggest model degradation or changing market dynamics

### How to Interpret Poor Results

If your validation results don't look good, consider these factors:

#### 1. **Data Quality Issues**
- **Insufficient Data**: LSTM models need large datasets (typically 1000+ samples)
- **Data Leakage**: Check if future information is accidentally included in features
- **Regime Stability**: If market regimes change frequently, HMM may not capture them well

#### 2. **Model Configuration Problems**
- **Overfitting**: Model may be too complex for the available data
- **Underfitting**: Model may be too simple to capture market patterns
- **Feature Engineering**: Technical indicators may not be relevant for your specific market

#### 3. **Market-Specific Issues**
- **High Noise**: Some markets (especially crypto) are inherently noisy
- **Regime Changes**: Market behavior may have changed since training
- **Timeframe Mismatch**: Model may work better on different timeframes

#### 4. **Optimization Issues**
- **Poor Hyperparameters**: Optuna may not have found optimal parameters
- **Objective Function**: Optimization may not align with your trading goals
- **Feature Selection**: Important features may be missing or irrelevant

### Recommendations for Improvement

1. **Increase Training Data**: Use longer historical periods or multiple symbols
2. **Feature Engineering**: Add domain-specific features or different technical indicators
3. **Hyperparameter Tuning**: Increase Optuna trials or adjust parameter ranges
4. **Ensemble Methods**: Combine multiple models or timeframes
5. **Market Regime Analysis**: Focus on specific regimes where the model performs well

#### JSON Results
**Location**: `src/ml/pipeline/p01_hmm_lstm/models/lstm/lstm_validation_{symbol}_{timeframe}_{timestamp}.json`

**Structure**:
```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "validation_timestamp": "20250816_192813",
  "model_path": "path/to/model.pkl",
  "data_file": "path/to/data.csv",
  "performance_metrics": {
    "lstm_metrics": {
      "mse": 0.000234,
      "mae": 0.0123,
      "rmse": 0.0153,
      "r2": 0.0456,
      "directional_accuracy": 52.3,
      "hit_rate": 45.7,
      "sharpe_ratio": 0.234
    },
    "baseline_metrics": {
      "mse": 0.000245,
      "mae": 0.0134,
      "rmse": 0.0156,
      "r2": 0.0234,
      "directional_accuracy": 48.9,
      "hit_rate": 42.1,
      "sharpe_ratio": 0.123
    },
    "improvements": {
      "mse_improvement_pct": 4.5,
      "directional_accuracy_improvement": 3.4,
      "mae_improvement_pct": 8.2,
      "r2_improvement": 0.022
    }
  },
  "regime_performance": {
    "regime_0": {
      "sample_count": 1250,
      "mse": 0.000198,
      "mae": 0.0102,
      "directional_accuracy": 54.2,
      "r2": 0.0678
    }
  },
  "png_files": {
    "predictions_time_series": "path/to/predictions_time_series.png",
    "predictions_scatter": "path/to/predictions_scatter.png",
    "error_distribution": "path/to/error_distribution.png",
    "error_over_time": "path/to/error_over_time.png",
    "cumulative_error": "path/to/cumulative_error.png",
    "predictions_by_regime": "path/to/predictions_by_regime.png",
    "rolling_performance": "path/to/rolling_performance.png"
  },
  "model_metadata": {
    "architecture": {...},
    "hyperparameters": {...},
    "features_count": 45,
    "training_results": {...}
  }
}
```

### Technical Implementation Notes

#### Matplotlib Backend Configuration
The validation script uses the 'Agg' backend to prevent file handle issues on Windows systems:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

#### Visualization Improvements
The validation script has been optimized for better visual clarity and distinguishability:

- **Color Scheme**: Optimized colors for better distinguishability:
  - Blue: LSTM predictions/errors (consistent across all charts)
  - Red: Baseline predictions/errors (consistent across all charts)
  - Orange: Naive baseline in time series
  - Multi-color: Different regimes in regime analysis (Red, Green, Blue, Orange, Purple, Brown, Pink, Gray)
- **Font Sizes**: Enhanced readability with larger fonts:
  - Legend: 16pt (twice the default size)
  - Titles: 24pt
  - Axis labels: 20pt
- **Figure Quality**: High DPI (300) PNG output for crisp visualizations
- **Line Widths**: Increased to 1.5 for better visibility
- **Point Sizes**: Increased scatter plot point sizes to 30 for better visibility

#### Error Handling
- Robust error handling for PNG file generation
- Automatic figure cleanup to prevent memory leaks
- Graceful degradation if individual charts fail to generate

#### File Organization
- Each validation run creates a timestamped subdirectory
- PNG files are saved with descriptive names
- JSON results include paths to all generated visualizations

## Data Flow Summary

### Input Data Evolution
1. **Raw OHLCV**: Basic price and volume data
2. **Technical Indicators**: Computed for HMM training
3. **Regime Labels**: Added by HMM application
4. **Optimized Indicators**: Applied with best parameters
5. **Feature Matrix**: Final input for LSTM training

### Key Design Decisions

#### Indicator Parameter Optimization
- **Why Optimized**: Different markets and timeframes require different indicator parameters
- **Storage Strategy**: Parameters stored in JSON files, not in labeled data
- **Application**: Optimized parameters applied dynamically during LSTM training

#### Regime Integration
- **HMM States**: Mapped to numeric regimes (0-7) for LSTM compatibility
- **One-Hot Encoding**: Regime information integrated into LSTM architecture
- **Confidence Scores**: Additional regime features for model robustness

#### Feature Engineering
- **Dynamic Scaling**: Indicator periods scaled based on timeframe
- **Optimized Features**: Prioritized over fixed-period indicators
- **Fallback Strategy**: Additional features used if optimized not available

#### Model Persistence
- **Comprehensive Metadata**: All training parameters and results saved
- **Scaler Persistence**: Feature and target scalers saved for inference
- **Version Control**: Timestamped files for model versioning

This design ensures a robust, scalable pipeline that can adapt to different markets and timeframes while maintaining comprehensive documentation and reproducibility.
