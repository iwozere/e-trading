Here's the updated **Requirements.md** with **Optuna optimization** for both indicators and model parameters included:

---

```markdown
# Technical Requirements: Automated Trading Pipeline with HMM + Single LSTM + Optuna Optimization

## 1. General Idea
Goal — to build a trading pipeline where:

- **Hidden Markov Model (HMM)** determines the market regime (`regime`).
- **A single LSTM** is used to predict the next price (or log return).
- **LSTM receives the regime as an input feature**.
- **Optuna** is used for:
  - Hyperparameter optimization of indicators (RSI, Bollinger Bands, moving averages, etc.).
  - Hyperparameter optimization of the LSTM model.
- The result is used to generate trading signals.
- The model's performance is compared with a naive prediction (`previous close`).
- As an example you can check `src/ml/lstm/lstm_optuna_log_return_from_csv.py`.

---

## 2. Pipeline Structure
### Steps:

1. **Data Loading (`x_01_data_loader.py`)**
   - Load OHLCV data for required timeframes: `5m`, `15m`, `1h`, `4h`.
   - Save to: `data/<symbol>_<tf>_<start_date>_<end_date>.csv`.
   - As an example you can take `src/util/data_downloader.py` or directly call it just by changing TEST_SCENARIOS object.

2. **Preprocessing (`x_02_preprocess.py`)**
   - Add `log_return`, normalization, and rolling statistics.
   - Save to: `data/processed/`.

3. **HMM Training (`x_03_train_hmm.py`)**
   - Train on the last 1–3 years of data (depending on timeframe).
   - Number of hidden states fixed (`n_components = 3`).
   - Save model to: `src/ml/hmm/model/hmm_<symbol>_<tf>_<timestamp>.pt`.

4. **HMM Application (`x_04_apply_hmm.py`)**
   - Generate `regime` column for each candle.
   - Save updated CSV to: `data/labeled/`.

5. **Optuna Optimization for Indicators (`x_05_optuna_indicators.py`)**
   - Optimize parameters for technical indicators (e.g., RSI length, Bollinger Bands period, SMA/EMA periods).
   - Objective function uses backtest performance metrics (Sharpe ratio, profit factor, drawdown).
   - Save best indicator parameters to: `results/indicators_<symbol>_<tf>_<timestamp>.json`.

6. **Optuna Optimization for LSTM (`x_06_optuna_lstm.py`)**
   - Search optimal LSTM parameters:
     - Sequence length
     - Hidden size
     - Batch size
     - Learning rate
     - Dropout
     - Number of layers
   - Use Optuna's pruning and TPE sampler for efficiency.
   - Save best model parameters to: `results/lstm_params_<symbol>_<tf>_<timestamp>.json`.

7. **LSTM Training (`x_07_train_lstm.py`)**
   - Use columns: `close`, `log_return`, `regime`, and additional features.
   - Add `regime` as a categorical feature (one-hot or embedding).
   - LSTM predicts `close[t+1]` or `log_return[t+1]`.
   - Save model to: `src/ml/lstm/model/lstm_<symbol>_<tf>_<timestamp>.pt`.

8. **Validation & Testing (`x_08_validate_lstm.py`)**
   - Use a hold-out set (last 10–20% of data).
   - Compare:
     - `MSE(LSTM)` vs `MSE(naive_pred = close[t])`
   - Generate chart + PDF report.
   - Save to: `results/lstm_<symbol>_<tf>_<timestamp>.pdf`, `.png` and `.json`.
   - Save best parameters to JSON.

---

## 3. Special Notes
- Only **one LSTM model** is used (not separate models for each regime).
- The `regime` is included as a feature in `X`, it does not define the architecture.
- The strategy uses:
  - Optimized indicator parameters for feature generation.
  - LSTM forecast for trading signal generation.

---

## 4. Configuration & Parameters
Example config (`config/pipeline/p01.yaml`):

```yaml
symbols: [BTCUSDT, ETHUSDT, LTCUSDT]
timeframes: [5m, 15m, 1h, 4h]

hmm:
  n_components: 3
  train_window_days: 730 # 2 years

lstm:
  sequence_length: 60
  hidden_size: 64
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  validation_split: 0.2
  dropout: 0.2
  num_layers: 2

optuna:
  n_trials: 50
  sampler: tpe
  pruning: true

evaluation:
  test_split: 0.1
  baseline_model: naive
  metrics: [mse, directional_accuracy, sharpe_ratio]
```

---

## 5. Quality Evaluation

Main metrics:

* **MSE(LSTM)** vs **MSE(Naive)**
* **Directional Accuracy**
* **Sharpe Ratio**
* **Max Drawdown**
* **Profit Factor**

Charts:

* Market regimes overlaid on prices
* Predictions vs Actual values
* Error over time
* Backtest equity curve

---

## 6. File Structure

```
project/
├── data/
│   ├── processed/
│   └── labeled/
├── src/
│   ├── ml/
│   │   ├── hmm/
│   │   └── lstm/
├── results/
├── reports/
├── scripts/
├── config/
└── retrain_pipeline.sh
```

---

## 7. Update Frequency

| Component            | Retraining Frequency | Data Window          |
| -------------------- | -------------------- | -------------------- |
| HMM                  | Monthly / Quarterly  | 1–3 years            |
| Indicator Parameters | Weekly / Bi-weekly   | 1–6 months           |
| LSTM                 | Weekly / Bi-weekly   | 1–3 months (rolling) |

---

## 8. Error Handling Requirements

### 8.1 Fail-Fast Mode (Default)
- **Enabled by default**: Pipeline stops immediately when a critical stage fails
- **Prevents resource waste**: No time spent on downstream stages that depend on failed upstream stages
- **Clear error reporting**: Detailed error messages with recovery suggestions

### 8.2 Stage Criticality Classification

#### Critical Stages (Fail-Fast Enabled)
These stages are essential for the pipeline to produce valid results:

1. **Data Loading** (Stage 1) - Downloads OHLCV data
2. **Data Preprocessing** (Stage 2) - Adds features and indicators
3. **HMM Training** (Stage 3) - Trains regime detection models
4. **HMM Application** (Stage 4) - Applies HMM models to label data
5. **LSTM Training** (Stage 7) - Trains the main prediction model

#### Optional Stages (Can Fail Without Stopping Pipeline)
These stages enhance the pipeline but are not essential:

6. **Indicator Optimization** (Stage 5) - Optimizes technical indicator parameters
7. **LSTM Optimization** (Stage 6) - Optimizes LSTM hyperparameters
8. **Model Validation** (Stage 8) - Validates models and generates reports

### 8.3 Command Line Options
```bash
# Default behavior (fail-fast enabled)
python run_pipeline.py

# Disable fail-fast mode
python run_pipeline.py --no-fail-fast

# Continue even if optional stages fail
python run_pipeline.py --continue-on-optional-failures

# Skip optional stages to focus on core pipeline
python run_pipeline.py --skip-stages 5,6,8
```

### 8.4 Error Recovery
- **Check the error message**: Understand what went wrong
- **Review logs**: Look for detailed error information
- **Fix the issue**: Address the root cause (data, config, etc.)
- **Restart from failed stage**: Use `--skip-stages` to resume

---

## 9. Multi-Provider Data Support

### 9.1 Configuration Format
The pipeline supports multiple data providers with provider-specific configurations:

```yaml
# Multi-provider data configuration
data_sources:
  binance:
    symbols: [LTCUSDT, BTCUSDT]
    timeframes: [5m, 4h]
  yfinance:
    symbols: [AAPL, MSFT]
    timeframes: [4h, 1d]
  alphavantage:
    symbols: [EURUSD, GBPUSD]
    timeframes: [1h, 1d]
```

### 9.2 Supported Data Providers

| Provider | Code | Asset Types | API Key Required | Rate Limits | Best For |
|----------|------|-------------|------------------|-------------|----------|
| **Binance** | `binance` | Cryptocurrencies | No | 1200 req/min | Crypto trading |
| **Yahoo Finance** | `yfinance` | Stocks, ETFs | No | None | Stock analysis |
| **Alpha Vantage** | `alphavantage` | Stocks, Forex | Yes | 5 req/min (free) | Professional data |
| **Finnhub** | `finnhub` | Stocks | Yes | 60 req/min (free) | Real-time data |
| **Polygon.io** | `polygon` | US Stocks | Yes | 5 req/min (free) | US markets |
| **Twelve Data** | `twelvedata` | Global markets | Yes | 8 req/min (free) | Global coverage |
| **CoinGecko** | `coingecko` | Cryptocurrencies | No | 50 req/min | Crypto research |

### 9.3 File Naming Convention
Files are saved with provider prefix to avoid conflicts:

```
data/raw/
├── binance_BTCUSDT_5m_20230101_20231231.csv
├── binance_LTCUSDT_4h_20230101_20231231.csv
├── yfinance_AAPL_4h_20230101_20231231.csv
├── yfinance_MSFT_1d_20230101_20231231.csv
└── alphavantage_EURUSD_1h_20230101_20231231.csv
```

---

## 10. Rate Limiting Requirements

### 10.1 Rate Limiting Specifications

#### Binance
- **Rate Limit**: 1200 requests per minute
- **Bar Limit**: 1000 bars per request
- **Implementation**: 0.05 second delay between requests + automatic batching

#### Yahoo Finance
- **Rate Limit**: 1 request per second (recommended)
- **Bar Limit**: No specific limit, but rate limiting prevents violations
- **Implementation**: 1 second delay between requests

#### General Batch Processing
- **Default Rate Limit**: 100ms between symbol downloads
- **Configurable**: Each downloader can override the default
- **Progress Tracking**: Detailed logging of download progress

### 10.2 Batching Logic for 1000 Bar Limit
The pipeline must implement automatic batching to respect provider limits:

```python
def _calculate_batch_dates(self, start_date, end_date, interval):
    """Calculate batch dates to respect the 1000 bar limit."""
    # Convert interval to minutes for calculation
    interval_minutes = {
        '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
        '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
    }
    
    minutes_per_interval = interval_minutes.get(interval, 1440)
    max_bars = 1000
    
    # Calculate maximum time span for 1000 bars
    max_minutes = minutes_per_interval * max_bars
    max_timedelta = timedelta(minutes=max_minutes)
    
    batches = []
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(current_start + max_timedelta, end_date)
        batches.append((current_start, current_end))
        current_start = current_end
        
    return batches
```

---

## 11. HMM Regime Analysis Requirements

### 11.1 Regime Detection Behavior by Timeframe

#### **5m Timeframes**
- ✅ Usually shows all three regimes clearly
- High volatility allows distinct bearish/bullish/sideways states
- Short-term noise creates clear regime separation

#### **15m Timeframes**
- ✅ Generally shows all three regimes
- Good balance of volatility and trend detection
- Clear regime differentiation

#### **1h Timeframes**
- ✅ Usually shows all three regimes
- Moderate volatility with trend persistence
- Good regime separation

#### **4h Timeframes**
- ⚠️ May show only 2 regimes (Sideways + Bullish)
- Lower volatility, longer trends
- All regimes might have positive returns
- **Solution**: Improved labeling logic handles this case

#### **1d Timeframes**
- ⚠️ Often shows only 2 regimes
- Very low volatility, long-term trends
- May need different approach (2-regime model)

### 11.2 Improved Labeling Logic
The pipeline must implement dynamic thresholds based on data characteristics:

```python
# Dynamic thresholds based on data characteristics
if self.n_components == 3:
    # Calculate return range for better thresholds
    returns = [stat['avg_return'] for stat in regime_stats]
    min_return = min(returns)
    max_return = max(returns)
    return_range = max_return - min_return
    
    # Dynamic thresholds based on data characteristics
    if return_range > 0.001:  # Good separation between regimes
        # Use relative positioning with dynamic thresholds
        for i, stat in enumerate(regime_stats):
            avg_return = stat['avg_return']
            relative_position = (avg_return - min_return) / return_range
            
            if i == 0:  # Lowest return regime
                if relative_position < 0.3:  # Bottom 30% of return range
                    labels[stat['regime_id']] = 'Bearish'
                else:
                    labels[stat['regime_id']] = 'Sideways'
            elif i == 1:  # Middle return regime
                if relative_position > 0.7:  # Top 30% of return range
                    labels[stat['regime_id']] = 'Bullish'
                else:
                    labels[stat['regime_id']] = 'Sideways'
            else:  # Highest return regime
                labels[stat['regime_id']] = 'Bullish'
    else:
        # Poor separation - use absolute thresholds and ensure all three labels
        for i, stat in enumerate(regime_stats):
            avg_return = stat['avg_return']
            
            if i == 0:  # Always label lowest as Bearish or Sideways
                if avg_return < 0:
                    labels[stat['regime_id']] = 'Bearish'
                else:
                    labels[stat['regime_id']] = 'Sideways'
            elif i == 1:  # Middle gets Sideways or Bullish
                if avg_return > 0.0005:  # Higher threshold for bullish
                    labels[stat['regime_id']] = 'Bullish'
                else:
                    labels[stat['regime_id']] = 'Sideways'
            else:  # Highest always gets Bullish
                labels[stat['regime_id']] = 'Bullish'
```

### 11.3 Debugging Requirements
The pipeline must provide comprehensive debugging information:

- Overall data statistics (mean, std, negative returns percentage)
- Detailed regime analysis for each detected state
- Bearish candidate identification
- Comprehensive logging for troubleshooting
- Timeframe-specific information
- Sample counts per regime
- Return thresholds and volatility metrics
- Regime distribution statistics
- Return range analysis and labeling method used

### 11.4 Configuration Recommendations

#### For 4h Timeframes
```yaml
hmm:
  n_components: 3  # Keep 3, let improved labeling handle it
  covariance_type: 'full'  # Better for complex regimes
  n_iter: 200  # More iterations for convergence
```

#### Alternative: 2-Regime Model
```yaml
hmm:
  n_components: 2  # Simpler model for 4h
  covariance_type: 'diag'  # Faster, sufficient for 2 regimes
```

---

## 12. Exit Codes and Logging

### 12.1 Exit Codes
- **0**: Pipeline completed successfully
- **1**: Pipeline failed (critical stage failure or user interruption)

### 12.2 Logging Levels
The pipeline provides comprehensive logging:

- **Info level**: Normal operation progress
- **Warning level**: Optional stage failures, non-critical issues
- **Error level**: Critical stage failures, fatal errors
- **Debug level**: Detailed execution information

### 12.3 Configuration
Error handling behavior can be configured in the pipeline configuration file:

```yaml
# config/pipeline/p01.yaml
pipeline:
  fail_fast: true  # Default: true
  continue_on_optional_failures: false  # Default: false
  max_retries: 3  # Future: retry failed stages
```

---

## 13. Troubleshooting Requirements

### 13.1 Common Issues

1. **Data not found**: Ensure data loading stage completed successfully
2. **Configuration errors**: Check YAML syntax and required fields
3. **Memory issues**: Reduce batch sizes or use fewer symbols/timeframes
4. **Network timeouts**: Increase timeout values for data downloads
5. **Rate limit violations**: Check provider-specific rate limits
6. **Missing bearish regimes**: Use improved labeling logic for longer timeframes

### 13.2 Getting Help

1. **Check logs**: Review detailed error messages
2. **Validate requirements**: Run `python run_pipeline.py --validate-only`
3. **List stages**: Run `python run_pipeline.py --list-stages`
4. **Test with minimal setup**: Use single symbol/timeframe for testing
5. **Debug regime analysis**: Check regime detection logs for timeframe-specific behavior
