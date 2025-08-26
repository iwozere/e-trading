# HMM-LSTM Trading Strategy Implementation

This directory contains the implementation of trading strategies that use the HMM-LSTM pipeline models for market prediction and trading decisions.

## Overview

The HMM-LSTM strategy combines:
- **Hidden Markov Models (HMM)** for market regime detection
- **LSTM Neural Networks** for price prediction
- **Optimized Technical Indicators** from the pipeline
- **Regime-aware decision making** for enhanced performance

## Implementation Approaches

### 1. Standalone Strategy (`hmm_lstm_strategy.py`)

A complete standalone strategy that directly uses the trained models.

**Features:**
- Direct model loading and inference
- Regime-aware position sizing
- Dynamic exit conditions based on regime changes
- Comprehensive risk management

**Usage:**
```python
# Configure and run with backtester
strategy_config = {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
            "models_dir": "src/ml/pipeline/p01_hmm_lstm/models",
    "prediction_threshold": 0.001,
    "regime_confidence_threshold": 0.6
}

# Use with test script
python src/strategy/test_hmm_lstm_strategy.py --data data.csv --variant balanced
```

### 2. Entry Mixin (`entry/hmm_lstm_entry_mixin.py`)

An entry mixin that integrates with the existing CustomStrategy framework.

**Features:**
- Modular design compatible with existing exit strategies
- Automatic model loading and feature calculation
- Clean separation of entry logic from exit logic
- Easy configuration through JSON files

**Usage:**
```python
# Register the mixin
from src.strategy.entry.register_hmm_lstm_mixin import register_hmm_lstm_mixin
register_hmm_lstm_mixin()

# Use with CustomStrategy
strategy_config = {
    "entry_logic": {
        "name": "HMMLSTMEntryMixin",
        "params": {
            "symbol": "BTCUSDT",
            "prediction_threshold": 0.001,
            "regime_confidence_threshold": 0.6
        }
    },
    "exit_logic": {
        "name": "TrailingStopExitMixin",
        "params": {"trail_percent": 3.0}
    }
}
```

## Configuration Files

### 1. `config/strategy/hmm_lstm_pipeline.json`
Comprehensive configuration with multiple variants and advanced features.

### 2. `config/strategy/hmm_lstm_simple.json`
Simplified configuration for basic usage and testing.

### 3. `config/strategy/hmm_lstm_mixin.json`
Configuration for using the entry mixin with existing exit strategies.

## Key Components

### Model Loading
- Automatically finds and loads the latest trained models
- Handles HMM models, LSTM models, and optimized parameters
- Graceful fallback when models are not available

### Feature Engineering
- Uses optimized technical indicator parameters from the pipeline
- Calculates real-time indicators using TA-Lib
- Maintains feature buffers for sequence-based predictions

### Regime Detection
- Real-time HMM inference for market regime classification
- Confidence scoring for regime predictions
- Regime transition detection and handling

### Price Prediction
- LSTM-based next-period price change prediction
- Regime-aware predictions using one-hot encoding
- Proper feature scaling and inverse transformation

### Risk Management
- Dynamic position sizing based on prediction confidence
- Regime-specific risk parameters
- Multiple exit conditions (profit target, stop loss, trailing stop)

## Strategy Variants

### Conservative
- Higher confidence thresholds
- Smaller position sizes
- Tighter risk management
- Focus on high-probability trades

### Balanced
- Moderate thresholds and position sizes
- Balanced risk-reward parameters
- Good for general market conditions

### Aggressive
- Lower confidence thresholds
- Larger position sizes
- Wider stop losses for trend following
- Higher frequency trading

### Multi-Symbol
- Portfolio approach across multiple assets
- Correlation-based position management
- Diversification rules

### Multi-Timeframe
- Uses multiple timeframes for confirmation
- Timeframe-specific model weights
- Enhanced signal quality

## Performance Monitoring

### Key Metrics
- Prediction accuracy by regime
- Regime detection confidence
- Trade performance by market condition
- Model health indicators

### Alerts and Monitoring
- Performance degradation detection
- Model staleness warnings
- Regime distribution changes
- Feature drift monitoring

## Testing and Validation

### Test Script (`test_hmm_lstm_strategy.py`)
```bash
# Test single variant
python src/strategy/test_hmm_lstm_strategy.py --data data.csv --variant conservative

# Compare all variants
python src/strategy/test_hmm_lstm_strategy.py --data data.csv --variant compare

# Custom configuration
python src/strategy/test_hmm_lstm_strategy.py --config custom_config.json --data data.csv
```

### Validation Steps
1. **Model Availability**: Check that required models exist
2. **Data Quality**: Validate input data format and completeness
3. **Feature Calculation**: Verify technical indicators are computed correctly
4. **Prediction Sanity**: Check prediction ranges and distributions
5. **Performance Metrics**: Compare against baseline strategies

## Integration with Backtesting Framework

### Data Requirements
- OHLCV data in CSV format
- Timestamp, open, high, low, close, volume columns
- Sufficient historical data for model initialization

### Backtrader Integration
- Compatible with Backtrader ecosystem
- Custom data feeds for additional features
- Analyzer integration for performance metrics

### Performance Analysis
- Sharpe ratio, maximum drawdown, win rate
- Regime-specific performance breakdown
- Comparison with buy-and-hold baseline

## Dependencies

### Core Requirements
- PyTorch (for LSTM models)
- scikit-learn (for preprocessing)
- hmmlearn (for HMM models)
- TA-Lib (for technical indicators)
- Backtrader (for backtesting)
- NumPy, Pandas (for data handling)

### Model Requirements
- Trained HMM models from the pipeline
- Trained LSTM models from the pipeline
- Optimized indicator parameters from Optuna

## Troubleshooting

### Common Issues

1. **Models Not Found**
   - Ensure pipeline has been run successfully
   - Check model directory paths in configuration
   - Verify symbol and timeframe match trained models

2. **Poor Performance**
   - Check if models are recent and relevant
   - Validate data quality and completeness
   - Adjust confidence thresholds
   - Review regime distribution in test data

3. **High Memory Usage**
   - Reduce sequence buffer sizes
   - Limit feature history length
   - Use model.eval() mode for LSTM

4. **Slow Execution**
   - Optimize indicator calculations
   - Reduce model inference frequency
   - Use vectorized operations where possible

### Best Practices

1. **Model Management**
   - Regularly retrain models with fresh data
   - Monitor model performance decay
   - Implement model versioning

2. **Risk Management**
   - Always use stop losses
   - Monitor position sizes relative to account
   - Implement maximum drawdown limits

3. **Strategy Tuning**
   - Backtest on out-of-sample data
   - Use walk-forward analysis
   - Monitor regime distribution changes

4. **Production Deployment**
   - Implement proper logging
   - Monitor prediction distributions
   - Set up performance alerts

## Future Enhancements

### Planned Features
- Ensemble model support
- Real-time model updating
- Advanced portfolio optimization
- Cross-asset regime correlation
- Sentiment integration

### Research Directions
- Transformer-based architectures
- Reinforcement learning integration
- Multi-scale temporal modeling
- Alternative regime identification methods

## Examples

See the `test_hmm_lstm_strategy.py` script for complete usage examples and the configuration files for various parameter settings.
