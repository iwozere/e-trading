# HMM-LSTM Trading Pipeline

This pipeline implements a Hidden Markov Model (HMM) for market regime detection combined with LSTM for time series forecasting. It processes multiple symbols and timeframes to identify market states and generate trading signals.

## Overview

The pipeline consists of three main stages:
1. **Data Loading**: Download and prepare OHLCV data for multiple symbols and timeframes
2. **Feature Engineering**: Calculate technical indicators and prepare features for HMM
3. **HMM Training**: Train Hidden Markov Models to detect market regimes

## Multi-Provider Features

The pipeline now supports multiple data providers with enhanced configuration:

### Supported Providers
- **Binance**: Cryptocurrency data (BTCUSDT, LTCUSDT, etc.)
- **Yahoo Finance**: Stock and ETF data (AAPL, MSFT, VT, etc.)

### Configuration Format
```yaml
data_sources:
  binance:
    symbols: [BTCUSDT, LTCUSDT]
    timeframes: [5m, 15m, 1h, 4h]
  yfinance:
    symbols: [VT, PSNY]
    timeframes: [1h, 4h, 1d]
```

### Rate Limiting Compliance
The pipeline now respects API rate limits for all data providers:

- **Binance**: Maximum 1000 bars per request with automatic batching
- **Yahoo Finance**: 1 request per second with built-in delays
- **Batch Processing**: Rate limiting between symbol downloads

### File Naming Convention
- **Single Provider**: `{symbol}_{timeframe}_{start_date}_{end_date}.csv`
- **Multi-Provider**: `{provider}_{symbol}_{timeframe}_{start_date}_{end_date}.csv`

## Stage 1: Data Loading

Downloads historical OHLCV data for all configured symbols and timeframes.

### Features
- Multi-provider support (Binance, Yahoo Finance)
- Parallel downloading with rate limiting
- Consistent file naming convention
- Progress tracking and error handling
- Automatic data validation

### Rate Limiting Implementation
- **Binance**: Automatic batching to respect 1000 bar limit per request
- **Yahoo Finance**: 1-second delays between requests
- **Batch Processing**: Configurable delays between symbol downloads

## Stage 2: Feature Engineering

Calculates technical indicators and prepares features for HMM training.

### Technical Indicators
- Moving averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)

### Feature Preparation
- Log returns calculation
- Feature normalization
- Missing data handling
- Data quality validation

## Stage 3: HMM Training

Trains Hidden Markov Models to detect market regimes for each symbol-timeframe combination.

### Features
- Multi-provider data processing
- Parallel HMM training
- Consistent color mapping for regime visualization
- Comprehensive regime analysis
- Detailed logging and progress tracking

### Consistent Color Mapping for Regime Visualizations
The pipeline ensures consistent color assignments across all visualizations:
- **Red**: Bearish regimes (lowest average returns)
- **Green**: Bullish regimes (highest average returns)  
- **Blue**: Sideways regimes (intermediate average returns)

This color mapping is determined by analyzing regime characteristics and ensures semantic consistency regardless of the order in which regimes are detected.

### Benefits
- **Semantic Consistency**: Colors always represent the same regime type
- **Visual Clarity**: Easy identification of market conditions
- **Cross-Timeframe Comparison**: Consistent interpretation across different timeframes
- **Reliable Analysis**: Eliminates confusion from inconsistent color assignments

## Configuration

### Main Configuration File
`config/pipeline/p01.yaml`

### Key Parameters
- `data_sources`: Multi-provider configuration
- `period`: Data download period (e.g., "2y")
- `paths`: Directory paths for data storage
- `hmm_params`: HMM training parameters

## Usage

### Basic Usage
```bash
python src/ml/pipeline/p01_hmm_lstm/run_pipeline.py
```

### Configuration
1. Edit `config/pipeline/p01.yaml`
2. Set your data sources and symbols
3. Configure HMM parameters
4. Run the pipeline

## Output

### Data Files
- Raw OHLCV data: `{provider}_{symbol}_{timeframe}_{dates}.csv`
- Processed features: `{symbol}_{timeframe}_features.csv`
- HMM models: `{symbol}_{timeframe}_hmm.pkl`

### Visualizations
- Regime plots: `{symbol}_{timeframe}_regimes.png`
- Feature plots: `{symbol}_{timeframe}_features.png`
- Training plots: `{symbol}_{timeframe}_training.png`

## Technical Implementation

### Data Loading Architecture
The pipeline uses a factory pattern for data downloaders:
```python
from src.data.data_downloader_factory import DataDownloaderFactory

# Create downloader for specific provider
downloader = DataDownloaderFactory.create_downloader("bnc")  # Binance
downloader = DataDownloaderFactory.create_downloader("yf")   # Yahoo Finance
```

### Rate Limiting Implementation
Each data downloader implements rate limiting:

**BinanceDataDownloader**:
- Batching logic to respect 1000 bar limit
- Rate limiting: 1200 requests/minute (0.05s between requests)
- Automatic date range splitting for large requests

**YahooDataDownloader**:
- Rate limiting: 1 request per second
- Built-in delays between symbol downloads
- Error handling for rate limit violations

### Consistent Color Mapping for Regime Visualizations

The color mapping is implemented in the `visualize_regimes` method:

```python
def visualize_regimes(self, df, regimes, regime_labels, save_path):
    # Create color mapping based on regime labels
    color_mapping = {
        'Bearish': 'red',
        'Sideways': 'blue', 
        'Bullish': 'green'
    }
    
    # Apply colors based on semantic labels, not regime IDs
    colors = [color_mapping[label] for label in regime_labels]
```

The regime labeling logic in `analyze_regime_characteristics` ensures proper classification:

```python
def analyze_regime_characteristics(self, df, regimes, timeframe):
    # Calculate regime statistics
    regime_stats = self._calculate_regime_statistics(df, regimes)
    
    # Use dynamic thresholding for robust labeling
    return self._label_regimes_with_thresholds(regime_stats)
```

This approach ensures that:
- Bearish regimes are always red, regardless of regime ID
- Bullish regimes are always green
- Sideways regimes are always blue
- Colors remain consistent across different timeframes and symbols

## Migration Guide

### From Legacy Configuration
If you have the old single-provider configuration:

**Old Format**:
```yaml
data:
  provider: binance
  symbols: [BTCUSDT, LTCUSDT]
  timeframes: [5m, 1h, 4h]
```

**New Format**:
```yaml
data_sources:
  binance:
    symbols: [BTCUSDT, LTCUSDT]
    timeframes: [5m, 1h, 4h]
```

### Benefits of Multi-Provider Support
- **Flexibility**: Use different providers for different asset types
- **Reliability**: Fallback options if one provider fails
- **Optimization**: Choose best provider for each data type
- **Scalability**: Easy to add new providers

## Best Practices

### Rate Limiting
- Always respect provider rate limits
- Use built-in rate limiting features
- Monitor API usage to avoid violations
- Implement exponential backoff for retries

### Data Quality
- Validate downloaded data
- Check for missing values
- Verify date ranges
- Monitor data freshness

### Performance
- Use parallel processing where possible
- Implement proper error handling
- Cache frequently used data
- Monitor memory usage

## Troubleshooting

### Common Issues

**Rate Limit Errors**:
- Check if rate limiting is properly configured
- Verify provider-specific limits
- Increase delays between requests

**Missing Data**:
- Verify symbol availability on provider
- Check date range validity
- Review provider-specific limitations

**Color Mapping Issues**:
- Ensure regime labeling logic is working
- Check for edge cases in regime detection
- Verify color mapping implementation

### Debug Information
The pipeline provides detailed logging:
- Download progress and status
- HMM training metrics
- Regime analysis details
- Error messages and stack traces

## Related Documentation

- [Multi-Provider Guide](MULTI_PROVIDER_GUIDE.md): Detailed guide for multi-provider configuration
- [Regime Analysis Debug](REGIME_ANALYSIS_DEBUG.md): Troubleshooting guide for regime detection issues
