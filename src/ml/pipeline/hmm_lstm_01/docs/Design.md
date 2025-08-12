Got it — I can draft **Design.md** and **Tasks.md** from your updated `Requirements.md` with Optuna included.

Here's what they could look like:

---

## **Design.md**

```markdown
# Design Document: Automated Trading Pipeline with HMM + LSTM + Optuna

## 1. System Overview
The pipeline integrates **Hidden Markov Models (HMM)** for market regime detection, a **single LSTM model** for next-step forecasting, and **Optuna** for hyperparameter optimization of both **technical indicators** and **LSTM architecture**.

Data flows sequentially through the following components:

1. **Data Loader** → Raw OHLCV historical data acquisition.
2. **Preprocessing** → Normalization, feature engineering, log returns.
3. **HMM Training** → Learn market regimes.
4. **HMM Application** → Label each candle with regime state.
5. **Optuna Indicator Optimization** → Tune technical indicator parameters.
6. **Optuna LSTM Optimization** → Tune model hyperparameters.
7. **LSTM Training** → Train final model with optimized parameters.
8. **Validation & Testing** → Compare against naive baseline, generate reports.

---

## 2. Architecture Diagram

```text
 ┌────────────┐      ┌───────────────┐      ┌─────────────┐
 │ Data Loader│──▶──▶│ Preprocessing │──▶──▶│ Train HMM   │
 └────────────┘      └───────────────┘      └─────┬───────┘
                                                    │
                                                    ▼
                                            ┌─────────────┐
                                            │ Apply HMM   │
                                            └─────┬───────┘
                                                  │
                                                  ▼
     ┌───────────────────────┐     ┌─────────────────────────┐
     │ Optuna Indicators     │     │ Optuna LSTM              │
     │ (Feature parameters)  │     │ (Model parameters)       │
     └──────────┬────────────┘     └───────────────┬──────────┘
                │                                  │
                ▼                                  ▼
         ┌───────────────┐                 ┌───────────────┐
         │ Train LSTM    │                 │ Validate/Test │
         └───────────────┘                 └───────────────┘
```

---

## 3. Data Flow

1. **Input**

   * OHLCV from exchange APIs (Binance, etc.) for multiple timeframes.
   * Stored in `data/`.

2. **Preprocessing**

   * Normalization (MinMax or StandardScaler).
   * Rolling statistics.
   * Technical indicators (parameters later tuned via Optuna).
   * Saved in `data/processed/`.

3. **HMM**

   * Trained with selected features.
   * Produces `regime` labels.
   * Output stored in `data/labeled/`.

4. **Optimization**

   * Indicators: Sharpe ratio, profit factor, and drawdown as objective metrics.
   * LSTM: MSE, directional accuracy, and Sharpe ratio as objective metrics.

5. **Model Training**

   * Inputs: `close`, `log_return`, `regime` (categorical), optimized indicators.
   * Target: Next-step `close` or `log_return`.

6. **Evaluation**

   * Metrics: MSE, Directional Accuracy, Sharpe Ratio, Max Drawdown, Profit Factor.
   * Output: Reports, charts, and JSON parameter files.

---

## 4. Key Components

* **HMM**

  * `n_components = 3`
  * Uses rolling training window.

* **Indicators** (to optimize with Optuna)

  * RSI period
  * Bollinger Bands period & std dev
  * SMA/EMA lengths

* **LSTM Hyperparameters** (to optimize with Optuna)

  * Sequence length
  * Hidden size
  * Batch size
  * Learning rate
  * Dropout
  * Number of layers

---

## 5. Persistence & Outputs

* Models stored in:

  * `src/ml/hmm/model/`
  * `src/ml/lstm/model/`
* Parameters stored in:

  * `results/indicators_<symbol>_<tf>_<timestamp>.json`
  * `results/lstm_params_<symbol>_<tf>_<timestamp>.json`
* Reports stored in `results/` and `reports/`.

---

## 6. Retraining Policy

| Component            | Retraining Frequency | Data Window          |
| -------------------- | -------------------- | -------------------- |
| HMM                  | Monthly / Quarterly  | 1–3 years            |
| Indicator Parameters | Weekly / Bi-weekly   | 1–6 months           |
| LSTM                 | Weekly / Bi-weekly   | 1–3 months (rolling) |

---

## 7. Error Handling Architecture

### 7.1 Fail-Fast Design Pattern
The pipeline implements a **fail-fast** approach with configurable error handling that distinguishes between critical and optional stages:

```python
class PipelineStage:
    def __init__(self, stage_id, is_critical=True):
        self.stage_id = stage_id
        self.is_critical = is_critical
        self.fail_fast = True  # Default behavior
    
    def execute(self):
        try:
            result = self._run_stage()
            return result
        except Exception as e:
            if self.is_critical and self.fail_fast:
                _logger.error("CRITICAL STAGE FAILED: Stage %d (%s)", 
                             self.stage_id, self.__class__.__name__)
                _logger.error("Pipeline stopped due to critical stage failure")
                sys.exit(1)
            else:
                _logger.warning("OPTIONAL STAGE FAILED: Stage %d (%s)", 
                               self.stage_id, self.__class__.__name__)
                return None
```

### 7.2 Stage Criticality Classification

#### Critical Stages (Fail-Fast Enabled)
- **Data Loading** (Stage 1): Essential for all downstream processing
- **Data Preprocessing** (Stage 2): Required for model training
- **HMM Training** (Stage 3): Core regime detection functionality
- **HMM Application** (Stage 4): Required for LSTM feature engineering
- **LSTM Training** (Stage 7): Main prediction model

#### Optional Stages (Can Fail Without Stopping Pipeline)
- **Indicator Optimization** (Stage 5): Enhancement, can use defaults
- **LSTM Optimization** (Stage 6): Enhancement, can use defaults
- **Model Validation** (Stage 8): Reporting, not essential for operation

### 7.3 Error Recovery Design
```python
class PipelineRunner:
    def __init__(self, config):
        self.config = config
        self.fail_fast = config.get('fail_fast', True)
        self.continue_on_optional_failures = config.get('continue_on_optional_failures', False)
    
    def run_from_stage(self, start_stage):
        """Resume pipeline from specific stage after failure."""
        stages_to_skip = list(range(1, start_stage))
        return self.run(skip_stages=stages_to_skip)
    
    def run(self, skip_stages=None):
        """Execute pipeline with optional stage skipping."""
        for stage in self.stages:
            if stage.stage_id in skip_stages:
                continue
            
            try:
                stage.execute()
            except Exception as e:
                if stage.is_critical:
                    self._handle_critical_failure(stage, e)
                else:
                    self._handle_optional_failure(stage, e)
```

---

## 8. Multi-Provider Data Architecture

### 8.1 Provider Abstraction Layer
```python
class DataProviderFactory:
    """Factory for creating data providers based on configuration."""
    
    @staticmethod
    def create_provider(provider_code, config):
        providers = {
            'binance': BinanceDataDownloader,
            'yfinance': YahooDataDownloader,
            'alphavantage': AlphaVantageDataDownloader,
            'finnhub': FinnhubDataDownloader,
            'polygon': PolygonDataDownloader,
            'twelvedata': TwelveDataDownloader,
            'coingecko': CoinGeckoDataDownloader
        }
        
        provider_class = providers.get(provider_code)
        if not provider_class:
            raise ValueError(f"Unsupported provider: {provider_code}")
        
        return provider_class(config)
```

### 8.2 Multi-Provider Configuration Design
```yaml
# Multi-provider data configuration
data_sources:
  binance:
    symbols: [LTCUSDT, BTCUSDT]
    timeframes: [5m, 4h]
    api_key: null  # Optional
    api_secret: null  # Optional
  yfinance:
    symbols: [AAPL, MSFT]
    timeframes: [4h, 1d]
    # No API key required
  alphavantage:
    symbols: [EURUSD, GBPUSD]
    timeframes: [1h, 1d]
    api_key: "${ALPHA_VANTAGE_KEY}"  # Environment variable
```

### 8.3 File Naming Strategy
```python
class FileNamingStrategy:
    """Handles provider-specific file naming to avoid conflicts."""
    
    @staticmethod
    def generate_filename(provider, symbol, timeframe, start_date, end_date):
        """Generate provider-prefixed filename."""
        date_range = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        return f"{provider}_{symbol}_{timeframe}_{date_range}.csv"
    
    @staticmethod
    def parse_filename(filename):
        """Parse provider-prefixed filename."""
        parts = filename.replace('.csv', '').split('_')
        if len(parts) >= 4:
            provider = parts[0]
            symbol = parts[1]
            timeframe = parts[2]
            date_range = '_'.join(parts[3:])
            return provider, symbol, timeframe, date_range
        return None
```

---

## 9. Rate Limiting Architecture

### 9.1 Rate Limiter Base Class
```python
class RateLimiter:
    """Base class for implementing rate limiting across providers."""
    
    def __init__(self, min_request_interval):
        self.min_request_interval = min_request_interval
        self.last_request_time = 0
        self._lock = threading.Lock()
    
    def wait_if_needed(self):
        """Ensure minimum time between requests."""
        with self._lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
```

### 9.2 Provider-Specific Rate Limiters
```python
class BinanceRateLimiter(RateLimiter):
    """Binance-specific rate limiter: 1200 requests/minute."""
    
    def __init__(self):
        super().__init__(min_request_interval=0.05)  # 1/1200 per second

class YahooFinanceRateLimiter(RateLimiter):
    """Yahoo Finance rate limiter: 1 request/second."""
    
    def __init__(self):
        super().__init__(min_request_interval=1.0)

class AlphaVantageRateLimiter(RateLimiter):
    """Alpha Vantage rate limiter: 5 requests/minute (free tier)."""
    
    def __init__(self):
        super().__init__(min_request_interval=12.0)  # 60/5 seconds
```

### 9.3 Batching Architecture
```python
class DataBatchingStrategy:
    """Handles automatic batching for providers with bar limits."""
    
    @staticmethod
    def calculate_batches(start_date, end_date, interval, max_bars=1000):
        """Calculate optimal batch dates to respect bar limits."""
        interval_minutes = {
            '1m': 1, '3m': 3, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
            '1d': 1440, '3d': 4320, '1w': 10080, '1M': 43200
        }
        
        minutes_per_interval = interval_minutes.get(interval, 1440)
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

## 10. HMM Regime Analysis Design

### 10.1 Regime Detection Architecture
```python
class HMMRegimeDetector:
    """Enhanced HMM regime detection with improved labeling logic."""
    
    def __init__(self, n_components=3, covariance_type='full'):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.model = None
        self.regime_labels = {}
    
    def train(self, features):
        """Train HMM model on features."""
        self.model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=200,
            random_state=42
        )
        self.model.fit(features)
        self._analyze_regimes(features)
    
    def _analyze_regimes(self, features):
        """Analyze detected regimes and assign meaningful labels."""
        regime_stats = self._calculate_regime_statistics(features)
        self.regime_labels = self._assign_regime_labels(regime_stats)
    
    def _assign_regime_labels(self, regime_stats):
        """Assign labels using improved dynamic threshold logic."""
        labels = {}
        
        if self.n_components == 3:
            returns = [stat['avg_return'] for stat in regime_stats]
            min_return = min(returns)
            max_return = max(returns)
            return_range = max_return - min_return
            
            if return_range > 0.001:  # Good separation
                for i, stat in enumerate(regime_stats):
                    avg_return = stat['avg_return']
                    relative_position = (avg_return - min_return) / return_range
                    
                    if i == 0:  # Lowest return regime
                        if relative_position < 0.3:
                            labels[stat['regime_id']] = 'Bearish'
                        else:
                            labels[stat['regime_id']] = 'Sideways'
                    elif i == 1:  # Middle return regime
                        if relative_position > 0.7:
                            labels[stat['regime_id']] = 'Bullish'
                        else:
                            labels[stat['regime_id']] = 'Sideways'
                    else:  # Highest return regime
                        labels[stat['regime_id']] = 'Bullish'
            else:
                # Poor separation - use absolute thresholds
                for i, stat in enumerate(regime_stats):
                    avg_return = stat['avg_return']
                    
                    if i == 0:
                        labels[stat['regime_id']] = 'Bearish' if avg_return < 0 else 'Sideways'
                    elif i == 1:
                        labels[stat['regime_id']] = 'Bullish' if avg_return > 0.0005 else 'Sideways'
                    else:
                        labels[stat['regime_id']] = 'Bullish'
        
        return labels
```

### 10.2 Debugging and Monitoring Design
```python
class RegimeAnalysisDebugger:
    """Provides comprehensive debugging information for regime analysis."""
    
    def __init__(self, timeframe):
        self.timeframe = timeframe
        self.debug_info = {}
    
    def analyze_data_characteristics(self, data):
        """Analyze overall data characteristics."""
        returns = data['log_return'].dropna()
        
        self.debug_info['overall_stats'] = {
            'mean_return': returns.mean(),
            'std_return': returns.std(),
            'negative_pct': (returns < 0).mean() * 100,
            'timeframe': self.timeframe
        }
    
    def analyze_regime_quality(self, regime_stats):
        """Analyze quality of detected regimes."""
        self.debug_info['regime_analysis'] = {
            'regime_counts': {stat['regime_id']: stat['sample_count'] for stat in regime_stats},
            'regime_returns': {stat['regime_id']: stat['avg_return'] for stat in regime_stats},
            'regime_volatility': {stat['regime_id']: stat['volatility'] for stat in regime_stats},
            'bearish_candidates': [stat['regime_id'] for stat in regime_stats if stat['avg_return'] < 0]
        }
    
    def generate_debug_report(self):
        """Generate comprehensive debug report."""
        report = f"=== Regime Analysis Debug for {self.timeframe} ===\n"
        
        if 'overall_stats' in self.debug_info:
            stats = self.debug_info['overall_stats']
            report += f"Overall data: mean_return={stats['mean_return']:.6f}, "
            report += f"std_return={stats['std_return']:.6f}, "
            report += f"negative_pct={stats['negative_pct']:.1f}%\n"
        
        if 'regime_analysis' in self.debug_info:
            analysis = self.debug_info['regime_analysis']
            report += f"Regime counts: {analysis['regime_counts']}\n"
            
            for regime_id, avg_return in analysis['regime_returns'].items():
                sample_count = analysis['regime_counts'][regime_id]
                volatility = analysis['regime_volatility'][regime_id]
                bearish_candidate = regime_id in analysis['bearish_candidates']
                
                report += f"Regime {regime_id}: samples={sample_count}, "
                report += f"avg_return={avg_return:.6f}, "
                report += f"volatility={volatility:.6f}, "
                report += f"bearish_candidate={bearish_candidate}\n"
        
        return report
```

---

## 11. Configuration Management Design

### 11.1 Pipeline Configuration Schema
```yaml
# config/pipeline/x01.yaml
pipeline:
  fail_fast: true
  continue_on_optional_failures: false
  max_retries: 3
  log_level: INFO

data_sources:
  binance:
    symbols: [BTCUSDT, ETHUSDT]
    timeframes: [5m, 15m, 1h, 4h]
    rate_limit: 0.05  # seconds between requests
  yfinance:
    symbols: [AAPL, MSFT]
    timeframes: [1h, 4h, 1d]
    rate_limit: 1.0

hmm:
  n_components: 3
  covariance_type: full
  n_iter: 200
  train_window_days: 730

lstm:
  sequence_length: 60
  hidden_size: 64
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  dropout: 0.2
  num_layers: 2

optuna:
  n_trials: 50
  sampler: tpe
  pruning: true
  timeout: 3600  # 1 hour

evaluation:
  test_split: 0.1
  baseline_model: naive
  metrics: [mse, directional_accuracy, sharpe_ratio]
```

### 11.2 Environment Variable Support
```python
class ConfigLoader:
    """Loads configuration with environment variable substitution."""
    
    @staticmethod
    def load_config(config_path):
        """Load YAML config with environment variable substitution."""
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Substitute environment variables
        config_content = os.path.expandvars(config_content)
        
        return yaml.safe_load(config_content)
    
    @staticmethod
    def validate_config(config):
        """Validate configuration structure and required fields."""
        required_sections = ['pipeline', 'data_sources', 'hmm', 'lstm', 'optuna']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data sources
        if not config.get('data_sources'):
            raise ValueError("At least one data source must be configured")
        
        return True
```

---

## 12. Logging and Monitoring Design

### 12.1 Structured Logging
```python
import logging
import json
from datetime import datetime

class PipelineLogger:
    """Structured logging for pipeline operations."""
    
    def __init__(self, log_level=logging.INFO):
        self.logger = logging.getLogger('pipeline')
        self.logger.setLevel(log_level)
        
        # Add structured formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler('pipeline.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def log_stage_start(self, stage_id, stage_name):
        """Log stage start with structured data."""
        self.logger.info("Stage %d started: %s", stage_id, stage_name)
    
    def log_stage_complete(self, stage_id, stage_name, duration, metrics=None):
        """Log stage completion with metrics."""
        log_data = {
            'stage_id': stage_id,
            'stage_name': stage_name,
            'duration': duration,
            'status': 'completed'
        }
        
        if metrics:
            log_data['metrics'] = metrics
        
        self.logger.info("Stage %d completed: %s (%.2fs)", 
                        stage_id, stage_name, duration)
    
    def log_stage_failure(self, stage_id, stage_name, error, is_critical=False):
        """Log stage failure with error details."""
        level = logging.ERROR if is_critical else logging.WARNING
        
        self.logger.log(level, "Stage %d failed: %s - %s", 
                       stage_id, stage_name, str(error))
```

### 12.2 Performance Monitoring
```python
class PerformanceMonitor:
    """Monitors pipeline performance and resource usage."""
    
    def __init__(self):
        self.stage_timings = {}
        self.memory_usage = {}
        self.api_calls = {}
    
    def start_stage(self, stage_id):
        """Start timing a stage."""
        self.stage_timings[stage_id] = {
            'start_time': time.time(),
            'memory_start': psutil.Process().memory_info().rss
        }
    
    def end_stage(self, stage_id):
        """End timing a stage and record metrics."""
        if stage_id in self.stage_timings:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            timing = self.stage_timings[stage_id]
            duration = end_time - timing['start_time']
            memory_used = end_memory - timing['memory_start']
            
            self.stage_timings[stage_id].update({
                'duration': duration,
                'memory_used': memory_used,
                'end_time': end_time
            })
    
    def generate_performance_report(self):
        """Generate performance summary report."""
        total_duration = sum(t['duration'] for t in self.stage_timings.values())
        total_memory = sum(t['memory_used'] for t in self.stage_timings.values())
        
        report = {
            'total_duration': total_duration,
            'total_memory_used': total_memory,
            'stage_breakdown': self.stage_timings,
            'api_calls': self.api_calls
        }
        
        return report
```

---

## 13. Testing and Validation Design

### 13.1 Unit Testing Strategy
```python
class PipelineTestSuite:
    """Comprehensive test suite for pipeline components."""
    
    def test_data_loading(self):
        """Test data loading with mock providers."""
        # Test rate limiting
        # Test batching logic
        # Test error handling
        pass
    
    def test_hmm_regime_detection(self):
        """Test HMM regime detection with synthetic data."""
        # Test regime labeling logic
        # Test debugging output
        # Test timeframe-specific behavior
        pass
    
    def test_error_handling(self):
        """Test error handling and recovery mechanisms."""
        # Test fail-fast behavior
        # Test optional stage failures
        # Test recovery from specific stages
        pass
    
    def test_multi_provider_integration(self):
        """Test multi-provider data integration."""
        # Test provider factory
        # Test file naming conflicts
        # Test configuration validation
        pass
```

### 13.2 Integration Testing
```python
class IntegrationTestRunner:
    """Runs integration tests with real data providers."""
    
    def __init__(self, test_config):
        self.test_config = test_config
        self.results = {}
    
    def run_full_pipeline_test(self):
        """Run complete pipeline with minimal test data."""
        # Use single symbol, short timeframe for testing
        test_config = {
            'data_sources': {
                'binance': {
                    'symbols': ['BTCUSDT'],
                    'timeframes': ['1h']
                }
            },
            'hmm': {'n_components': 2},  # Simpler for testing
            'lstm': {'epochs': 5},  # Fewer epochs for testing
            'optuna': {'n_trials': 3}  # Fewer trials for testing
        }
        
        # Run pipeline and validate outputs
        pass
    
    def test_error_recovery(self):
        """Test pipeline recovery from various failure scenarios."""
        # Test data loading failures
        # Test HMM training failures
        # Test LSTM training failures
        pass
```

---

## 14. Deployment and Operations Design

### 14.1 Containerization Strategy
```dockerfile
# Dockerfile for pipeline deployment
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create data directories
RUN mkdir -p data/raw data/processed data/labeled results reports logs

# Set environment variables
ENV PYTHONPATH=/app/src
ENV LOG_LEVEL=INFO

# Default command
CMD ["python", "src/ml/pipeline/hmm_lstm_01/run_pipeline.py"]
```

### 14.2 Configuration Management
```yaml
# docker-compose.yml for local development
version: '3.8'

services:
  pipeline:
    build: .
    volumes:
      - ./data:/app/data
      - ./results:/app/results
      - ./reports:/app/reports
      - ./logs:/app/logs
    environment:
      - BINANCE_API_KEY=${BINANCE_API_KEY}
      - ALPHA_VANTAGE_KEY=${ALPHA_VANTAGE_KEY}
      - LOG_LEVEL=DEBUG
    command: ["python", "src/ml/pipeline/hmm_lstm_01/run_pipeline.py"]
```

---

## 15. Future Enhancements

### 15.1 Adaptive Rate Limiting
- Dynamic adjustment based on API responses
- Exponential backoff for rate limit violations
- Provider-specific rate limiting strategies

### 15.2 Advanced Regime Detection
- Volatility-based regime detection for longer timeframes
- Multi-timeframe regime alignment
- Regime quality metrics and validation

### 15.3 Performance Optimization
- Parallel processing where possible
- Caching strategies for repeated operations
- Memory optimization for large datasets

### 15.4 Monitoring and Alerting
- Real-time pipeline monitoring
- Performance metrics dashboard
- Automated alerting for failures

```

