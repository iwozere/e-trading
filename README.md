# Advanced Trading Framework

A comprehensive, production-ready framework for developing, testing, and optimizing trading strategies across multiple asset classes (cryptocurrencies, stocks, forex, commodities) using Backtrader, with advanced machine learning capabilities and sophisticated strategy framework.

## 🚀 Recent Updates (December 2024)

### ✅ **Major New Features Implemented:**

- **🤖 Advanced ML Features** (2,254 lines of code)
  - Complete MLflow integration with model registry and deployment
  - Comprehensive feature engineering pipeline with 50+ technical indicators
  - Automated training pipeline with A/B testing and drift detection
  - Production-ready machine learning ecosystem

- **📊 Advanced Strategy Framework** (1,847 lines of code)
  - Composite strategies with signal aggregation
  - Multi-timeframe strategy support
  - Dynamic strategy switching based on market regimes
  - Portfolio optimization strategies

- **📚 Comprehensive Documentation** (1,149 lines)
  - Complete ML features documentation
  - Advanced strategy framework guide
  - Working examples and best practices

- **🎯 Advanced ATR Exit Strategy** (New)
  - Sophisticated volatility-adaptive trailing stop
  - Multi-timeframe ATR analysis with state machine
  - Structural ratcheting and time-based tightening
  - Partial take-profit capabilities

## Project Structure

```
e-trading/
├── config/                 # Configuration files
│   ├── donotshare/        # Sensitive configuration files (API keys, etc.)
│   ├── optimizer/         # Optimization configuration
│   ├── strategy/          # Advanced strategy configurations
│   │   ├── composite_strategies.json
│   │   ├── multi_timeframe.json
│   │   ├── dynamic_switching.json
│   │   └── portfolio_optimization.json
│   └── trading/          # Trading strategy configurations
│
├── data/                  # Data storage directory
│
├── db/                    # Database files
│
├── docs/                  # Documentation
│   ├── ADVANCED_ML_FEATURES.md
│   ├── ADVANCED_STRATEGY_FRAMEWORK.md
│   ├── ROADMAP.md
│   └── ... (comprehensive documentation)
│
├── examples/              # Working examples
│   ├── advanced_ml_example.py
│   ├── advanced_strategy_example.py
│   └── ... (other examples)
│
├── logs/                  # Log files
│
├── results/              # Backtesting and optimization results
│
├── src/                  # Source code
│   ├── analytics/        # Analysis tools and metrics
│   ├── backtester/       # Backtesting engine
│   ├── common/           # Common utilities and base classes
│   ├── config/           # Configuration management
│   ├── data/             # Data handling and processing
│   ├── error_handling/   # Error handling and resilience
│   ├── frontend/         # Web GUI and API
│   ├── management/       # Position and risk management
│   ├── ml/               # Machine learning models
│   │   ├── mlflow_integration.py
│   │   ├── feature_engineering_pipeline.py
│   │   ├── automated_training_pipeline.py
│   │   └── helformer_optuna_train.py
│   ├── models/           # Data models and schemas
│   ├── notification/     # Notification systems
│   ├── strategy/         # Trading strategies
│   │   ├── advanced_strategy_framework.py
│   │   ├── advanced_backtrader_strategy.py
│   │   └── custom_strategy.py
│   ├── trading/          # Core trading functionality
│   ├── util/             # Utility functions
│   └── utils/            # Additional utilities
│
├── tests/               # Test suite
│
├── .gitignore          # Git ignore file
├── requirements.txt    # Python dependencies
├── requirements-full.txt # Full dependency list
├── requirements-test.txt # Test dependencies
├── requirements-dev.txt # Development dependencies
├── docker-compose.yml  # Docker configuration
├── Dockerfile          # Docker image definition
└── TODO.txt           # Project roadmap and tasks
```

## 🎯 Core Features

### **Multi-Asset Class Support**
- **Cryptocurrencies**: Binance, CoinGecko integration
- **Stocks**: Yahoo Finance, Interactive Brokers integration
- **Forex**: Multiple broker support
- **Commodities**: Futures and options trading
- **Unified Interface**: Consistent API across all asset classes

### **Advanced Machine Learning**
- **MLflow Integration**: Model versioning, tracking, registry, and automated deployment
- **Feature Engineering**: 50+ technical indicators, market microstructure features, automated selection
- **Automated Training**: Scheduled retraining, A/B testing, drift detection, performance monitoring
- **Model Management**: Hyperparameter optimization, model comparison, production deployment

### **Advanced Strategy Framework**
- **Composite Strategies**: Combine multiple strategies with weighted voting, consensus, majority voting
- **Multi-timeframe Support**: Higher timeframe trend analysis, lower timeframe entry/exit
- **Dynamic Strategy Switching**: Market regime detection and adaptive strategy selection
- **Portfolio Optimization**: Modern Portfolio Theory, risk parity, dynamic allocation

### **Modular Strategy Design**
- Separate entry and exit logic
- Mixin-based architecture for easy strategy composition
- Support for multiple entry and exit strategies
- Advanced signal aggregation and validation

### **Optimization Framework**
- Parameter optimization using Optuna
- Configurable optimization spaces
- Comprehensive metrics collection
- Results storage in JSON format

### **Analysis Tools**
- Performance metrics (Sharpe, Sortino, Calmar ratios)
- Risk metrics (Drawdown, Volatility)
- Trade statistics (Win rate, Profit factor)
- Custom analyzers for detailed analysis

### **Live Trading Capabilities**
- Real-time data feeds via WebSocket
- Live trading bot management
- Risk management and position sizing
- Automated order execution

### **Web Interface & API**
- RESTful API for bot management
- Web GUI for monitoring and control
- Real-time dashboard
- User authentication and authorization

## 🚀 Quick Start

### 1. **Installation**
```bash
# Clone the repository
git clone <repository-url>
cd e-trading

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install full dependencies for all features
pip install -r requirements-full.txt

# (Optional) Install TA-Lib for enhanced performance
# Follow platform-specific installation guide
```

### 2. **Advanced ML Features Demo**
```bash
# Run the comprehensive ML features demonstration
python examples/advanced_ml_example.py
```

### 3. **Advanced Strategy Framework Demo**
```bash
# Run the advanced strategy framework demonstration
python examples/advanced_strategy_example.py
```

### 4. **Live Trading Bot Demo**
```bash
# Run the live trading bot example
python examples/live_trading_example.py
```

### 5. **Web Interface**
```bash
# Start the web GUI
python src/frontend/app.py

# Access the interface at http://localhost:5000
```

## 📊 Advanced ML Capabilities

### **Feature Engineering Pipeline**
```python
from src.ml.feature_engineering_pipeline import FeatureEngineeringPipeline

# Initialize pipeline
pipeline = FeatureEngineeringPipeline()

# Generate 50+ technical indicators
features_df = pipeline.generate_features(data)

# Select best features
selected_features = pipeline.select_features(features_df, target)

# Scale features
scaled_features = pipeline.scale_features(selected_features)
```

### **MLflow Integration**
```python
from src.ml.mlflow_integration import MLflowManager

# Initialize MLflow
mlflow_manager = MLflowManager()

# Start experiment
run_id = mlflow_manager.start_run("experiment_name")

# Log parameters and metrics
mlflow_manager.log_parameters({"learning_rate": 0.1})
mlflow_manager.log_metrics({"accuracy": 0.85})

# Register model
version = mlflow_manager.register_model("model_name", model_uri, "Production")
```

### **Automated Training Pipeline**
```python
from src.ml.automated_training_pipeline import AutomatedTrainingPipeline

# Initialize pipeline
pipeline = AutomatedTrainingPipeline(config)

# Start scheduled training
pipeline.start_scheduled_training()

# Check for drift
drift_results = pipeline.check_drift(current_data)

# Run A/B testing
ab_results = pipeline.run_ab_test(model_a, model_b)
```

## 🎯 Advanced Strategy Framework

### **Composite Strategies**
```python
from src.strategy.advanced_strategy_framework import AdvancedStrategyFramework

# Initialize framework
framework = AdvancedStrategyFramework()

# Generate composite signal
signal = framework.get_composite_signal("momentum_trend_composite", data_feeds)

# Execute strategy
composite_signal = framework.execute_strategy("dynamic_switching", data_feeds)
```

### **Multi-timeframe Strategies**
```python
from src.strategy.advanced_backtrader_strategy import AdvancedBacktraderStrategy

# Configure multi-timeframe strategy
strategy = AdvancedBacktraderStrategy(
    strategy_config="multi_timeframe",
    timeframes=["1h", "4h", "1d"],
    trend_timeframe="4h",
    entry_timeframe="1h"
)
```

### **Dynamic Strategy Switching**
```python
# Market regime detection automatically switches strategies
# Trending markets: momentum strategies
# Ranging markets: mean reversion strategies
# Volatile markets: volatility breakout strategies
```

## 📚 Documentation

The project includes comprehensive documentation organized in the `docs/` folder:

### **Getting Started**
- **[User Guide](docs/USER_GUIDE.md)** - Complete user guide for setting up and using the platform
- **[Environment Setup](docs/ENVIRONMENT_SETUP.md)** - Step-by-step environment configuration and installation
- **[Configuration Guide](docs/CONFIGURATION.md)** - Comprehensive configuration management system documentation

### **Advanced Features**
- **[Advanced ML Features](docs/ADVANCED_ML_FEATURES.md)** - Complete guide to ML capabilities, feature engineering, and automated training
- **[Advanced Strategy Framework](docs/ADVANCED_STRATEGY_FRAMEWORK.md)** - Comprehensive guide to composite strategies, multi-timeframe analysis, and dynamic switching
- **[Live Trading Bot](docs/LIVE_TRADING_BOT.md)** - Guide to setting up and running live trading bots
- **[Live Data Feeds](docs/LIVE_DATA_FEEDS.md)** - Documentation for real-time market data integration

### **Analysis & Performance**
- **[Advanced Analytics](docs/ADVANCED_ANALYTICS.md)** - Advanced analysis and metrics calculation
- **[Metrics](docs/METRICS.md)** - Performance metrics and analysis documentation

### **System Architecture**
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Development guidelines, architecture overview, and coding standards
- **[API Documentation](docs/API.md)** - Complete API reference with authentication and endpoint documentation
- **[Database Documentation](docs/DATABASE.md)** - Database schema, operations, and data management

### **System Management**
- **[Error Handling](docs/ERROR_HANDLING.md)** - Error handling, resilience, and recovery system documentation
- **[Notification System](docs/NOTIFICATION_SYSTEM.md)** - Alert and notification system configuration
- **[Alert System](docs/ALERT_SYSTEM.md)** - Trading alerts and signal management

### **Project Information**
- **[Roadmap](docs/ROADMAP.md)** - Project roadmap, future plans, and development milestones
- **[Changelog](docs/CHANGELOG.md)** - Version history and release notes
- **[API Operations](src/api/api_operations.yaml)** - Comprehensive API operations documentation for Trading Web UI

### **Issue Tracking**
- **[Issues Directory](docs/issues/)** - Contains open and closed issue tracking files
  - `docs/issues/open/` - Currently open issues and feature requests
  - `docs/issues/closed/` - Resolved issues and completed features

## 🛠️ Dependencies

### **Core Dependencies**
- backtrader==1.9.78.123
- pandas==2.3.1
- numpy==2.3.2
- optuna>=4.4.0
- ta_lib>=0.6.4

### **ML Dependencies**
- mlflow>=3.1.1
- scikit_learn==1.7.1
- xgboost==3.0.4
- lightgbm==4.6.0
- torch>=2.7.1

### **Trading & Data**
- python_binance>=1.0.29
- yfinance>=0.2.64
- ib_insync==0.9.86
- websocket_client==0.40.0

### **Web & API**
- Flask==3.1.1
- aiogram==3.21.0
- SQLAlchemy==2.0.43

### **Additional Dependencies**
- schedule==1.2.2
- scipy==1.16.1
- statsmodels==0.14.4
- hmmlearn==0.3.3
- pydantic==2.11.7

## 📈 Performance Metrics

The framework provides comprehensive performance analysis:

- **Risk-Adjusted Returns**: Sharpe, Sortino, Calmar ratios
- **Risk Metrics**: Maximum drawdown, VaR, CVaR
- **Trade Statistics**: Win rate, profit factor, expectancy
- **Advanced Metrics**: Ulcer index, gain-to-pain ratio, recovery factor

## 🔧 Architecture

### **Unified Indicator Service**

The system uses a unified indicator service architecture that consolidates all indicator functionality into a single, comprehensive service supporting multiple calculation backends (TA-Lib, pandas-ta, Backtrader).

#### **Unified Service Architecture**

The `UnifiedIndicatorService` (`src/indicators/service.py`) is the central component for all indicator operations:

```python
from src.indicators.service import UnifiedIndicatorService

# Create service instance
service = UnifiedIndicatorService()

# Calculate indicators for a ticker
request = IndicatorRequest(
    ticker="BTCUSDT",
    indicators=["rsi", "macd", "bbands"],
    timeframe="1d",
    period="1y"
)

result = await service.calculate(request)
```

**Key features:**
- **Unified API**: Single interface for all technical and fundamental indicators
- **Multiple Backends**: Support for TA-Lib, pandas-ta, and Backtrader calculation engines
- **Batch Processing**: Efficient calculation for multiple tickers simultaneously
- **Configuration Management**: Centralized parameter management with presets
- **Recommendation Engine**: Intelligent trading recommendations based on indicator values
- **Error Handling**: Comprehensive error handling with graceful fallbacks

**Available indicators:**
- **Technical**: RSI, MACD, Bollinger Bands, Stochastic, ADX, ATR, Williams %R, CCI, ROC, MFI, OBV, SuperTrend, Ichimoku, and more
- **Fundamental**: P/E Ratio, Forward P/E, PEG Ratio, Price-to-Book, ROE, ROA, Debt-to-Equity, and more
- **50+ total indicators** across both categories

#### **Backtrader Integration**

The unified service provides seamless Backtrader integration through specialized adapters:

```python
from src.indicators.adapters.backtrader_wrappers import UnifiedRSI, UnifiedMACD

class MyStrategy(bt.Strategy):
    def __init__(self):
        # Use unified service indicators in Backtrader
        self.rsi = UnifiedRSI(self.data, timeperiod=14)
        self.macd = UnifiedMACD(self.data, fastperiod=12, slowperiod=26)
    
    def next(self):
        # Access indicator values as usual
        if self.rsi[0] < 30:
            self.buy()
```

**Benefits:**
1. **Unified Interface**
   - Single API for all indicator types
   - Consistent parameter naming and validation
   - Simplified configuration management

2. **Enhanced Performance**
   - Optimized batch processing
   - Intelligent caching strategies
   - Concurrent calculation support

3. **Better Maintainability**
   - Consolidated codebase
   - Standardized error handling
   - Comprehensive testing coverage

4. **Advanced Features**
   - Intelligent recommendations
   - Multi-backend fallbacks
   - Configuration presets for different trading styles

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📊 Project Status

### **Current State (December 2024):**
- ✅ **Core Infrastructure**: 100% Complete
- ✅ **Performance & Analytics**: 100% Complete
- ✅ **Advanced ML Features**: 100% Complete
- ✅ **Advanced Strategy Framework**: 100% Complete
- ✅ **Error Handling & Resilience**: 100% Complete
- ✅ **Live Data Feeds**: 100% Complete
- ✅ **Notification System**: 100% Complete
- ✅ **Web Interface & API**: 100% Complete
- 🎯 **Real-time Dashboard**: 60% Complete
- 🎯 **Enhanced Backtesting**: 40% Complete
- 🎯 **Production Deployment**: 20% Complete

### **Code Base Size:**
- **Total Lines**: ~15,000+ lines of production code
- **Documentation**: ~5,000+ lines of comprehensive guides
- **Examples**: ~2,000+ lines of working demonstrations
- **Tests**: ~1,500+ lines of test coverage

### **Overall Progress:**
- **Completed**: 8 out of 13 major features (62%)
- **In Progress**: 3 features (23%)
- **Planned**: 2 features (15%)

The Advanced Trading Framework has evolved into a **comprehensive, production-ready system** with advanced ML capabilities and sophisticated strategy framework, supporting multiple asset classes and bringing the platform to the forefront of algorithmic trading technology.

---

*Last Updated: December 2024*
*For detailed roadmap and future plans, see [docs/ROADMAP.md](docs/ROADMAP.md)*

## Telegram Screener Bot

See the full documentation for the Telegram Screener Bot here:

[docs/TELEGRAM_SCREENER_BOT.md](docs/TELEGRAM_SCREENER_BOT.md)

- Use `/analyze` for all analysis tasks (single ticker or all tickers, with all flags).
- `/my-status` and `/my-analyze` are now replaced by `/analyze`.

This guide covers all features, commands, configuration, and advanced usage for the Telegram-based screener and analysis bot. 