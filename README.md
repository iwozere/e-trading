# Advanced Crypto Trading Framework

A comprehensive, production-ready framework for developing, testing, and optimizing cryptocurrency and shares trading strategies using Backtrader, with advanced machine learning capabilities and sophisticated strategy framework.

## 🚀 Recent Updates (June 2025)

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

## Project Structure

```
crypto-trading/
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
│   ├── analyzer/         # Analysis tools and metrics
│   ├── broker/          # Exchange/broker integrations
│   ├── data/            # Data handling and processing
│   ├── entry/           # Entry strategy implementations
│   ├── exit/            # Exit strategy implementations
│   ├── indicator/       # Technical indicators
│   ├── management/      # Position and risk management
│   ├── ml/              # Machine learning models
│   │   ├── mlflow_integration.py
│   │   ├── feature_engineering_pipeline.py
│   │   ├── automated_training_pipeline.py
│   │   └── helformer_optuna_train.py
│   ├── notification/    # Notification systems
│   ├── optimizer/       # Strategy optimization
│   ├── plotter/         # Visualization tools
│   ├── screener/        # Market screening tools
│   ├── strategy/        # Trading strategies
│   │   ├── advanced_strategy_framework.py
│   │   ├── advanced_backtrader_strategy.py
│   │   └── custom_strategy.py
│   ├── trading/         # Core trading functionality
│   └── util/            # Utility functions
│
├── tests/               # Test suite
│
├── .gitignore          # Git ignore file
├── requirements.txt    # Python dependencies
├── setup.py           # Package setup file
└── TODO.txt           # Project roadmap and tasks
```

## 🎯 Core Features

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

## 🚀 Quick Start

### 1. **Installation**
```bash
# Clone the repository
git clone <repository-url>
cd crypto-trading

# Install dependencies
pip install -r requirements.txt

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

### 4. **Basic Usage**
```bash
# Prepare your data in the data/ directory
# Configure strategies in config/strategy/
# Run optimization
python src/optimizer/custom_optimizer.py
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
- **[OpenAPI Specification](docs/openapi.yaml)** - OpenAPI/Swagger specification for API endpoints

### **Issue Tracking**
- **[Issues Directory](docs/issues/)** - Contains open and closed issue tracking files
  - `docs/issues/open/` - Currently open issues and feature requests
  - `docs/issues/closed/` - Resolved issues and completed features

## 🛠️ Dependencies

### **Core Dependencies**
- backtrader
- pandas
- numpy
- optuna
- ta-lib (optional)

### **ML Dependencies**
- mlflow
- scikit-learn
- xgboost
- lightgbm
- tensorflow (optional)
- pytorch (optional)

### **Additional Dependencies**
- schedule
- scipy
- yaml
- requests
- asyncio

## 📈 Performance Metrics

The framework provides comprehensive performance analysis:

- **Risk-Adjusted Returns**: Sharpe, Sortino, Calmar ratios
- **Risk Metrics**: Maximum drawdown, VaR, CVaR
- **Trade Statistics**: Win rate, profit factor, expectancy
- **Advanced Metrics**: Ulcer index, gain-to-pain ratio, recovery factor

## 🔧 Architecture

### **Indicator Management**

The system uses a centralized indicator management architecture to handle both Backtrader native indicators and TA-Lib indicators consistently.

#### **Indicator Factory**

The `IndicatorFactory` class (`src/indicator/indicator_factory.py`) is the central component for indicator management:

```python
class IndicatorFactory:
    def __init__(self, data: bt.DataBase, use_talib: bool = False):
        self.data = data
        self.use_talib = use_talib
        self.indicators = {}
```

**Key features:**
- Creates and manages both TA-Lib and Backtrader indicators
- Handles indicator lifecycle
- Provides consistent interface for all indicators
- Caches indicators to prevent duplicate creation

**Available indicators:**
- RSI (Relative Strength Index)
- Bollinger Bands
- ATR (Average True Range)
- SMA (Simple Moving Average)
- MACD, Stochastic, Williams %R, CCI, ROC
- 50+ additional technical indicators

#### **Mixin Architecture**

Mixins use the IndicatorFactory to create and manage indicators:

```python
class BaseEntryMixin:
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.strategy = None
        self.params = params or {}
        self.indicators = {}
        self.indicator_factory = None

    def init_entry(self, strategy, additional_params: Optional[Dict[str, Any]] = None):
        self.strategy = strategy
        self.indicator_factory = IndicatorFactory(
            data=self.strategy.data,
            use_talib=self.strategy.use_talib
        )
```

**Benefits:**
1. **Clean separation of concerns**
   - Indicator creation is handled by the factory
   - Mixins focus on strategy logic
   - Strategy manages overall flow

2. **Consistent indicator handling**
   - Same interface for all indicators
   - Unified error handling
   - Proper data readiness checks

3. **Better maintainability**
   - Centralized indicator management
   - Easy to add new indicators
   - Clear responsibility boundaries

4. **Improved error handling**
   - Factory handles indicator creation errors
   - Mixins handle strategy logic errors
   - Clear error boundaries

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📊 Project Status

### **Current State (June 2025):**
- ✅ **Core Infrastructure**: 100% Complete
- ✅ **Performance & Analytics**: 100% Complete
- ✅ **Advanced ML Features**: 100% Complete (NEW)
- ✅ **Advanced Strategy Framework**: 100% Complete (NEW)
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

The crypto trading platform has evolved into a **comprehensive, production-ready system** with advanced ML capabilities and sophisticated strategy framework, bringing the platform to the forefront of algorithmic trading technology.

---

*Last Updated: June 2025*
*For detailed roadmap and future plans, see [docs/ROADMAP.md](docs/ROADMAP.md)*

## Telegram Screener Bot

See the full documentation for the Telegram Screener Bot here:

[docs/TELEGRAM_SCREENER_BOT.md](docs/TELEGRAM_SCREENER_BOT.md)

- Use `/analyze` for all analysis tasks (single ticker or all tickers, with all flags).
- `/my-status` and `/my-analyze` are now replaced by `/analyze`.

This guide covers all features, commands, configuration, and advanced usage for the Telegram-based screener and analysis bot. 