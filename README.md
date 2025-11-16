# Advanced Trading Framework

A comprehensive, production-ready framework for developing, testing, and optimizing trading strategies across multiple asset classes (cryptocurrencies, stocks, forex, commodities) using Backtrader, with advanced machine learning capabilities, modern web UI, and sophisticated strategy framework.

## 🚀 Recent Updates (November 2024)

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
├── config/                    # Configuration files
│   ├── backtester/           # Backtester configurations
│   ├── data/                 # Data source configurations
│   ├── donotshare/           # Sensitive configuration files (API keys, etc.)
│   ├── optimizer/            # Optimization configurations (entry/exit mixins)
│   ├── pipeline/             # ML pipeline configurations
│   ├── risk/                 # Risk management configurations
│   ├── schemas/              # JSON schemas for validation
│   │   ├── bot_config.yaml
│   │   └── screener_sets.yml
│   ├── screener/             # Screener configurations
│   ├── strategy/             # Strategy configurations
│   └── trading/              # Live/paper trading configurations
│
├── data/                      # Market data storage
│
├── docs/                      # Comprehensive documentation
│   ├── HLA/                  # High-level architecture documentation
│   ├── issues/               # Issue tracking
│   └── ... (comprehensive documentation)
│
├── src/                       # Source code (593 Python files)
│   ├── analytics/            # Performance analytics and metrics
│   ├── api/                  # FastAPI REST API backend
│   ├── backtester/           # Backtrader-based backtesting engine
│   │   ├── analyzer/         # Custom analyzers
│   │   ├── optimizer/        # Strategy optimization (Optuna)
│   │   ├── plotter/          # Results visualization
│   │   └── validator/        # Configuration validation
│   ├── common/               # Shared utilities
│   │   ├── alerts/           # Alert system
│   │   ├── config/           # Configuration loaders
│   │   └── health/           # Health monitoring
│   ├── data/                 # Data management
│   │   ├── db/               # PostgreSQL/SQLite database layer
│   │   ├── downloader/       # 13+ data provider integrations
│   │   └── feed/             # Live data feeds
│   ├── error_handling/       # Circuit breakers, retry logic
│   ├── ml/                   # Machine learning pipelines
│   │   ├── pipeline/         # 5 ML pipelines (HMM, LSTM, CNN, XGBoost)
│   │   │   ├── p00_hmm_3lstm/
│   │   │   ├── p01_hmm_lstm/
│   │   │   ├── p02_cnn_lstm_xgboost/
│   │   │   ├── p03_cnn_xgboost/
│   │   │   └── p04_short_squeeze/
│   │   └── ... (model training and evaluation)
│   ├── notification/         # Multi-channel notifications
│   ├── scheduler/            # Job scheduling system
│   ├── strategy/             # Trading strategy framework
│   │   ├── entry/            # Entry logic mixins
│   │   ├── exit/             # Exit logic mixins
│   │   ├── base_strategy.py  # Base strategy class
│   │   └── custom_strategy.py # Modular strategy implementation
│   ├── telegram/             # Telegram bot integration
│   ├── trading/              # Live trading engine
│   │   └── broker/           # Broker integrations (Binance, IBKR)
│   └── web_ui/               # React + Vite web interface
│       └── frontend/         # Modern React frontend
│
├── tests/                     # Comprehensive test suite
│
├── .gitignore
├── requirements.txt           # Python dependencies
├── docker-compose.yml         # PostgreSQL + Redis + WebUI
├── Dockerfile
└── alembic.ini               # Database migrations
```

## 🎯 Core Features

### **Modern Web Architecture**
- **Frontend**: React 18 + TypeScript + Vite
  - Material-UI components with modern design
  - Real-time updates via WebSocket
  - State management with Zustand
  - Form validation with React Hook Form + Zod
- **Backend**: FastAPI with async support
  - REST API with OpenAPI documentation
  - JWT authentication and role-based access
  - WebSocket for real-time data
- **Database**: PostgreSQL with SQLAlchemy ORM
  - Alembic migrations for schema management
  - SQLite fallback for development
  - Connection pooling and health checks

### **Multi-Asset Class Support**
- **Cryptocurrencies**: Binance (spot & futures), CoinGecko
- **Stocks**: Yahoo Finance, Interactive Brokers, Alpaca, Polygon
- **Market Data**: 13+ data provider integrations
  - Alpha Vantage, TwelveData, Finnhub, FMP, Tiingo, FINRA
- **Live Trading**: Binance & IBKR with paper trading support
- **Unified Interface**: Factory pattern for consistent data access

### **Advanced Machine Learning**
- **5 Production ML Pipelines**:
  - HMM + 3 LSTM Ensemble (p00)
  - HMM + LSTM Regime Detection (p01)
  - CNN + LSTM + XGBoost Hybrid (p02)
  - CNN + XGBoost Simplified (p03)
  - Short Squeeze Detection Pipeline (p04)
- **MLflow Integration**: Model versioning, tracking, registry, deployment
- **Hyperparameter Optimization**: Optuna integration for automated tuning
- **Libraries**: PyTorch, XGBoost, LightGBM, scikit-learn, hmmlearn

### **Modular Strategy Framework**
- **CustomStrategy**: Production-ready modular strategy system
  - Separate entry and exit logic mixins
  - Support for multiple entry/exit combinations
  - TALib-based indicator architecture
- **Entry Mixins**: RSI+BB, RSI+Ichimoku, Volume-weighted, etc.
- **Exit Mixins**:
  - Fixed ratio take-profit/stop-loss
  - Trailing stop with ATR
  - Advanced ATR with volatility adaptation
  - Multi-timeframe exit logic
- **Legacy Support**: Backward compatible with old configurations
- **Future/Experimental**:
  - Composite strategies (multiple strategy aggregation)
  - Dynamic strategy switching (regime-based)
  - Multi-timeframe strategies

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
# Start the FastAPI backend
cd src/api
uvicorn main:app --reload --port 8080

# In another terminal, start the React frontend
cd src/web_ui/frontend
npm install
npm run dev

# Access the interface at http://localhost:5173
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

### **Backend (Python)**

#### **Core Trading & Backtesting**
- backtrader==1.9.78.123 - Backtesting engine
- pandas==2.3.1 - Data manipulation
- numpy==2.3.2 - Numerical computing
- ta_lib>=0.6.4 - Technical indicators (150+ functions)

#### **Machine Learning**
- torch>=2.7.1 - Deep learning framework
- xgboost==3.0.4 - Gradient boosting
- lightgbm==4.6.0 - Fast gradient boosting
- scikit_learn==1.7.1 - ML algorithms
- hmmlearn==0.3.3 - Hidden Markov Models
- statsmodels==0.14.4 - Statistical models
- optuna>=4.4.0 - Hyperparameter optimization
- mlflow>=3.1.1 - ML experiment tracking

#### **Web Framework & API**
- fastapi==0.115.6 - Modern async web framework
- uvicorn==0.34.0 - ASGI server
- pydantic==2.11.7 - Data validation
- SQLAlchemy==2.0.43 - ORM for PostgreSQL/SQLite
- alembic==1.14.0 - Database migrations
- python-jose==3.3.0 - JWT authentication

#### **Data Providers & Brokers**
- python_binance>=1.0.29 - Binance API
- yfinance>=0.2.64 - Yahoo Finance
- ib_insync==0.9.86 - Interactive Brokers
- alpaca-py>=0.35.3 - Alpaca trading
- websocket_client==0.40.0 - WebSocket support

#### **Telegram Bot**
- aiogram==3.21.0 - Telegram bot framework
- dramatiq==1.18.0 - Task queue for async jobs

#### **Utilities**
- schedule==1.2.2 - Job scheduling
- scipy==1.16.1 - Scientific computing
- psycopg2-binary==2.9.10 - PostgreSQL driver
- redis==5.2.1 - Redis client

### **Frontend (Node.js/React)**

#### **Core Framework**
- react==18.2.0 - UI framework
- typescript==5.2.2 - Type safety
- vite==5.0.0 - Build tool

#### **UI Components & Styling**
- @mui/material==5.14.18 - Material-UI components
- @emotion/react==11.11.1 - CSS-in-JS
- recharts==2.8.0 - Charts and visualizations

#### **State & Data**
- zustand==4.4.7 - State management
- @tanstack/react-query==5.8.4 - Server state management
- axios==1.6.2 - HTTP client
- socket.io-client==4.7.4 - Real-time communication

#### **Forms & Validation**
- react-hook-form==7.48.2 - Form handling
- zod==3.22.4 - Schema validation

## 📈 Performance Metrics

The framework provides comprehensive performance analysis:

- **Risk-Adjusted Returns**: Sharpe, Sortino, Calmar ratios
- **Risk Metrics**: Maximum drawdown, VaR, CVaR
- **Trade Statistics**: Win rate, profit factor, expectancy
- **Advanced Metrics**: Ulcer index, gain-to-pain ratio, recovery factor

## 🔧 Architecture

### **Indicator System**

The framework uses **TA-Lib** directly for technical indicator calculations within Backtrader strategies. The previous UnifiedIndicatorService has been deprecated in favor of a simpler, more performant approach.

#### **TALib Integration**

Strategies use TA-Lib functions directly via Backtrader's built-in wrapper:

```python
import backtrader as bt

class MyStrategy(bt.Strategy):
    def __init__(self):
        # Use TALib indicators directly
        self.rsi = bt.talib.RSI(self.data.close, timeperiod=14)
        self.macd = bt.talib.MACD(
            self.data.close,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        self.bbands = bt.talib.BBANDS(
            self.data.close,
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2
        )

    def next(self):
        # Access indicator values
        if self.rsi[0] < 30:
            self.buy()
```

**Benefits:**
- **Performance**: Direct TA-Lib C library calls for maximum speed
- **Simplicity**: No abstraction layers, straightforward implementation
- **Reliability**: Well-tested industry-standard library
- **Flexibility**: Full access to 150+ TA-Lib functions

**Available Indicators**: 150+ technical indicators including:
- Trend: SMA, EMA, DEMA, TEMA, WMA, TRIMA, KAMA, MAMA, T3
- Momentum: RSI, MACD, Stochastic, Williams %R, ROC, MOM, CMO, ADX
- Volatility: ATR, NATR, TRANGE, Bollinger Bands
- Volume: OBV, AD, ADOSC
- Pattern Recognition: 60+ candlestick patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📊 Project Status

### **Current State (November 2024):**
- ✅ **Core Infrastructure**: 100% Complete
  - Backtrader integration with 593 Python files
  - PostgreSQL database with Alembic migrations
  - Configuration management system
- ✅ **Web Interface**: 100% Complete
  - React 18 + TypeScript + Vite frontend
  - FastAPI backend with async support
  - Material-UI components
  - Real-time WebSocket updates
- ✅ **ML Pipelines**: 100% Complete
  - 5 production ML pipelines (HMM, LSTM, CNN, XGBoost)
  - MLflow integration
  - Optuna hyperparameter optimization
- ✅ **Strategy Framework**: 100% Complete
  - CustomStrategy with modular entry/exit mixins
  - Multiple entry logic (RSI+BB, RSI+Ichimoku, Volume-weighted)
  - Multiple exit logic (Fixed ratio, Trailing stop, ATR-based)
  - TALib indicator integration
- ✅ **Data Providers**: 100% Complete
  - 13+ data provider integrations
  - Binance & IBKR live trading support
  - Factory pattern for unified access
- ✅ **Error Handling & Resilience**: 100% Complete
  - Circuit breakers and retry logic
  - Health monitoring system
  - Comprehensive error handling
- ✅ **Notification System**: 100% Complete
  - Telegram bot integration
  - Multi-channel alerts
  - Job scheduling system
- ✅ **Performance Analytics**: 100% Complete
  - Comprehensive metrics (Sharpe, Sortino, Calmar)
  - Risk analysis tools
  - Trade statistics
- 🎯 **Advanced Features** (Experimental):
  - Composite strategies (in future/)
  - Dynamic strategy switching (in future/)
  - Multi-timeframe strategies (in future/)

### **Code Base Size:**
- **Python Files**: 593 files
- **Source Code**: ~50,000+ lines of production Python code
- **Frontend Code**: React + TypeScript with Material-UI
- **Documentation**: Comprehensive guides in docs/
- **Tests**: Comprehensive test coverage

### **Technology Stack:**
- **Backend**: Python 3.11+, FastAPI, PostgreSQL, SQLAlchemy, Redis
- **Frontend**: React 18, TypeScript, Vite, Material-UI, Zustand
- **ML**: PyTorch, XGBoost, LightGBM, scikit-learn, Optuna, MLflow
- **Trading**: Backtrader, TA-Lib, Binance, IBKR
- **Infrastructure**: Docker, Alembic, Dramatiq

The Advanced Trading Framework is a **mature, production-ready system** with 593 Python files implementing comprehensive trading functionality, modern web interface, advanced ML capabilities, and support for multiple asset classes.

---

*Last Updated: November 2024*
*For detailed documentation, see the [docs/](docs/) directory*

## Telegram Screener Bot

See the full documentation for the Telegram Screener Bot here:

[docs/TELEGRAM_SCREENER_BOT.md](docs/TELEGRAM_SCREENER_BOT.md)

- Use `/analyze` for all analysis tasks (single ticker or all tickers, with all flags).
- `/my-status` and `/my-analyze` are now replaced by `/analyze`.

This guide covers all features, commands, configuration, and advanced usage for the Telegram-based screener and analysis bot. 