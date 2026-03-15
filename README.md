# Advanced Trading Framework

A comprehensive, production-ready framework for developing, testing, and optimizing trading strategies across multiple asset classes (cryptocurrencies, stocks, forex, commodities) using Backtrader, with advanced machine learning capabilities, modern web UI, and sophisticated strategy framework.

## 🚀 Recent Updates (2025-2026)

### ✅ **Major New Features Implemented:**

- **📉 Statistical Arbitrage Pipeline (P09)**
  - Automated pairs discovery and cointegration testing (Engle-Granger).
  - Dynamic beta (hedge ratio) estimation and half-life calculation.
  - Pair-centric results organization with automated multi-timeframe analysis.

- **🌊 Order Flow Analysis Pipeline (P12)**
  - Integrated Binance Futures data (CVD, Open Interest, Funding Rates, Liquidations).
  - Microstructure analysis for identifying institutional positioning.
  - Dedicated storage and caching for high-frequency order flow data.

- **💾 DataManager 2.0 & Caching**
  - Robust OHLCV caching with Parquet support and "update-append" strategy.
  - Automated symbol discovery and corporate action handling.
  - Unified facade for 13+ data providers with intelligent failover.

- **🤖 Expanded ML Ecosystem**
  - 13 production pipelines (P00 to P12) including HMM-LSTM, CNN-XGBoost, and Meta-Regime strategies.
  - Optuna-driven hyperparameter optimization and MLflow model registry.
  - Robustness checks and generalization tests for strategy validation.

- **🎯 Sophisticated Strategy Mixins**
  - **Entry**: EOM (Ease of Movement) Breakouts/Pullbacks, Volume Supertrend, HMM-LSTM.
  - **Exit**: Advanced ATR with volatility adaptation, Multi-level ATR, Time-based tightening.

## Project Structure

```
e-trading/
├── config/                    # Configuration files
│   ├── pipeline/             # ML pipeline configurations (P00-P12)
│   ├── strategy/             # Strategy configurations
│   └── ...
├── data/                      # Market data storage (OHLCV, Order Flow)
├── docs/                      # Comprehensive documentation (HLA, User Guides)
├── results/                   # Backtesting and analysis output
│   ├── p09_arbitrage/        # Pair-centric arbitrage results
│   └── _runs/                # Global execution archives
├── src/                       # Source code (600+ Python files)
│   ├── data/                 # DataManager, Downloaders, Caching
│   ├── ml/                   # Machine learning pipelines
│   │   ├── pipeline/         # 13 ML pipelines (HMM, LSTM, Arbitrage, Order Flow)
│   │   │   ├── p08_mtf/      # Multi-Timeframe Strategy
│   │   │   ├── p09_arbitrage/# Statistical Arbitrage
│   │   │   ├── p10_emps3/    # Advanced Market Performance
│   │   │   ├── p11_meta/     # Meta-Strategy/Regime Filtering
│   │   │   └── p12_order_flow/ # Order Flow Microstructure
│   ├── strategy/             # Trading strategy framework (Entry/Exit Mixins)
│   ├── trading/              # Live trading engine (Binance, IBKR)
│   └── web_ui/               # React + Vite web interface
└── tests/                     # Comprehensive test suite
```

## 🎯 Core Features

### **Advanced Machine Learning (13 Pipelines)**
- **P00-P04**: Core architectures (HMM, LSTM, CNN, XGBoost, Short Squeeze).
- **P05-P07**: Enhanced Market Performance (EMPS) and Combined strategies.
- **P08**: Multi-Timeframe (MTF) reinforcement learning.
- **P09**: Statistical Arbitrage (Pairs Trading) with automated cointegration testing.
- **P10**: EMPS3 - Advanced market performance with fractal analysis.
- **P11**: Meta-Strategy for regime-aware signal filtering.
- **P12**: Order Flow Microstructure (CVD, OI, Funding, Liquidations).

### **DataManager & Persistence**
- **13+ Data Providers**: Binance, Yahoo, FMP, Polygon, Alpaca, Finnhub, etc.
- **Unified Cache**: Provider-agnostic Parquet storage with lightning-fast retrieval.
- **Auto-Discovery**: Automatic scanning and alignment of large historical datasets (2020-2025).

### **Modular Strategy Framework**
- **Modular Design**: Separate Entry and Exit mixins for rapid prototyping.
- **Dynamic Risk**: Volatility-adaptive ATR stops and structural ratcheting.
- **Backtrader Integration**: Direct TA-Lib integration for high-performance indicator logic.

## 🚀 Quick Start

### 1. **Installation**
```bash
# Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-full.txt
```

### 2. **Running P09 Arbitrage (Pairs Trading)**
```bash
# Automated analysis across all data/ timeframes
python src/ml/pipeline/p09_arbitrage/run_arbitrage.py --p-value 0.05
```

### 3. **Running P12 Order Flow Analysis**
```bash
# Analyze microstructure and institutional volume
python src/ml/pipeline/p12_order_flow/run_order_flow.py --symbol BTCUSDT
```

### 4. **Web Interface**
```bash
# Backend
cd src/api && uvicorn main:app --port 8080
# Frontend
cd src/web_ui/frontend && npm run dev
```

## 📚 Documentation
- **[User Guide](docs/USER_GUIDE.md)**: setup and usage.
- **[Advanced ML](docs/ADVANCED_ML_FEATURES.md)**: ML capabilities and pipelines.
- **[Arbitrage Setup](src/ml/pipeline/p09_arbitrage/docs/pipeline-specification.md)**: Pairs trading details.
- **[Order Flow](src/ml/pipeline/p12_order_flow/docs/pipeline-specification.md)**: Data ingestion and analysis.

---
*Last Updated: March 2026*