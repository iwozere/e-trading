# Backtester Module (`src/backtester`)

This directory contains the core components for backtesting trading strategies, including plotting, optimization, and analysis tools.

## Two Backtest Engines — Division of Labour

The platform contains **two distinct backtest engines**:

| Engine | Location | Purpose |
|--------|----------|---------|
| **Backtrader** (`src/backtester`) | Event-driven, bar-by-bar | Walk-forward validation, final OOS evaluation, realistic simulation (commissions, slippage, partial fills, live-broker replay). Use this for **production strategy validation**. |
| **VectorBT** (`src/vectorbt`) | Vectorized, array-based | Fast parameter sweeps during research and optimization; generates candidate parameter ranges that are then validated by the Backtrader engine. Use this for **exploratory optimization only**. |

### Why two engines?

Vectorbt can evaluate thousands of parameter combinations in seconds but makes simplifying assumptions (no realistic order routing, no bar-by-bar state). Backtrader is slower but faithful to live execution. The intended workflow is:

1. VectorBT sweep → shortlist of promising parameter ranges.
2. Backtrader walk-forward validation → OOS confirmation before live deployment.

Results from the two engines **should not be compared directly** (different fill assumptions). If a strategy shows a large performance gap between engines, investigate the modelling assumptions before deploying.

## Folder Structure & Descriptions

### plotter/
Tools for visualizing backtest results and strategy indicators.
- **run_plotter.py**: Main script for generating plots from backtest results.
- **base_plotter.py**: Base class for plotter implementations.
- **description.txt**: Design notes and requirements for indicator plotting, layout, and trade visualization.
- **indicators/**: Individual indicator plotters:
  - `bollinger_bands_plotter.py`, `ichimoku_plotter.py`, `rsi_plotter.py`, `supertrend_plotter.py`, `volume_plotter.py`, `base_indicator_plotter.py`
  - Each file implements plotting logic for a specific technical indicator.

### optimizer/
Optimization tools for strategy parameters and data conversion.
- **run_optimizer.py**: Main script for running parameter optimization.
- **custom_optimizer.py**: Custom optimization logic and extensions.
- **cnn_lstm_xgboost_tbd.py**: Hybrid deep learning optimizer (CNN/LSTM/XGBoost, to be developed).
- **run_json2csv.py**: Utility for converting JSON backtest results to CSV format.

### analyzer/
Analysis tools for backtest results.
- **bt_analyzers.py**: Custom analyzers for extracting metrics and statistics from backtest runs.
- **__init__.py**: Module initializer.

---

For more details, see the code and docstrings in each file and submodule. 
