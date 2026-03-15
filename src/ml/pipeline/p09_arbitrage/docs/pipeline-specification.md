# Pipeline Specification: P09 Arbitrage (Statistical Pairs Trading)

## Overview
This document specifies the architecture and implementation details for the `p09_arbitrage` pipeline, designed to execute market-neutral statistical arbitrage (Pairs Trading) strategies. It focuses on identifying historically cointegrated asset pairs (e.g., BTC/ETH, SOL/ADA) and capturing profit when their price spread deviates significantly from the mean, reverting back to equilibrium.

## Architecture & Integration Strategy
`p09_arbitrage` relies on robust asset selection, spread calculation, and mean-reversion signaling. It will integrate with existing data management and execution modules where applicable.
- **Data Acquisition**: Fetch continuous OHLCV data for a defined universe of correlated pairs.
- **Custom Modules (P09 Specific)**: `ArbitragePipeline` (Orchestrator), `CointegrationAnalyzer` (Pair Selection), `SpreadZScoreTracker` (Signal Generation), configuration and CLI runner.

### Output Destination
All execution outputs, logs, and state files will be persisted in `results/p09_arbitrage/YYYY-MM-DD/`.

## Component Design

### 1. `config.py`
Dataclass-driven configuration for pairs trading.
**Key Thresholds**:
- `min_cointegration_p_value`: 0.05
- `zscore_entry_threshold`: 2.5 (Standard deviations)
- `zscore_exit_threshold`: 0.5 (Standard deviations)
- `lookback_window`: 100 (Days/Periods for rolling mean/std)
- `hedge_ratio_recalculation_freq`: 7 (Days)
- `max_holding_period`: 14 (Days - time stop)

### 2. `arbitrage_pipeline.py`
The orchestration engine for the pairs trading flow:
- **Stage A (Universe Selection & Data Fetching)**: Load candidate pairs and fetch historical data.
- **Stage B (Cointegration Testing)**: Filter pairs using Engle-Granger or Johansen tests.
- **Stage C (Spread Tracking)**: Calculate hedge ratios (using OLS regression) and track the rolling mean/z-score of the spread.
- **Stage D (Signal Engine)**: Generate long/short spread signals based on z-score thresholds.

### 3. `cointegration_analyzer.py`
The core engine for finding tradeable pairs.
#### Calculations:
- **Engle-Granger Test**: Tests the residuals of the regression between two assets for stationarity.
- **Hedge Ratio ($\beta$)**: Calculated via Ordinary Least Squares (OLS). Spread = Price_A - ($\beta$ * Price_B).
- **Half-Life of Reversion**: Calculates the Ornstein-Uhlenbeck half-life. Rejects pairs where the half-life is too long (e.g., > 30 days) or too short to trade profitably after fees.

### 4. `spread_tracker.py`
Real-time or daily tracking of selected pairs.
- Maintains rolling windows to dynamically recalculate the Z-Score of the spread.
- Handles corporate actions / structural breaks that might invalidate the historical relationship.

## Execution Outputs
- `pipeline.log`: Full execution trace.
- `01_candidate_pairs.csv`: Raw pairs fed into the system.
- `02_cointegrated_universe.csv`: Pairs passing the statistical tests with stable hedge ratios.
- `03_active_spreads.csv`: Real-time tracking of current Z-Scores for the cointegrated universe.
- `04_arbitrage_signals.csv`: Actionable entry/exit signals for the trading engine.
