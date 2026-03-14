# Pipeline Specification: P10 EMPS3 (Accumulation Phase Detection)

## Overview
This document specifies the architecture and implementation details for the `p10_emps3` pipeline, designed to detect stocks in a "coiled spring" state (high institutional absorption with low price volatility) prior to major breakouts. It leverages shared modules from `p06_emps2` while introducing specialized logic tailored to the Precursor Phase.

## Architecture & Integration Strategy
To minimize code duplication, `p10_emps3` utilizes common infrastructure from `p06_emps2`:
- **Shared Modules**: `NasdaqUniverseDownloader`, `FundamentalFilter`, `SentimentFilter`.
- **Custom Modules (P10 Specific)**: `EMPS3Pipeline` (Orchestrator), `AccumulationAnalyzer` (Stage C), `RollingMemoryScanner` (Stage D), configuration and CLI runner.

### Output Destination
All execution outputs, logs, and state files will be persisted in `results/p10_emps3/YYYY-MM-DD/`.

## Component Design

### 1. `config.py`
A dataclass driven configuration extending the base filter configuration. 
**Key Thresholds**:
- `min_vol_zscore`: 1.5
- `max_price_impact`: 0.025 (2.5%)
- `min_vol_rv_ratio`: 2.0
- `lookback_days`: 10
- `require_dark_pool_surge`: true
- `max_distance_from_resistance`: 0.03 (3%)
- Pre-Breakout weights: AR > 2.5 (+30), Squeeze state (+20), Resistance Pressing (+30), Virality (+20).

### 2. `emps3_pipeline.py`
The orchestration engine modifying the P06 stage flow:
- **Stage A (Universe & Fundamentals)**: Reuse P06 Fundamental logic.
- **Stage B (TRF Integration)**: Fetch TRF correction factors via FINRA downloader.
- **Stage C (Pre-Breakout Filter Logic)**: Handled by `AccumulationAnalyzer`.
- **Stage D (Rolling Memory)**: Call custom rolling memory class for Phase 1.5.

### 3. `accumulation_analyzer.py`
The core computational brain replacing P06's `VolatilityFilter`.
#### Calculations:
- **Volume Z-Score ($V_z$)**: Tracks daily volume deviations relative to a 20-day SMA.
- **Realized Volatility ($RV$)**: Annualized standard deviation of log returns using intraday data over the past 5 sessions.
- **Absorption Ratio ($AR$)**: $V_z / RV$. Tickers with $AR > 2.0$ pass the base filter.
- **Price Compression (The Squeeze)**: Checks for Inside Day, relative 12-month BB Width minimum, and V/R divergence.
- **Dark Pool Validation**: Evaluates 3-day TRF shift > 20% along with tight price bounds ($\pm 1.5\%$).
- **Pre-Breakout Filter**: Discards if recent price change > 3.5%, > 10% away from SMA(20), but includes if within 3% of 52w High.
- **Scoring**: Computes `prebreakout_score` (0-100) based on specified weighting logic. Promotes elements scoring $>70$ to the high-priority output list.

### 4. `rolling_memory.py`
Tailored scanner for Phase 1.5 (Early Warning).
- Checks `p10_emps3` historical result directories up to 5 days back.
- Tracks candidates appearing $\ge 3$ times.
- Calculates trend direction for ATR (must be downward) and Volume Z-Score (must be upward).

## Execution Outputs
- `pipeline.log`: Full execution trace.
- `01_nasdaq_universe.csv`, `02_fundamental_raw_data.csv`, `03_fundamental_filtered.csv` (Standard Stage 1/2 outputs).
- `08_absorption_diagnostics.csv`: Comprehensive log containing all evaluated candidates with $Vol/RV$ and TRF divergences.
- `07_prebreakout_watchlist.csv`: High-priority actionable candidates (`prebreakout_score > 70`).
- Alerts triggered for candidates hitting `07_prebreakout_watchlist.csv`.
