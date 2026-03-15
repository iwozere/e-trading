# Pipeline Specification: P11 Meta (Regime Switching Ensemble)

## Overview
This document specifies the architecture and implementation details for the `p11_meta` pipeline. Unlike standard trading pipelines, P11 acts as a "Meta-Pipeline" or portfolio manager. It utilizes statistical models (such as Gaussian Mixture Models or Hidden Markov Models) to detect the current overarching market regime (e.g., High-Volatility Trend, Low-Volatility Mean Reversion, Market Crash) and dynamically reallocates capital across the other active pipelines (e.g., P07, P08, P09, P10).

## Architecture & Integration Strategy
`p11_meta` sits one level above the individual strategy pipelines. It consumes macro indicators and broad market data, determining optimal strategy weights.
- **Shared Data**: Broad market indices (e.g., BTC Market Cap, Total Crypto Market Cap, VIX-equivalents).
- **Custom Modules (P11 Specific)**: `MetaOrchestrator` (Manager), `RegimeDetectorHMM` (State Detection), `CapitalAllocator` (Weight Distribution), configuration and CLI runner.

### Output Destination
All execution outputs, logs, and regime states will be persisted in `results/p11_meta/YYYY-MM-DD/`.

## Component Design

### 1. `config.py`
Configuration defining the macro inputs and strategy mappings.
**Key Parameters**:
- `hmm_components`: 3 (Target number of hidden states/regimes)
- `macro_lookback_days`: 365
- `rebalance_frequency`: `weekly`
- `pipeline_mapping`: Maps specific regimes to pipeline families (e.g., Regime 0 -> P07 Trend Following, Regime 1 -> P09 Arbitrage).
- `max_allocation_per_pipeline`: 0.6 (60%)

### 2. `meta_orchestrator.py`
The overarching loop that manages other pipelines.
- **Stage A (Macro Data Aggregation)**: Fetches broad market data (volume, volatility, Bitcoin dominance, global market cap).
- **Stage B (Regime Classification)**: Feeds data into the HMM to predict the most likely current hidden state.
- **Stage C (Weight Generation)**: Maps the detected state to a vector of capital allocation weights for individual pipelines.
- **Stage D (Signal Dispatch)**: Outputs the new portfolio weights for the execution engine to adjust balances.

### 3. `regime_detector.py`
The core machine learning engine.
#### Calculations:
- **Feature Engineering**: Calculates trailing realized variance, cumulative volume delta (broad market), and aggregate momentum.
- **Hidden Markov Model (HMM)**: Fits an HMM (typically using `hmmlearn` or `pomegranate`) to the feature space.
- **State Inference**: Uses the Viterbi algorithm or forward-backward algorithm to predict the current discrete regime (0, 1, or 2).

### 4. `capital_allocator.py`
Transforms abstract states into actionable portfolio directives.
- Example mapping:
  - *State 0 (Bullish/Trend)*: Overweight P07 (MTF Trend).
  - *State 1 (Chop/Sideways)*: Overweight P09 (Arbitrage) & P10 (Mean Reversion/Accumulation).
  - *State 2 (High Volatility/Crash)*: Reduce gross exposure, move to cash, or activate specific short-only pipelines.

## Execution Outputs
- `meta_pipeline.log`: Full execution trace.
- `01_macro_features.csv`: Processed inputs for the regime model.
- `02_regime_probabilities.csv`: Daily probabilities of being in each hidden state.
- `03_target_allocations.csv`: The final, actionable weight distribution for the trading bots (e.g., P07: 40%, P08: 0%, P09: 60%).
