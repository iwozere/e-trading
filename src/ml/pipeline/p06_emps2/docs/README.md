# P06 EMPS2 Pipeline - Enhanced Explosive Move Pre-Screener

**Version:** 1.0 (Integration Proposal)
**Status:** Planning Phase
**Last Updated:** 2025-11-27

---

## Table of Contents

- [Overview](#overview)
- [Current State Analysis](#current-state-analysis)
- [Integration Strategy](#integration-strategy)
- [Implementation Plan](#implementation-plan)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Technical Requirements](#technical-requirements)
- [Timeline](#timeline)

---

## Overview

EMPS2 is an enhanced version of the EMPS (Explosive Move Probability Score) pipeline that focuses on **pre-filtering** the trading universe using multi-stage fundamental and technical filters before applying the full EMPS scoring.

### Key Differences from P05 EMPS

| Aspect | P05 EMPS | P06 EMPS2 |
|--------|----------|-----------|
| **Universe Source** | FMP Screener API | NASDAQ Trader raw universe (all US stocks) |
| **Filtering Stages** | 1 stage (FMP screener) | 3 stages (fundamentals → volume → volatility) |
| **Data Providers** | FMP only | Finnhub (fundamentals) + YFinance (OHLCV) |
| **Target Size** | 500-1000 tickers | 50-200 highly filtered tickers |
| **Focus** | Broad screening | Precision filtering for explosive candidates |
| **Indicators** | Basic VWAP/Volume | ATR-based volatility + price range |

### What EMPS2 Adds

1. **Multi-Stage Filtering**
   - Stage 1: Fundamental filter (market cap, float, volume)
   - Stage 2: Price filter (min price threshold)
   - Stage 3: Volatility filter (ATR/Price ratio, price range)

2. **Enhanced Data Sources**
   - NASDAQ Trader for complete US universe (~8000 tickers)
   - Finnhub for fundamental data (market cap, float, sector)
   - YFinance for 15-min intraday data (free, no API limits for OHLCV)

3. **Volatility-First Approach**
   - ATR (Average True Range) / Price > 2% threshold
   - Price range > 5% in lookback period
   - Optimized for stocks showing early volatility expansion

4. **Organized Results Storage**
   - All files stored in `results/emps2/YYYY-MM-DD/`
   - NASDAQ universe, fundamental filter results, volatility filter results
   - No cache in `data/cache/` - everything in dated results folders

---

## Current State Analysis

### Existing Implementation (`emps2.py`)

**Strengths:**
- ✅ Complete end-to-end pipeline in single file
- ✅ Multi-stage filtering logic (fundamentals → activity)
- ✅ ATR-based volatility calculation
- ✅ Uses free data sources (NASDAQ Trader, Finnhub, YFinance)

**Integration Gaps:**
- ❌ Hardcoded API keys and config values
- ❌ Direct `yfinance` and `requests` usage (bypasses project downloaders)
- ❌ No TA-Lib integration for indicator calculation
- ❌ Results saved to current directory (not `results/emps2/yyyy-mm-dd/`)
- ❌ No integration with existing project structure
- ❌ Single monolithic file (not modular pipeline)

### Existing Project Infrastructure

**Available Downloaders:**
- [FinnhubDataDownloader](../../../data/downloader/finnhub_data_downloader.py) - Fundamentals, market data
- [YahooDataDownloader](../../../data/downloader/yahoo_data_downloader.py) - OHLCV, batch operations
- Both support standardized interfaces: `get_ohlcv()`, `get_fundamentals()`, batch methods

**Existing Patterns (from P05):**
- Results storage in `results/emps/yyyy-mm-dd/`
- Modular pipeline structure (loader → adapter → scanner)
- Configuration classes with dataclasses
- Integration with project logging
- All output files organized by date for historical tracking

---

## Integration Strategy

### 1. Use Existing Downloaders

**Replace:**
```python
# Current approach
response = requests.get(f"https://finnhub.io/api/v1/stock/profile2?symbol={t}&token={FINNHUB_KEY}")
data = yf.download(tickers, interval="15m", period="7d")
```

**With:**
```python
# Integrated approach
from src.data.downloader.finnhub_data_downloader import FinnhubDataDownloader
from src.data.downloader.yahoo_data_downloader import YahooDataDownloader

finnhub = FinnhubDataDownloader()
fundamentals = finnhub.get_fundamentals(symbol)  # Returns Fundamentals schema

yahoo = YahooDataDownloader()
ohlcv_data = yahoo.get_ohlcv_batch(symbols, interval="15m", start_date, end_date)
```

**Benefits:**
- Standardized error handling and logging
- Connection pooling and rate limiting
- Consistent data schemas
- Integration with project config

### 2. Use TA-Lib for Indicators

**Replace:**
```python
# Current ATR calculation (manual)
def compute_ATR(df, period=14):
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = (df["High"] - df["Close"].shift()).abs()
    df["L-PC"] = (df["Low"] - df["Close"].shift()).abs()
    tr = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    atr = tr.rolling(period).mean()
    return atr
```

**With:**
```python
# TA-Lib implementation
import talib

def compute_atr_talib(df, period=14):
    """Calculate ATR using TA-Lib for better performance and accuracy."""
    atr = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
    return pd.Series(atr, index=df.index)
```

**Benefits:**
- Industry-standard calculations
- Better performance (C implementation)
- Consistent with other project indicators
- Additional indicators available (RSI, MACD, etc.)

### 3. Standardized Results Storage

**Replace:**
```python
# Current approach
final_df.to_csv("prefiltered_universe.csv", index=False)
```

**With:**
```python
# Integrated approach
from pathlib import Path
from datetime import datetime

def get_results_path(filename: str) -> Path:
    """Get standardized results path under results/emps2/yyyy-mm-dd/."""
    today = datetime.now().strftime('%Y-%m-%d')
    results_dir = Path('results') / 'emps2' / today
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / filename

# Save results
output_path = get_results_path('prefiltered_universe.csv')
final_df.to_csv(output_path, index=False)
logger.info("Results saved to %s", output_path)
```

**Benefits:**
- Consistent with P05 pattern
- Timestamped results for historical tracking
- Organized file structure
- Easy comparison across runs

### 4. Modular Pipeline Structure

**Split monolithic `emps2.py` into:**

```
src/ml/pipeline/p06_emps2/
├── __init__.py
├── config.py                    # Configuration dataclasses
├── universe_downloader.py       # NASDAQ Trader universe fetching
├── fundamental_filter.py        # Stage 1: Fundamental filtering
├── volatility_filter.py         # Stage 2: ATR + price range filtering
├── emps2_pipeline.py           # Main orchestration
├── run_emps2_scan.py           # CLI entry point
├── docs/
│   ├── README.md               # This file
│   ├── Requirements.md         # Dependencies
│   ├── Design.md               # Architecture details
│   └── Tasks.md                # Implementation tracking
└── tests/
    ├── test_universe_downloader.py
    ├── test_fundamental_filter.py
    └── test_volatility_filter.py
```

---

## Implementation Plan

### Phase 1: Project Structure & Configuration

**Tasks:**
1. ✅ Create `src/ml/pipeline/p06_emps2/docs/README.md` (this file)
2. Create `src/ml/pipeline/p06_emps2/docs/Requirements.md`
3. Create `src/ml/pipeline/p06_emps2/docs/Design.md`
4. Create `src/ml/pipeline/p06_emps2/docs/Tasks.md`
5. Create `config.py` with dataclasses

**Example Config:**
```python
from dataclasses import dataclass

@dataclass
class EMPS2FilterConfig:
    """EMPS2 filtering parameters."""
    # Fundamental filters
    min_price: float = 1.0
    min_avg_volume: int = 400_000
    min_market_cap: int = 50_000_000      # $50M
    max_market_cap: int = 5_000_000_000   # $5B
    max_float: int = 60_000_000           # 60M shares

    # Volatility filters
    min_volatility_threshold: float = 0.02  # ATR/Price > 2%
    min_price_range: float = 0.05           # 5% range

    # Data parameters
    lookback_days: int = 7
    interval: str = "15m"
    atr_period: int = 14
```

### Phase 2: Universe Downloader

**Create `universe_downloader.py`:**
- Fetch NASDAQ Trader universe
- Parse and clean ticker list
- Save to `results/emps2/YYYY-MM-DD/nasdaq_universe.csv`
- Remove test issues, non-alphabetic tickers
- Check for today's universe file before re-downloading

### Phase 3: Fundamental Filter

**Create `fundamental_filter.py`:**
- Use `FinnhubDataDownloader` for fundamentals
- Apply market cap, float, volume filters
- Batch processing with rate limiting
- Save to `results/emps2/YYYY-MM-DD/fundamental_filtered.csv`
- Return filtered DataFrame with metadata

### Phase 4: Volatility Filter

**Create `volatility_filter.py`:**
- Use `YahooDataDownloader` for 15m OHLCV data
- Calculate ATR using TA-Lib
- Apply price, ATR/Price, and range filters
- Batch download for efficiency
- Save to `results/emps2/YYYY-MM-DD/volatility_filtered.csv`

### Phase 5: Pipeline Orchestration

**Create `emps2_pipeline.py`:**
- Orchestrate all stages
- Handle errors and logging
- Save results to `results/emps2/yyyy-mm-dd/`
- Generate summary statistics

### Phase 6: CLI Interface

**Create `run_emps2_scan.py`:**
- Command-line interface (similar to P05)
- Argument parsing for all parameters
- Progress reporting

**Example Usage:**
```bash
# Run with defaults
python src/ml/pipeline/p06_emps2/run_emps2_scan.py

# Custom parameters
python src/ml/pipeline/p06_emps2/run_emps2_scan.py \
    --min-cap 100000000 \
    --max-cap 2000000000 \
    --min-volatility 0.025 \
    --output my_universe.csv
```

### Phase 7: Testing & Documentation

**Tasks:**
1. Write unit tests for each component
2. Test end-to-end pipeline
3. Document configuration options
4. Create usage examples

---

## Architecture

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Universe Download                                       │
│ • Source: NASDAQ Trader FTP                                      │
│ • Output: ~8000 tickers (NASDAQ + NYSE + AMEX)                  │
│ • File: results/emps2/YYYY-MM-DD/nasdaq_universe.csv            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Fundamental Filter                                      │
│ • Provider: FinnhubDataDownloader                                │
│ • Filters: Market Cap ($50M-$5B), Float (<60M), Volume (>400K)  │
│ • Output: ~500-1000 tickers                                      │
│ • File: results/emps2/YYYY-MM-DD/fundamental_filtered.csv       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Volatility Filter                                       │
│ • Provider: YahooDataDownloader (15m data, 7 days)              │
│ • Indicators: ATR (TA-Lib), Price Range                         │
│ • Filters: Price (>$1), ATR/Price (>2%), Range (>5%)            │
│ • Output: ~50-200 high-volatility candidates                     │
│ • File: results/emps2/YYYY-MM-DD/volatility_filtered.csv        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: Final Results                                           │
│ • Path: results/emps2/YYYY-MM-DD/                               │
│ • Files: prefiltered_universe.csv, summary.json, run_log.txt    │
│ • All intermediate files preserved for debugging                │
└─────────────────────────────────────────────────────────────────┘
```

### Integration with P05 EMPS

**Standalone Usage:**
```python
from src.ml.pipeline.p06_emps2.emps2_pipeline import EMPS2Pipeline

pipeline = EMPS2Pipeline()
filtered_universe = pipeline.run()
```

**Combined with P05:**
```python
from src.ml.pipeline.p06_emps2.emps2_pipeline import EMPS2Pipeline
from src.ml.pipeline.p05_emps.emps_integration import EMPSUniverseScanner

# EMPS2 for pre-filtering
emps2 = EMPS2Pipeline()
prefiltered = emps2.run()

# P05 EMPS for scoring
scanner = EMPSUniverseScanner(...)
results = scanner.scan_universe(
    tickers=prefiltered['ticker'].tolist(),
    min_emps_score=0.6
)
```

---

## Technical Requirements

### Python Dependencies

**Core:**
- `pandas >= 1.3.0`
- `numpy >= 1.21.0`
- `requests >= 2.26.0`

**Data Providers:**
- `yfinance >= 0.2.0`
- Finnhub API key

**Indicators:**
- `TA-Lib >= 0.4.0`

**Project:**
- `src.data.downloader`
- `src.notification.logger`
- `src.model.schemas`

### External Services

- **NASDAQ Trader FTP** - No auth required
- **Finnhub API** - 60 calls/min (free tier)
- **Yahoo Finance** - No API key required

---

## Next Steps

1. **Review and approve this proposal**
2. **Create additional documentation** (Requirements.md, Design.md, Tasks.md)
3. **Begin Phase 1 implementation**

---

## Related Documentation

- [P05 EMPS Documentation](../../p05_emps/docs/README.md)
- [Project Coding Conventions](../../../../.claude/CLAUDE.md)

---

**Last Updated:** 2025-11-27
**Status:** Awaiting Approval
