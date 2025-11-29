# Design

## Purpose

EMPS2 (Enhanced Explosive Move Pre-Screener) is designed to identify stocks with the highest probability of explosive price movements using multi-stage filtering.

Unlike P05 EMPS which scores a pre-defined universe, EMPS2 starts from the complete US stock universe (~8000 tickers) and applies progressively stricter filters to narrow down to 50-200 highest-probability candidates.

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                       EMPS2 Pipeline                             │
│                                                                   │
│  Stage 1: Universe      →  Stage 2: Fundamental  →  Stage 3:    │
│  Download                  Filter                    Volatility  │
│  (NASDAQ Trader)          (Finnhub)                 (Yahoo+TA)   │
│  ~8000 tickers            ~500-1000 tickers         ~50-200      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Results Storage                               │
│                results/emps2/YYYY-MM-DD/                        │
└─────────────────────────────────────────────────────────────────┘
```

### Component Design

#### 1. Configuration Module (`config.py`)

**Responsibility:** Define all filtering thresholds and pipeline parameters.

**Design Decisions:**
- Use dataclasses for type safety and clarity
- Provide three preset configs: default, aggressive, conservative
- Separate filter config from universe config for modularity

**Classes:**
- `EMPS2FilterConfig` - Filter thresholds (fundamental + volatility)
- `EMPS2UniverseConfig` - Universe download settings
- `EMPS2PipelineConfig` - Complete pipeline configuration

#### 2. Universe Downloader (`universe_downloader.py`)

**Responsibility:** Download complete US stock universe from NASDAQ Trader.

**Design Decisions:**
- Use NASDAQ Trader FTP (free, no API key, complete data)
- Cache results in `results/emps2/YYYY-MM-DD/` for date-based organization
- 24-hour cache TTL (universe doesn't change frequently)
- Filter test issues and non-alphabetic tickers

**Data Flow:**
```
NASDAQ Trader FTP
    ↓
Download nasdaqlisted.txt + otherlisted.txt
    ↓
Combine and filter (remove test issues, special chars)
    ↓
Save to 01_nasdaq_universe.csv + cache
```

#### 3. Fundamental Filter (`03_fundamental_filter.py`)

**Responsibility:** Apply fundamental screening using Finnhub data.

**Design Decisions:**
- Use Finnhub for fundamental data (market cap, float, volume, sector)
- Rate limiting: 1.1s between calls (Finnhub free tier: 60 calls/min)
- Batch processing with progress logging
- Save intermediate results for debugging

**Filtering Criteria:**
- Market cap: $50M - $5B (adjustable)
- Average volume: > 400K shares/day
- Float: < 60M shares (if available)
- Missing data handling: Drop critical fields, keep optional fields

**Data Flow:**
```
Ticker List
    ↓
For each ticker:
    - Get fundamentals (market cap, float, sector)
    - Get quote (avg volume, current price)
    - Rate limit: wait 1.1s
    ↓
Apply filters (market cap, volume, float)
    ↓
Save to fundamental_filtered.csv
```

#### 4. Volatility Filter (`volatility_filter.py`)

**Responsibility:** Apply volatility-based filtering using intraday data.

**Design Decisions:**
- Use Yahoo Finance for intraday data (free, no API key, 60 days history)
- Calculate ATR using TA-Lib (industry standard, C implementation)
- Batch download for efficiency
- 7-day lookback with 15-minute bars (balance between data and performance)

**Filtering Criteria:**
- Price: > $1.00
- ATR/Price ratio: > 2% (adjustable)
- Price range: > 5% over lookback period

**Data Flow:**
```
Ticker List
    ↓
Batch download 15m OHLCV data (7 days) from Yahoo
    ↓
For each ticker:
    - Calculate ATR using TA-Lib
    - Compute ATR/Price ratio
    - Calculate price range (high - low) / low
    - Check minimum bars (>= 20)
    ↓
Apply filters
    ↓
Save to volatility_filtered.csv
```

#### 5. Pipeline Orchestrator (`emps2_pipeline.py`)

**Responsibility:** Coordinate all stages and manage results.

**Design Decisions:**
- Sequential execution (Stage 1 → Stage 2 → Stage 3)
- Save intermediate results at each stage
- Generate summary JSON with statistics
- Comprehensive logging for monitoring
- Graceful error handling with empty DataFrame returns

**Orchestration Flow:**
```
Initialize components (downloaders, filters)
    ↓
Stage 1: Download universe → ~8000 tickers
    ↓
Stage 2: Fundamental filter → ~500-1000 tickers
    ↓
Stage 3: Volatility filter → ~50-200 tickers
    ↓
Create final results (merge fundamental + volatility data)
    ↓
Generate summary (counts, percentages, timing)
    ↓
Save all files to results/emps2/YYYY-MM-DD/
```

#### 6. CLI Interface (`run_emps2_scan.py`)

**Responsibility:** Provide command-line access to pipeline.

**Design Decisions:**
- Argparse for robust CLI parsing
- Support preset configs (--aggressive, --conservative)
- Allow all parameters to be overridden via CLI
- Pretty-printed output for user-friendliness
- Exit codes for scriptability (0 = success, 1 = failure)

## Data Storage Design

### Results Directory Structure

```
results/emps2/
├── 2025-11-27/
│   ├── 01_nasdaq_universe.csv            # Full universe from NASDAQ
│   ├── nasdaq_universe_cache.json     # Cache metadata
│   ├── fundamental_filtered.csv       # After stage 2
│   ├── volatility_filtered.csv        # After stage 3
│   ├── prefiltered_universe.csv       # Final results
│   └── summary.json                   # Pipeline summary
├── 2025-11-28/
│   └── ...
```

### File Formats

**01_nasdaq_universe.csv:**
- Columns: Symbol, Security Name, Market Category, Test Issue, etc.
- Source: NASDAQ Trader FTP (raw format)

**fundamental_filtered.csv:**
- Columns: ticker, market_cap, float, sector, avg_volume, current_price
- Includes all fundamental data from Finnhub

**volatility_filtered.csv:**
- Columns: ticker, last_price, atr, atr_ratio, price_range, price_high, price_low, bars_count
- Sorted by ATR ratio (highest volatility first)

**prefiltered_universe.csv:**
- Merged fundamental + volatility data
- Final output for consumption by trading strategies
- Includes scan_date and scan_timestamp metadata

**summary.json:**
- Pipeline statistics (counts at each stage, timing)
- Configuration used
- Percentages and removal counts

## Integration Patterns

### Standalone Usage

```python
from src.ml.pipeline.p06_emps2 import EMPS2Pipeline

pipeline = EMPS2Pipeline()
results = pipeline.run()

# Use results
for ticker in results['ticker']:
    print(f"Trading {ticker}")
```

### Integration with P05 EMPS

```python
from src.ml.pipeline.p06_emps2 import EMPS2Pipeline
from src.ml.pipeline.p05_emps.emps_integration import EMPSUniverseScanner

# Stage 1: EMPS2 pre-screening
emps2 = EMPS2Pipeline()
prefiltered = emps2.run()

# Stage 2: EMPS scoring on pre-filtered universe
scanner = EMPSUniverseScanner(...)
results = scanner.scan_universe(
    tickers=prefiltered['ticker'].tolist(),
    min_emps_score=0.6
)
```

### Custom Configuration

```python
from src.ml.pipeline.p06_emps2 import EMPS2PipelineConfig, EMPS2FilterConfig

# Create custom filter config
filter_config = EMPS2FilterConfig(
    min_market_cap=100_000_000,  # $100M
    max_market_cap=2_000_000_000,  # $2B
    min_volatility_threshold=0.025,  # 2.5%
    min_price_range=0.07  # 7%
)

# Create pipeline config
pipeline_config = EMPS2PipelineConfig(
    filter_config=filter_config,
    universe_config=EMPS2UniverseConfig(),
    save_intermediate_results=True,
    generate_summary=True
)

# Run pipeline
pipeline = EMPS2Pipeline(pipeline_config)
results = pipeline.run()
```

## Error Handling Strategy

### Graceful Degradation
- Empty DataFrames returned on failure (not exceptions)
- Continue processing even if individual tickers fail
- Log errors but don't stop pipeline

### Retry Logic
- Fundamental filter: max_retries=3 for API calls
- Exponential backoff for rate limit errors
- Skip ticker on repeated failures

### Validation
- Check for required columns in DataFrames
- Validate ticker format (alphabetic, length <= 5)
- Verify minimum data requirements (>= 20 bars for ATR)

## Performance Considerations

### Bottlenecks
1. **Finnhub Rate Limits** - 60 calls/min = ~4.5 hours for 8000 tickers
2. **Yahoo Finance Downloads** - Batch operation helps, but still slow for 500+ tickers
3. **TA-Lib Calculations** - Fast (C implementation), negligible overhead

### Optimizations
1. **Caching:** 24h TTL for universe reduces redundant downloads
2. **Batch Operations:** Yahoo Finance batch downloads (all tickers in one call)
3. **Early Filtering:** Fundamental filter reduces volatility filter workload
4. **Progress Logging:** Every 100 tickers for monitoring

### Future Optimizations
1. **Parallel Processing:** Multi-threading for fundamental filter
2. **Premium API:** Faster Finnhub tier (300 calls/min)
3. **Database Storage:** Faster lookups for repeated scans
4. **Incremental Updates:** Only update changed tickers

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock external API calls
- Validate filter logic with sample data

### Integration Tests
- Test full pipeline with small universe (10 tickers)
- Verify file outputs
- Check summary statistics

### Manual Testing
- Run with aggressive config (expect ~30-50 results)
- Run with conservative config (expect ~100-200 results)
- Verify results make sense (known volatile stocks included)

## Security Considerations

### API Key Management
- Finnhub API key in `config/donotshare/donotshare.py`
- Never log API keys
- Git ignore config/donotshare/

### Data Privacy
- Public market data only
- No PII
- Results stored locally

## Design Trade-Offs

### NASDAQ Trader vs FMP Screener
- **Chosen:** NASDAQ Trader
- **Reason:** Free, complete, no API limits
- **Trade-off:** Less metadata (no sector initially)

### Finnhub vs FMP for Fundamentals
- **Chosen:** Finnhub
- **Reason:** Already integrated, good free tier
- **Trade-off:** Slower rate limits (60 vs 250 calls/min)

### Yahoo Finance vs FMP for OHLCV
- **Chosen:** Yahoo Finance
- **Reason:** Free, no API key, supports batch downloads
- **Trade-off:** Only 60 days of intraday data

### Sequential vs Parallel Processing
- **Chosen:** Sequential
- **Reason:** Simpler, respects rate limits, easier to debug
- **Trade-off:** Slower execution (~4-5 hours)
- **Future:** Parallel processing with rate limit pool

---

**Last Updated:** 2025-11-27
