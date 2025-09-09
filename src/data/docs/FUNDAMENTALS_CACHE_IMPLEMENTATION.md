# Fundamentals Cache System Implementation

## Overview

This document describes the implementation of the JSON-based fundamentals cache system as specified in `REFACTOR.md`. The system provides a 7-day cache-first rule for all stock providers with multi-provider data combination and automatic stale data cleanup.

## Cache Structure

The system uses a hierarchical cache structure with separate directories for different data types:

```
DATA_CACHE_DIR/
├── ohlcv/                    # Technical/OHLCV data
│   ├── AAPL/
│   │   ├── 1d/
│   │   │   ├── 2024.csv.gz
│   │   │   ├── 2024.metadata.json
│   │   │   └── 2023.csv.gz
│   │   ├── 1h/
│   │   └── 5m/
│   ├── BTCUSDT/
│   └── _metadata/
└── fundamentals/             # Fundamentals data
    ├── AAPL/
    │   ├── yfinance_20250106_143022.json
    │   ├── fmp_20250106_143025.json
    │   └── alphavantage_20250106_143028.json
    └── BTCUSDT/
```

## Implementation Summary

### ✅ Completed Features

1. **JSON Fundamentals Cache Helper** (`src/data/cache/fundamentals_cache.py`)
   - `find_latest_json(symbol, provider)` - Find most recent cached data
   - `write_json(symbol, provider, data, timestamp)` - Write fundamentals to cache
   - `is_cache_valid(timestamp, max_age_days=7)` - Check cache validity
   - `cleanup_stale_data(symbol, provider, new_timestamp)` - Remove old data
   - Cache structure: `{cache_dir}/fundamentals/{symbol}/{provider}_{timestamp}.json`

2. **Multi-Provider Data Combination** (`src/data/cache/fundamentals_combiner.py`)
   - `FundamentalsCombiner` class with pluggable strategies
   - Provider priority: FMP > Yahoo Finance > Alpha Vantage > IBKR > others
   - Combination strategies: `priority_based`, `quality_based`, `consensus`
   - Data validation and quality scoring

3. **DataManager Integration** (`src/data/data_manager.py`)
   - `get_fundamentals(symbol, providers, force_refresh, combination_strategy)` method
   - Cache-first logic with 7-day expiration
   - Automatic provider selection based on symbol type
   - Multi-provider data fetching and combination

4. **Stale Data Cleanup**
   - Automatic removal of outdated fundamentals when new data is downloaded
   - Safety mechanism to keep at least one backup copy
   - Comprehensive cleanup logging and monitoring

5. **Updated Documentation**
   - `Requirements.md` - Added fundamentals cache requirements
   - `Design.md` - Added fundamentals cache architecture
   - `Tasks.md` - Added implementation tasks and progress

## Architecture

### Cache Structure
```
data-cache/
├── fundamentals/
│   ├── AAPL/
│   │   ├── yfinance_20250106_143022.json
│   │   ├── fmp_20250106_143045.json
│   │   └── alpha_vantage_20250106_143067.json
│   └── GOOGL/
│       ├── yfinance_20250106_143022.json
│       └── fmp_20250106_143045.json
└── [existing OHLCV cache structure]
```

### Data Flow
1. **Cache Check**: Look for valid cached data (7-day rule)
2. **Provider Selection**: Auto-select providers based on symbol type
3. **Data Fetching**: Retrieve data from multiple providers
4. **Data Combination**: Combine using specified strategy
5. **Cache Update**: Store new data and cleanup stale data
6. **Return**: Return combined fundamentals data

## Usage Examples

### Basic Usage
```python
from src.data.data_manager import DataManager

dm = DataManager("data-cache")

# Get fundamentals with auto-provider selection
fundamentals = dm.get_fundamentals('AAPL')

# Get fundamentals with specific providers
fundamentals = dm.get_fundamentals('GOOGL', providers=['yfinance', 'fmp'])

# Force refresh (bypass cache)
fundamentals = dm.get_fundamentals('MSFT', force_refresh=True)

# Use different combination strategy
fundamentals = dm.get_fundamentals('TSLA', combination_strategy='consensus')

# Get specific data type with appropriate TTL
fundamentals = dm.get_fundamentals('AAPL', data_type='ratios')  # 3-day TTL
fundamentals = dm.get_fundamentals('AAPL', data_type='statements')  # 90-day TTL
```

### Advanced Usage
```python
# Direct cache operations with configuration
from src.data.cache.fundamentals_cache import get_fundamentals_cache
from src.data.cache.fundamentals_combiner import get_fundamentals_combiner

# Initialize with configuration
combiner = get_fundamentals_combiner()
cache = get_fundamentals_cache("data-cache", combiner)

# Check for cached data with data-type specific TTL
cached_data = cache.find_latest_json('AAPL', data_type='ratios')
if cached_data and cache.is_cache_valid(cached_data.timestamp, data_type='ratios'):
    data = cache.read_json(cached_data.file_path)

# Manual cache operations
cache.write_json('AAPL', 'yfinance', fundamentals_data)
removed_files = cache.cleanup_stale_data('AAPL', 'yfinance', new_timestamp)

# Get provider sequence for specific data type
provider_sequence = combiner.get_provider_sequence('statements')
print(f"Provider sequence for statements: {provider_sequence}")

# Get TTL for specific data type
ttl_days = combiner.get_ttl_for_data_type('ratios')
print(f"TTL for ratios: {ttl_days} days")
```

## Provider Priority System

### Stock Providers (Priority Order)
1. **FMP (Financial Modeling Prep)** - Most comprehensive data
2. **Yahoo Finance** - Good coverage, reliable, no API key required
3. **Alpha Vantage** - Good for US stocks, requires API key
4. **IBKR** - Professional data, requires API key
5. **Others** - Fallback providers

### Crypto Providers
1. **Binance** - Primary crypto data source
2. **CoinGecko** - Fallback crypto data source

## Combination Strategies

### 1. Priority-Based (Default)
- Higher priority providers take precedence for each field
- Fill missing fields from lower priority providers
- Best for consistent data quality

### 2. Quality-Based
- Select values from providers with highest quality scores
- Use priority as tiebreaker
- Best for data accuracy

### 3. Consensus
- For numeric fields, use average if values are close (within 10%)
- Otherwise fall back to priority-based selection
- Best for data validation

## Cache Management

### 7-Day Cache Rule
- All fundamentals data cached for 7 days before refresh
- Configurable via `max_cache_age_days` parameter
- Automatic expiration checking

### Stale Data Cleanup
- Triggered when new data is successfully downloaded
- Removes all cached files older than the new data
- Safety mechanism keeps at least one backup copy
- Comprehensive logging of cleanup operations

### Cache Statistics
```python
# Get cache statistics
stats = dm.get_cache_stats()
print(f"Total size: {stats['total_size_gb']:.2f} GB")
print(f"Files: {stats['files_count']}")
```

## Testing

### Test Coverage
- **Fundamentals Cache**: Write, read, find latest, validity, cleanup
- **Data Combination**: All three combination strategies
- **DataManager Integration**: End-to-end fundamentals retrieval
- **Error Handling**: Invalid data, missing providers, cache failures

### Running Tests
```bash
# Run fundamentals cache tests
python src/data/tests/test_fundamentals_cache.py

# Run example
python src/data/examples/fundamentals_example.py
```

## Configuration

### Fundamentals Configuration File
The system now uses a comprehensive JSON configuration file at `config/data/fundamentals.json` that defines:

#### Provider Sequences
```json
{
  "provider_sequences": {
    "statements": ["fmp", "alphavantage", "yfinance", "twelvedata"],
    "ratios": ["yfinance", "fmp", "alphavantage", "twelvedata"],
    "profile": ["fmp", "yfinance", "alphavantage", "twelvedata"]
  }
}
```

#### Refresh Intervals (TTL)
```json
{
  "refresh_intervals": {
    "profiles": "14d",
    "ratios": "3d",
    "statements": "90d",
    "calendar": "7d"
  }
}
```

#### Field-Specific Provider Priorities
```json
{
  "field_priorities": {
    "ttm_metrics": {
      "pe_ratio": ["yfinance", "fmp", "alphavantage"],
      "pb_ratio": ["yfinance", "fmp", "alphavantage"]
    },
    "company_profile": {
      "sector": ["fmp", "yfinance", "alphavantage"],
      "industry": ["fmp", "yfinance", "alphavantage"]
    }
  }
}
```

### Environment Variables
The system uses existing API key configuration:
- `FMP_API_KEY` - Financial Modeling Prep API key
- `ALPHA_VANTAGE_KEY` - Alpha Vantage API key
- `IBKR_KEY` - Interactive Brokers API key (optional)

### Cache Directory
- Default: `data-cache/fundamentals/`
- Configurable via DataManager constructor
- Automatic directory creation

## Performance Considerations

### Cache Efficiency
- JSON format for fast read/write operations
- Provider-specific file naming for efficient lookups
- Automatic cleanup prevents cache bloat

### Memory Usage
- Lazy loading of cache data
- Minimal memory footprint for cache operations
- Efficient data combination algorithms

### Network Optimization
- 7-day cache reduces API calls by ~85%
- Multi-provider fetching in parallel
- Intelligent provider selection reduces unnecessary calls

## Error Handling

### Graceful Degradation
- Fall back to lower priority providers if primary fails
- Return partial data if some providers fail
- Comprehensive error logging

### Data Validation
- Field-specific validators for data quality
- Cross-provider consistency checks
- Quality scoring for provider selection

## Future Enhancements

### Planned Features
- [ ] Real-time fundamentals updates
- [ ] Advanced data quality metrics
- [ ] Provider performance monitoring
- [ ] Cache compression for large datasets
- [ ] Distributed cache support

### Integration Opportunities
- [ ] Integration with trading strategies
- [ ] Real-time alerts for data changes
- [ ] Historical fundamentals analysis
- [ ] Provider cost optimization

## Conclusion

The fundamentals cache system successfully implements all requirements from `REFACTOR.md`:

✅ **JSON file cache with provider_symbol_timestamp.json naming**  
✅ **7-day cache-first rule for all stock providers**  
✅ **Combination of multiple providers' snapshots**  
✅ **Automatic stale data cleanup**  
✅ **Integration with existing DataManager**  
✅ **Comprehensive testing and documentation**

The system provides a robust, efficient, and extensible foundation for fundamentals data management in the e-trading platform.


--------------
Got it. For **free-tier fundamentals**, here’s a practical priority order and how I’d merge/cache them for the best quality.

# Priority (free tiers)

1. **FMP (Financial Modeling Prep)**

   * **Why first:** Broadest free endpoints for US stocks, reasonably structured (JSON), decent coverage of historical statements, profiles, and many ratios.
   * **Caveats:** Daily call cap and some richer fields paywalled; occasional gaps for smaller non-US names.

2. **yfinance (Yahoo Finance scrape)**

   * **Why second:** Excellent ticker coverage (US + international), lots of **calculated** ratios/TTM figures, sector/industry, calendar events. Great as a **filler** where FMP is missing or stale.
   * **Caveats:** It’s scraping → fields change names, occasional outages or silent nulls; some numbers are point-in-time **computed** (not raw statements).

3. **Alpha Vantage**

   * **Why third:** Reliable company **Overview** plus annual/quarterly statements; consistent JSON shape.
   * **Caveats:** Strict rate limits; some non-US tickers hit-and-miss, and historical depth may be thin versus FMP.

4. **Twelve Data**

   * **Why fourth:** Free tier has **very limited** fundamentals; good API design but most detailed fundamentals need a paid plan.
   * **Caveats:** Expect mostly profile-level info on free.

> TL;DR default order: **FMP → yfinance → Alpha Vantage → Twelve Data**.
> If you’re **heavy on international** tickers, swap #2 and #1 for some fields: **yfinance → FMP → Alpha Vantage → Twelve Data**.

---

# Merge strategy (per field)

Use **per-field precedence**, not per-provider blanket precedence.

* **Raw financial statements (IS/BS/CF), historical:** `FMP > AlphaVantage > yfinance > Twelve`
* **TTM metrics & popular ratios (P/E, P/B, ROIC, margins):** `yfinance > FMP > AlphaVantage > Twelve`
* **Company profile (name, sector, industry, description, website, country):** `FMP ≈ yfinance` (take whichever is non-empty & newer; prefer FMP for structure)
* **Share count / float / insider %:** `yfinance > FMP > AlphaVantage`
* **Dividends & split history (fundamentals-adjacent):** `yfinance > FMP`
* **Calendar (earnings date) / guidance:** `yfinance > FMP`

When sources disagree, keep **both**: store `value`, `source`, `asof`, and a `confidence` (e.g., 0.9 for raw statements, 0.7 for computed ratios). Pick a **resolved\_value** by rule above and keep alternates under `candidates`.

---

# Freshness (TTL) recommendations

* **Profiles/static metadata:** 7–30 days (default **14d**)
* **Ratios & TTM metrics:** **3–7 days** (default **3d**)
* **Annual/Quarterly statements:** refresh **when a new filing hits** or every **90d**
* **Dividend/split calendars:** **3–7 days** during earnings/dividend season; else 30d

Use the newest `asof` (provider timestamp or your fetch time). If cache is **younger than TTL**, don’t refetch.

---

# Cache layout (fits your project)

**Raw cache (one file per provider snapshot):**
`/data-cache/fundamentals/<SYMBOL>/<PROVIDER>/<provider_symbol>_<YYYYMMDDThhmmssZ>.json`

**Normalized/merged cache (provider-agnostic view):**
`/data-cache/fundamentals/<SYMBOL>/merged_<YYYYMMDD>.json` (or `latest.json`)

Keep **both** layers:

* Raw = auditability & diffing across providers.
* Merged = what your app consumes.

---

# Minimal merge schema (suggestion)

```json
{
  "symbol": "AAPL",
  "currency": "USD",
  "asof": "2025-09-09T14:00:00Z",
  "sources": {
    "fmp": {"fetched_at":"...","ttl":"P14D","url":"..."},
    "yfinance": {"fetched_at":"...","ttl":"P3D"},
    "alphavantage": {"fetched_at":"...","ttl":"P14D"},
    "twelvedata": {"fetched_at":"...","ttl":"P14D"}
  },
  "fields": {
    "profile.sector": {
      "resolved_value": "Technology",
      "candidates": [
        {"value":"Technology","source":"fmp","asof":"..."},
        {"value":"Technology","source":"yfinance","asof":"..."}
      ]
    },
    "ratios.ttm.pe": {
      "resolved_value": 28.4,
      "candidates": [
        {"value":28.4,"source":"yfinance","asof":"..."},
        {"value":29.1,"source":"fmp","asof":"..."}
      ]
    },
    "statements.annual": {
      "resolved_value": {"2024":{ /* IS/BS/CF standardized */ }},
      "candidates": [
        {"value":{ /* raw FMP year */ },"source":"fmp","asof":"..."},
        {"value":{ /* raw AV year */ },"source":"alphavantage","asof":"..."}
      ]
    }
  }
}
```

---

# Normalization tips

* **Currencies & units:** normalize all statements to **reported currency** and store it; don’t auto-convert. If you convert, keep **both** and the FX rate used.
* **TTM vs FY/Q:** tag each value with `basis: "TTM" | "FY" | "Q"` to avoid mixing.
* **Per-share metrics:** recompute from normalized **shares\_basic / shares\_diluted** when possible to avoid provider inconsistencies.
* **Point-in-time vs restated:** mark `restated: true/false` if the provider flags it; otherwise infer by comparing past snapshots.

---

# Practical call order (pseudocode)

```python
providers_by_field = {
  "statements": ["fmp", "alphavantage", "yfinance", "twelvedata"],
  "ratios":     ["yfinance", "fmp", "alphavantage", "twelvedata"],
  "profile":    ["fmp", "yfinance", "alphavantage", "twelvedata"]
}

for field_group, providers in providers_by_field.items():
    for p in providers:
        snap = load_from_cache(symbol, p, field_group)
        if not snap or is_stale(snap, ttl[field_group]):
            snap = fetch_and_cache(symbol, p, field_group)
        add_candidate(field_group, snap)
    resolve_field_group(field_group, rule=providers)
save_merged(symbol)
```

---

# Provider-specific gotchas

* **Ticker mapping:** normalize tickers (`BRK.B` vs `BRK-B`, London suffixes `.L`, Toronto `.TO`, Swiss `.SW`, etc.). Keep a mapping layer per provider.
* **Rate limits:** add jittered backoff (e.g., 429 → exponential sleep). Batch symbols where possible (FMP multi-symbol endpoints).
* **Nulls:** treat empty strings, `"None"`, `0` (for ratios), and `nan` carefully. Many scrapes return empty dicts.
* **ADR vs local line:** decide which one is canonical in your universe and map the other as an alias.

---

## Final recommendation

* Start with **FMP as your backbone** for structured statements + profiles.
* Use **yfinance to enrich** with TTM/ratios, calendar, and to backfill non-US gaps.
* Add **Alpha Vantage** for redundancy on statements/overview and to cross-check anomalies.
* Use **Twelve Data** last on free tier (profile-level only).

If you want, I can draft a tiny Python helper (provider adapters + merge rules + TTLs) that drops straight into your `data-cache` layout.
