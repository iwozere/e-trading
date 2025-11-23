# EMPS Backtesting Module

Comprehensive backtesting tools for validating EMPS detection accuracy on historical data.

---

## Quick Start

### 1. Prepare Historical Data

Create a CSV file with historical intraday bars:

**Format:**
```csv
timestamp,open,high,low,close,volume
2021-01-25 09:30:00,36.50,37.20,36.00,37.00,1500000
2021-01-25 09:35:00,37.00,38.50,36.80,38.20,2100000
2021-01-25 09:40:00,38.20,39.00,37.50,38.80,1800000
...
```

**Requirements:**
- Columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- Timestamp format: `YYYY-MM-DD HH:MM:SS`
- Data should be sorted by timestamp
- Use 5m or 15m intervals (recommended)

---

### 2. Run Backtest Simulation

Simulate scheduled EMPS scans at predefined times (e.g., market open/close):

```bash
# Simulate market open scan (09:30 ET = 14:30 UTC)
python emps_backtest_simulator.py \
    --data-file data/GME_5m_Jan2021.csv \
    --ticker GME \
    --scan-time "14:30"

# Simulate market close scan (16:00 ET = 21:00 UTC)
python emps_backtest_simulator.py \
    --data-file data/AMC_15m_Jun2021.csv \
    --ticker AMC \
    --scan-time "21:00"

# Simulate both scans (market open + market close)
python emps_backtest_simulator.py \
    --data-file data/BBBY_5m_Aug2022.csv \
    --ticker BBBY \
    --scan-time "14:30,21:00"

# With output file
python emps_backtest_simulator.py \
    --data-file data/GME_5m_Jan2021.csv \
    --ticker GME \
    --scan-time "14:30" \
    --output results/GME_backtest_results.csv
```

**Time Conversion (ET to UTC):**
- Market Open: 09:30 ET → 14:30 UTC (EST) or 13:30 UTC (EDT)
- Market Close: 16:00 ET → 21:00 UTC (EST) or 20:00 UTC (EDT)
- Pre-Market: 08:00 ET → 13:00 UTC (EST) or 12:00 UTC (EDT)

---

### 3. Review Results

The simulator logs detailed output for each scan:

```
==================================================================
Day 1/5: 2021-01-25
==================================================================

[SCAN] 2021-01-25 14:30 UTC
  Analyzing 200 bars
  [ELEVATED]
    EMPS Score: 0.543
    Explosion Flag: False
    Hard Flag: False
    Components:
      Vol Z-Score: 3.24
      VWAP Dev: 0.021 (2.1%)
      RV Ratio: 1.65
      Liquidity: 0.750
    Price: $38.80
    Volume: 1,800,000

[SCAN] 2021-01-25 21:00 UTC
  Analyzing 200 bars
  [!EXPLOSION!]
    EMPS Score: 0.687
    Explosion Flag: True
    Hard Flag: False
    Components:
      Vol Z-Score: 5.12
      VWAP Dev: 0.034 (3.4%)
      RV Ratio: 1.92
      Liquidity: 0.750
    Price: $65.00
    Volume: 50,000,000

==================================================================
BACKTEST SUMMARY
==================================================================

Ticker: GME
Data File: data/GME_5m_Jan2021.csv
Total Trading Days: 5
Total Scans: 10

Detection Statistics:
  Explosion Flags: 7
  Hard Flags: 2
  Days with Explosions: 4
  Detection Rate: 80.0% of days

EMPS Scores:
  Max EMPS Score: 0.892
  Avg EMPS Score: 0.621

First Explosion Detected:
  2021-01-25 21:00 UTC

==================================================================
```

---

## Files

### emps_backtest_simulator.py

**Main backtesting script** - Simulates scheduled EMPS pipeline runs

**Features:**
- Reads historical CSV data
- Simulates daily scans at predefined UTC times
- Logs EMPS signals and component values
- Generates summary statistics
- Exports results to CSV

**Usage:**
```bash
python emps_backtest_simulator.py --help
```

---

## Example Workflows

### Workflow 1: Test Single Explosive Move

**Objective:** Validate EMPS detected GME squeeze

```bash
# 1. Prepare data (GME Jan 2021, 5m bars)
# Download from data provider or use cached data

# 2. Run backtest
python emps_backtest_simulator.py \
    --data-file data/GME_5m_Jan25-29_2021.csv \
    --ticker GME \
    --scan-time "14:30,21:00" \
    --output results/GME_Jan2021_backtest.csv

# 3. Review results
cat results/GME_Jan2021_backtest.csv
```

**Expected Output:**
- First explosion flag: Jan 26 or Jan 27
- Max EMPS score: 0.80-0.95
- Multiple explosion flags during squeeze

---

### Workflow 2: Test Multiple Cases

**Objective:** Test all 24 explosive moves from test cases

```bash
# Create script to batch test
for ticker in GME AMC CLOV SPRT BBBY HKD; do
    echo "Testing $ticker..."
    python emps_backtest_simulator.py \
        --data-file data/${ticker}_5m.csv \
        --ticker $ticker \
        --scan-time "14:30,21:00" \
        --output results/${ticker}_backtest.csv
done
```

---

### Workflow 3: Compare Scan Times

**Objective:** Determine optimal scan time (market open vs close)

```bash
# Market open scan
python emps_backtest_simulator.py \
    --data-file data/GME_5m_Jan2021.csv \
    --ticker GME \
    --scan-time "14:30" \
    --output results/GME_market_open.csv

# Market close scan
python emps_backtest_simulator.py \
    --data-file data/GME_5m_Jan2021.csv \
    --ticker GME \
    --scan-time "21:00" \
    --output results/GME_market_close.csv

# Compare detection rates
```

---

## Data Sources

### Option 1: FMP Historical Data

Use FMP API to download historical intraday data:

```python
from src.data.downloader.fmp_data_downloader import FMPDataDownloader

fmp = FMPDataDownloader()

# Note: FMP historical intraday has limitations
# May only have recent months available
```

### Option 2: Polygon.io

Polygon has excellent historical coverage:

```python
import requests

# Polygon API (requires subscription for historical data)
url = f"https://api.polygon.io/v2/aggs/ticker/GME/range/5/minute/2021-01-25/2021-01-29"
params = {'apiKey': 'YOUR_KEY'}
response = requests.get(url, params=params)
```

### Option 3: Alpaca

Alpaca offers free historical data:

```python
from alpaca_trade_api.rest import REST

api = REST()
bars = api.get_bars('GME', '5Min', '2021-01-25', '2021-01-29')
```

### Option 4: Local Cache

Cache historical data locally for repeated testing:

```
data/
├── GME_5m_Jan2021.csv
├── AMC_5m_Jun2021.csv
├── BBBY_5m_Aug2022.csv
└── HKD_5m_Aug2022.csv
```

---

## Output Format

The backtest exports results to CSV with these columns:

```csv
scan_datetime,scan_time_utc,emps_score,explosion_flag,hard_flag,vol_zscore,vwap_dev,rv_ratio,liquidity_score,close_price,volume,bars_analyzed
2021-01-25 14:30:00,14:30,0.543,False,False,3.24,0.021,1.65,0.750,38.80,1800000,200
2021-01-25 21:00:00,21:00,0.687,True,False,5.12,0.034,1.92,0.750,65.00,50000000,200
...
```

---

## Interpreting Results

### Detection Metrics

**Explosion Flags:**
- Number of scans that triggered explosion flag
- Should be 40-80% of scans during explosive move periods

**Hard Flags:**
- Number of scans that triggered hard explosion flag
- Rare, only on most extreme moves (10-20% of explosions)

**Detection Rate:**
- Percentage of trading days with at least one explosion flag
- Target: 60-80% for known explosive moves

**Max EMPS Score:**
- Highest score achieved during backtest
- Target: > 0.75 for explosive moves, < 0.60 for normal stocks

### Component Analysis

**Vol Z-Score:**
- > 4.0: Significant volume spike
- > 6.0: Extreme volume spike
- GME peak: 8-12

**VWAP Deviation:**
- > 0.03 (3%): Significant price dislocation
- > 0.05 (5%): Extreme dislocation
- GME peak: 8-15%

**RV Ratio:**
- > 1.8: Volatility acceleration
- > 2.2: Strong acceleration
- GME peak: 3-4x

---

## Troubleshooting

### Issue: "No data available at scan time"

**Cause:** Scan time is before market open or after available data

**Solution:** Check data timestamp range and adjust scan times

### Issue: "Insufficient bars"

**Cause:** Not enough historical bars for EMPS calculation

**Solution:**
- Use more days of data
- Ensure at least 200 bars before first scan

### Issue: "All scans show NORMAL"

**Possible causes:**
1. Data doesn't include explosive move period
2. EMPS thresholds too high
3. Data quality issues (missing volume, etc.)

**Solution:**
- Verify date range includes explosive move
- Lower threshold: `--emps-threshold 0.5`
- Check data quality (no zeros, correct format)

### Issue: "Too many explosion flags"

**Cause:** Thresholds too low, detecting normal volatility

**Solution:** Increase threshold: `--emps-threshold 0.7`

---

## Advanced Usage

### Custom EMPS Parameters

Modify the script to use custom parameters:

```python
# In emps_backtest_simulator.py, modify:
emps_params = {
    **DEFAULTS,
    'vol_zscore_thresh': 3.5,      # Lower threshold
    'combined_score_thresh': 0.5,  # Lower threshold
    'weights': {
        'vol': 0.50,               # Higher volume weight
        'vwap': 0.25,
        'rv': 0.20,
        'liquidity': 0.05,
    }
}
```

### Batch Processing

Create a batch script to test all cases:

```bash
#!/bin/bash
# batch_backtest.sh

CASES=(
    "GME,2021-01-25,2021-01-29"
    "AMC,2021-06-01,2021-06-04"
    "BBBY,2022-08-15,2022-08-18"
    "HKD,2022-08-01,2022-08-05"
)

for case in "${CASES[@]}"; do
    IFS=',' read -r ticker start end <<< "$case"
    echo "Processing $ticker..."

    python emps_backtest_simulator.py \
        --data-file data/${ticker}_5m.csv \
        --ticker $ticker \
        --scan-time "14:30,21:00" \
        --output results/${ticker}_results.csv
done

echo "Batch complete!"
```

---

## Next Steps

1. **Collect Historical Data** - Download data for 24 test cases
2. **Run Backtests** - Test all explosive moves
3. **Analyze Results** - Calculate recall, precision, F1 score
4. **Optimize Parameters** - Adjust thresholds based on results
5. **Validate** - Test on holdout set of 10+ additional cases

---

## References

- [BACKTESTING_PROPOSAL.md](../docs/BACKTESTING_PROPOSAL.md) - Full backtesting proposal
- [EMPS_DETAILED_EXPLANATION.md](../docs/EMPS_DETAILED_EXPLANATION.md) - Component explanations
- [DATA_OPTIMIZATION_GUIDE.md](../docs/DATA_OPTIMIZATION_GUIDE.md) - Performance guide

---

**Last Updated:** 2025-11-22
**Status:** Ready for Testing
