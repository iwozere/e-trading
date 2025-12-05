# UOA (Unusual Options Activity) Analysis Pipeline

## Overview

This document outlines the implementation plan for detecting and analyzing unusual options activity using EODHD data.

## Concept

* Maintain daily options volume history per ticker (sum of all calls/puts per day, by strike)
* Calculate `unusual_volume_30d` = today's total volume (calls or total) / 30-day average (or (today - mean)/std for z-score)
* UOA Score combines multiple factors: unusual_volume, call/put skew, IV jump, OI change, strike concentration

## Data Storage

### Raw Data
- Store raw options chain data as JSON in `results/emps2/YYYY-MM-DD/raw/` for future reference
- Naming convention: `{ticker}_{date}.json`

### Processed Metrics
- Store aggregated metrics in `results/emps2/YYYY-MM-DD/uoa.csv`
- Note: Data for date D is processed on D+1 and stored in D's folder
- Example: Data for 2025-12-04 is processed on 2025-12-05 and stored in `results/emps2/2025-12-04/uoa.csv`

### Schema (uoa.csv)
- trade_date (date)
- ticker (str)
- call_volume (int) - total across all call strikes
- put_volume (int)
- call_oi (int) - total open interest for calls
- put_oi (int)
- iv_mean (float) - average implied volatility across the chain (%)
- top_strike_volume (int) - volume of the most active strike
- contracts_count (int) - number of contracts with volume > 0
- uoa_score (float) - calculated UOA score (0-100)

## Implementation Plan

1. **Data Collection**:
   - Use EODHD API to fetch options chain data
   - Process data for previous trading day
   - Store raw JSON for reference

2. **Metrics Calculation**:
   - Aggregate raw data into daily metrics
   - Calculate UOA score based on multiple factors
   - Store processed metrics in the appropriate dated folder

3. **Scheduling**:
   - Daily batch processing
   - Run after market close (next day)

4. **Storage**:
   - Maintain folder structure by date
   - Keep raw and processed data separated
   - Use efficient formats (JSON for raw, CSV for metrics)

## Next Steps
- [ ] Implement data collection from EODHD
- [ ] Add metrics calculation logic
- [ ] Set up daily batch processing
- [ ] Add error handling and logging
- [ ] Document the API and usage