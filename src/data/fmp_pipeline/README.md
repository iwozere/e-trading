# FMP Historical Data Pipeline

## Overview
One-off, resumable download of FMP Premium history (raw prices, corporate events, fundamentals, delisted
symbols) into an immutable Parquet "vintage", plus read-time adjustment and a yfinance increment layer. Full
specification: `docs/FMP_DATA_PIPELINE_SPEC-2.md`; build plan: `docs/FMP_DATA_PIPELINE_IMPLEMENTATION_PLAN.md`.

**Status (2026-10-14):** WP-1 (free-key probe) and WP-7 (`adjust()`) are done. Everything else in Phase 1 is still
to build; see [Tasks](docs/Tasks.md). Do **not** buy FMP Premium before gate G1.

## Features
- `adjust/` — pure, vectorized `adjust(prices_raw, splits, dividends, mode)` with `mode` in `raw`/`split`/`total`
  (spec §8), plus `yfinance_to_raw()` to un-adjust yfinance increments (spec §10). Verified against FMP's own
  adjusted series to ≤ 2.5e-5 (probe F4).
- `probe/` — `fmp probe-free`: budgeted (≤ 200 calls), cached, idempotent free-key probe that writes
  `docs/fmp_probe_report.md` and synthetic test fixtures.
- `cli.py` — the `fmp` command line (`probe-free` so far).

## Quick Start

```python
import pandas as pd

from src.data.fmp_pipeline.adjust.adjust import adjust

raw = pd.DataFrame({"close": [400.0, 101.0]}, index=pd.to_datetime(["2020-08-28", "2020-08-31"]))
splits = pd.DataFrame({"date": ["2020-08-31"], "numerator": [4], "denominator": [1]})
dividends = pd.DataFrame({"date": ["2020-08-31"], "amount": [0.205]})  # raw basis of the ex-date (FMP `dividend`)
adjusted = adjust(raw, splits, dividends, mode="total")
```

Run the probe (PowerShell, project root):

```powershell
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli probe-free            # live, within the call budget
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli probe-free --offline  # rebuild report from cache only
```

## Integration
- `src.config.provider_config.get_api_key("FMP_API_KEY")` — the single place the FMP key is resolved.
- `config.donotshare.donotshare.DATA_CACHE_DIR` — `data_root` is `DATA_CACHE_DIR/fmp-data`.
- `src.notification.logger` — logging.
- Later: `src.data.data_manager.DataManager` (WP-8 adapter), `src.data.pipeline` (scheduled `PluginSpec`s).

## Configuration
- `FMP_API_KEY` (environment variable or `config/donotshare/donotshare.py`). Sent only as the `apikey` request
  header, never in a URL.
- Probe cache and call ledger: `DATA_CACHE_DIR/fmp-data/probe-free/`. Never commit anything from `data_root` (D5).

## Related Documentation
- [Requirements](docs/Requirements.md) - Technical requirements
- [Design](docs/Design.md) - Architecture and design
- [Tasks](docs/Tasks.md) - Implementation roadmap
- `docs/decisions.md`, `docs/fmp_probe_report.md` (repository level)
