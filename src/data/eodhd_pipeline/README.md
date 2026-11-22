# EODHD Pipeline

## Overview

Downloads and keeps EODHD's daily price history for US common stocks, delisted ones included. It closes the main
gap of the FMP vintage: stocks that stopped trading before 2021 (the vintage covers about 10% of the 2012–2020
delistings and almost none before).

EODHD's terms let a private user keep what was downloaded, so this is a kept dataset, not a research window like
Sharadar. It has prices, splits and dividends only: no fundamentals, no delisting reasons or deal terms.

**Status: built and tested; mechanics run on the free key (2026-11-22). Needs the paid "EOD Historical Data — All
World" plan ($19.99 a month) for the real download.**

## Read first

- [Runbook](docs/Runbook.md) — purchase day, step by step
- [Design](docs/Design.md) — layout, EODHD facts the conversion relies on, decisions
- [Requirements](docs/Requirements.md)
- [Tasks](docs/Tasks.md)

## Features

- `universe`: EODHD's live and delisted US symbol lists (2 calls), filtered to common stock on the main exchanges
  (about 16,700 delisted and 5,900 live codes), each code given a stable id from 1,000,001 up
- `download`: daily bars, splits and dividends per code (3 calls each, about 68,000 in all), resumable, paced
  under the per-minute limit, stops cleanly on the call budget or the daily quota; refuses a free-plan key
- `build`: Parquet store with raw (as-traded) prices and volume, splits, raw-basis dividends, mapping of each
  security to its FMP vintage id (symbol, overlapping dates, company name), and a table of how many delisted stocks
  per period the FMP vintage lacks
- `crosscheck`: our adjustment against EODHD's adjusted close, and prices, volumes, splits and dividends against
  the FMP vintage
- `EodhdStore`: same reader shape as `FmpStore`, so the P24 helpers and the event bench run on it
- `quota` / `status`: plan and calls left today (free of charge); progress and store summary

## Quick Start

```bash
python -m src.data.eodhd_pipeline.cli quota
python -m src.data.eodhd_pipeline.cli universe
python -m src.data.eodhd_pipeline.cli download
python -m src.data.eodhd_pipeline.cli build
python -m src.data.eodhd_pipeline.cli crosscheck
python -m src.data.eodhd_pipeline.cli status
```

```python
from datetime import date
from typing import Any, cast

from src.data.eodhd_pipeline.config import default_data_root
from src.data.eodhd_pipeline.store import EodhdStore
from src.ml.pipeline.p24_falling_knife.data import load_daily_batch
from src.ml.pipeline.p24_falling_knife.universe import is_common_stock, load_securities

store = EodhdStore(default_data_root())
delisted = [s for s in load_securities(store.dir) if is_common_stock(s) and s.delisted]
daily = load_daily_batch(cast(Any, store), [s.security_id for s in delisted[:100]], date(1998, 1, 1), date(2020, 12, 31))
```

`load_daily_batch` is annotated for `FmpStore`; `EodhdStore` has the same methods (cast it for the type checker).

## Integration

- `src.data.fmp_pipeline` — vintage securities (mapping), `adjust()`, reader shape
- `src.ml.pipeline.p24_falling_knife` — universe and daily preparation helpers used by the screens
- `src.ml.research.event_bench` — forward-return engine for event studies
- `src.data.sharadar_pipeline` — its `eodhd-check` (depth test against Sharadar's ticker list) uses this client

## Configuration

- `EODHD_API_KEY` in the project configuration (the existing key; upgrading the plan does not change it)
- Data root `<DATA_CACHE_DIR>/eodhd-data/` (outside the repository)
