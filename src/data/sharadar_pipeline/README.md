# Sharadar Pipeline

## Overview

Pipeline for a time-limited research window on Sharadar data (survivorship-bias-free US daily prices and
point-in-time fundamentals back to 1998). It closes the main gap of the FMP vintage: stocks delisted before 2021.

**Status: built and tested end to end on the free sample (2026-11-14). No data bought.**

## Read first

- [Findings](docs/Findings.md) — licence, depth, coverage, quality, volume, access, other providers
- [Runbook](docs/Runbook.md) — what to do on purchase day and before the deletion deadline
- [Requirements](docs/Requirements.md)
- [Design](docs/Design.md)
- [Tasks](docs/Tasks.md)

## Three facts that shape everything

1. The personal licence requires deleting all Sharadar data within 30 days of termination. A one-month purchase is
   a research window of about 60 days, not a kept dataset.
2. Only derived results (backtests, statistics, models) may be kept. Sharadar data is never merged into the FMP
   vintage and lives in its own data root.
3. This repository is public: no Sharadar data, no API key, and no published evaluation of the data in git.

## Features

- `download`: bulk files of every Bundle table (paid key), resumable, with a manifest; or the free sample
- `build`: Parquet store with raw (as-traded) prices, splits, raw-basis dividends, delistings with deal
  consideration, and as-reported fundamentals only
- Mapping of each security to its FMP vintage `security_id` by CIK (an extra column; ids stay Sharadar's `permaticker`)
- `crosscheck`: our adjustment against Sharadar's adjusted close, and prices and splits against the FMP vintage
- `SharadarStore`: same reader shape as `FmpStore`, so the P24 helpers and the event bench run on it
- `status`: files and the deletion deadline; `purge`: delete the data root and write a report
- `eodhd-check`: measures how deep EODHD's delisted price history is, using Sharadar's free ticker list as the
  yardstick (needs a paid EODHD key; see the Runbook)

## Quick Start

```bash
python -m src.data.sharadar_pipeline.cli download --sample   # free key, 20 large caps
python -m src.data.sharadar_pipeline.cli build
python -m src.data.sharadar_pipeline.cli crosscheck
python -m src.data.sharadar_pipeline.cli purge --confirm
```

```python
from datetime import date

from src.data.sharadar_pipeline.config import default_data_root
from src.data.sharadar_pipeline.store import SharadarStore
from src.ml.pipeline.p24_falling_knife.data import load_daily_batch
from src.ml.pipeline.p24_falling_knife.universe import is_common_stock, load_securities

store = SharadarStore(default_data_root())
common = [s for s in load_securities(store.dir) if is_common_stock(s)]
daily = load_daily_batch(store, [s.security_id for s in common[:100]], date(1998, 1, 1), date(2020, 12, 31))
```

`load_daily_batch` is annotated for `FmpStore`; `SharadarStore` has the same methods (cast it for the type checker).

## Integration

- `src.data.fmp_pipeline` — vintage securities (CIK mapping), `adjust()`, reader shape
- `src.ml.pipeline.p24_falling_knife` — universe, cost model and daily preparation helpers used by the screens
- `src.ml.research.event_bench` — forward-return engine for event studies

## Configuration

- `SHARADAR_API_KEY` in `.env` (never committed)
- Data root `<DATA_CACHE_DIR>/sharadar-data/` (outside the repository, outside the FMP data root)
