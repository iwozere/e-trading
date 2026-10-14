# FMP Historical Data Pipeline

## Overview
One-off, resumable download of FMP Premium history (raw prices, corporate events, fundamentals, delisted
symbols) into an immutable Parquet "vintage", plus read-time adjustment and a yfinance increment layer. Full
specification: `docs/FMP_DATA_PIPELINE_SPEC-2.md`; build plan: `docs/FMP_DATA_PIPELINE_IMPLEMENTATION_PLAN.md`.

**Status (2026-10-14):** WP-1 (free-key probe), WP-7 (`adjust()`), WP-3 (mock server), WP-2 (client, manifest,
downloader), WP-5 (raw archive) and WP-4 (planner, `fmp plan`) are done. Everything else in Phase 1 is still to build; see [Tasks](docs/Tasks.md). Do **not** buy
FMP Premium before gate G1.

## Features
- `adjust/` — pure, vectorized `adjust(prices_raw, splits, dividends, mode)` with `mode` in `raw`/`split`/`total`
  (spec §8), plus `yfinance_to_raw()` to un-adjust yfinance increments (spec §10). Verified against FMP's own
  adjusted series to ≤ 2.5e-5 (probe F4).
- `probe/` — `fmp probe-free`: budgeted (≤ 200 calls), cached, idempotent free-key probe that writes
  `docs/fmp_probe_report.md` and synthetic test fixtures.
- `client/`, `manifest/`, `runner.py` — async FMP client (retries, circuit breaker, byte counting), SQLite task
  manifest and the resumable `Downloader` worker pool with quota/auth/bandwidth stops (spec §6.2–§6.4).
- `archive/` — append-only raw archive `raw/<dump_id>/<endpoint>/<symbol>.jsonl.gz` (spec §6.6): `RawArchive` is
  the `Downloader` sink, `reader.iter_records` / `latest_successful` read it back, `symbol_path` makes symbols
  Windows-safe file names.
- `planner/` — `fmp plan`: fetches the universe lists, assembles the universe (spec §7.2, `universe_report.md`),
  generates tasks in spec §6.3.5 order with tiers A–E, and projects bandwidth (wire and decoded) and disk need
  into `budget_plan.md`. `FollowUps` is the run-time reviewer for paging and intraday truncation.
- `cli.py` — the `fmp` command line (`probe-free`, `plan`).
- `tests/mock_server.py` — offline mock of FMP `/stable` (spec §6.7) for the downloader test suite:
  `with MockFmpServer(MockConfig(...)) as fmp: client.get(fmp.base_url + "profile", ...)`.

## Quick Start

```python
import pandas as pd

from src.data.fmp_pipeline.adjust.adjust import adjust

raw = pd.DataFrame({"close": [400.0, 101.0]}, index=pd.to_datetime(["2020-08-28", "2020-08-31"]))
splits = pd.DataFrame({"date": ["2020-08-31"], "numerator": [4], "denominator": [1]})
dividends = pd.DataFrame({"date": ["2020-08-31"], "amount": [0.205]})  # raw basis of the ex-date (FMP `dividend`)
adjusted = adjust(raw, splits, dividends, mode="total")
```

Archive downloads and read them back:

```python
from src.data.fmp_pipeline.archive.archive import RawArchive
from src.data.fmp_pipeline.archive.reader import iter_records, latest_successful
from src.data.fmp_pipeline.runner import Downloader

archive = RawArchive(data_root / "raw", "2026-10")
summary = await Downloader(manifest, client, sink=archive).run()   # stop_reason "archive" if the drive vanishes
latest = latest_successful(iter_records(archive.path_for("splits", "AAPL")))
```

Plan a dump (PowerShell, project root). Planning is iterative: re-run it after each `fmp run` phase until no
task is deferred (D36).

```powershell
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli plan --dry-run   # reports only (fetches the universe lists)
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli plan             # also writes the tasks to the manifest
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli plan --offline   # re-plan from the archive, no calls
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
- `config/fmp/fmp.yaml` — dump id, priorities, history start, bandwidth cap, intraday settings (`max_rows` comes
  from the paid probe), universe rules. All keys optional; see `planner/config.py`.
- Under `data_root`: `raw/<dump_id>/` archive, `manifests/<dump_id>.sqlite`, `reports/<dump_id>/`, `locks/`.
- Probe cache and call ledger: `DATA_CACHE_DIR/fmp-data/probe-free/`. Never commit anything from `data_root` (D5).

## Related Documentation
- [Requirements](docs/Requirements.md) - Technical requirements
- [Design](docs/Design.md) - Architecture and design
- [Tasks](docs/Tasks.md) - Implementation roadmap
- `docs/decisions.md`, `docs/fmp_probe_report.md` (repository level)
