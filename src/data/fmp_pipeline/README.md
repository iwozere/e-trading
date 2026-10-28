# FMP Historical Data Pipeline

## Overview
One-off, resumable download of FMP Premium history (raw prices, corporate events, fundamentals, delisted
symbols) into an immutable Parquet "vintage", plus read-time adjustment and a yfinance increment layer. Full
specification: `docs/FMP_DATA_PIPELINE_SPEC-2.md`; build plan: `docs/FMP_DATA_PIPELINE_IMPLEMENTATION_PLAN.md`.

**Status (2026-10-15):** WP-1 (free-key probe), WP-7 (`adjust()`), WP-3 (mock server), WP-2 (client, manifest,
downloader), WP-5 (raw archive), WP-4 (planner, `fmp plan`), `fmp run` / `status` / `retry-failed`, WP-6
(builder, `fmp build`), WP-8 (DataManager adapter, off by default), WP-9 (yfinance increments) and WP-A
(validators, `fmp validate`, `fmp verify --final`) are done, and the probe report's `[VERIFY]` items are audited.
`fmp probe-paid`, `fmp calibrate-usage`, `fmp snapshot-universe`, `fmp backup`/`publish`/`sync`, the drift report
and `fmp archive-increments` are built: every command of spec §6.1 exists. Everything else in Phase 1 is still to build; see [Tasks](docs/Tasks.md). Do **not** buy
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
- `executor.py` — `fmp run`: the planned tasks block by block in spec order, with a disk check per block, an
  optional per-run byte limit and Windows sleep suppression (`power.py`). `status.py` — `fmp status`.
  `run_log.py` — the JSON-lines run log (`logs/<dump_id>/run-*.jsonl`): one record per request, progress every
  minute.
- `builders/` — `fmp build`: archive → Parquet vintage `vintages/<dump_id>/` (spec §7): securities with stable
  ids (`security_ids.json`), symbol history, splits/dividends, raw daily/intraday prices per security with the
  suspect-data `flags` bitmask, every other endpoint as a lossless table (`_extra`), `_meta/build_info.json`.
  Switches `current.txt` only after structural validation passes.
- `datamanager_adapter/` — serves `DataManager.get_ohlcv` from the active vintage when `fmp.adapter.enabled` is
  true (off by default): `FmpStore`, `FmpOhlcvAdapter` (`adjust="legacy"` = `total`, opt-in `raw`/`split`,
  `flags`, `source`, `exclude_suspect`, `as_of`), and `fmp golden` — the offline regression snapshot of
  `get_ohlcv` over the legacy cache. See `docs/datamanager_integration.md`.
- `increments/` — `fmp ingest-increments`: extends the vintage with yfinance bars and events, converted to
  **raw** (spec §10), only on the configured writer host; the adapter appends them after the vintage's last bar.
- `validators/` — `fmp validate`: the 23 checks of spec §11 → `validation_report.md/.json`,
  `coverage_report.md`; `fmp verify --final`: the gate-G3 checklist (checksums, backup check, read-only archive,
  READY TO CANCEL).
- `probe/` — `fmp probe-free` (Phase 0b) and `fmp probe-paid` (Phase 2: everything the free key could not answer;
  writes `docs/fmp_paid_probe_report.md` and `probe_results.json`, whose settings every command applies);
  `fmp calibrate-usage --dashboard-gb X` settles whether the 50 GB cap counts wire or decoded bytes.
- `snapshots/` — `fmp snapshot-universe`: monthly, no FMP key; US symbol directories and index lists with checksums,
  a diff against the last snapshot, and one last yfinance download of every removed symbol.
- `transfer/` — `fmp backup` (packed, verified copy on the NAS/USB), `fmp publish` (hand-off folder, `READY` when
  verified) and `fmp sync` (reader installs and switches `current.txt`).
- `increments/drift.py` — the drift report `fmp build` writes for a new vintage, and `fmp archive-increments`.
- `cli.py` — the `fmp` command line (`probe-free`, `probe-paid`, `calibrate-usage`, `plan`, `run`, `status`,
  `retry-failed`, `build`, `golden`, `ingest-increments`, `archive-increments`, `validate`, `verify`,
  `snapshot-universe`, `backup`, `publish`, `sync`).
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

Right after buying Premium (PowerShell, project root), before the first `fmp plan`:

```powershell
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli probe-paid                      # ~100 calls, a few MB
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli calibrate-usage --dashboard-gb 4.1  # later, after a few GB
```

Plan a dump (PowerShell, project root). Planning is iterative: re-run it after each `fmp run` phase until no
task is deferred (D36).

```powershell
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli plan --dry-run   # reports only (fetches the universe lists)
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli plan             # also writes the tasks to the manifest
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli plan --offline   # re-plan from the archive, no calls
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli run              # all blocks in order (or --phase N / --tier T)
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli run --max-gb 5   # stop this run after 5 GB
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli status           # safe while a run is going
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli retry-failed     # re-queue failed tasks
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli build            # vintage from the archive, no network
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli build --no-switch  # build + validate, keep current.txt
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli golden capture     # hash get_ohlcv over the legacy cache
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli golden compare     # exit 1 on any difference
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli ingest-increments --symbols AAPL --timeframes 1d  # writer only
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli validate --cache-dir D:\data-cache   # exit 1 on any FAIL
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli verify --final --backup E:\fmp-backup --confirm-postponed
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli snapshot-universe        # monthly; no FMP key
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli backup --to E:\fmp-backup  # packed + verified there
.venv\Scripts\python.exe -m src.data.fmp_pipeline.cli publish --to R:\fmp-handoff
python -m src.data.fmp_pipeline.cli sync --from /path/to/fmp-handoff            # on the Pi
```

`verify --final` first writes `checksums/<dump_id>/ARCHIVE.sha256` and `VINTAGE.sha256`; copy `raw/<dump_id>`,
`vintages/<dump_id>` and `checksums/<dump_id>` to the backup volume (same layout) and run it again with `--backup`.
Only when it prints READY TO CANCEL is the FMP subscription cancelled.

Read through the DataManager once a vintage is built (after setting `fmp.adapter.enabled: true`):

```python
from datetime import datetime

from src.data.data_manager import get_data_manager

dm = get_data_manager()
legacy_like = dm.get_ohlcv("AAPL", "1d", datetime(2020, 1, 1), datetime(2020, 12, 31))  # total-adjusted
raw = dm.get_ohlcv("AAPL", "1d", datetime(2020, 1, 1), datetime(2020, 12, 31), adjust="raw", include_flags=True)
```

`fmp build` writes `reports/<dump_id>/missing_splits_candidates.csv` (probable missing splits, for review; nothing
is inserted) and `identity_uncertain.csv` (suspected ticker reuse, renames with duplicate history). Exit code 1
means the build was rejected by structural validation and left in `vintages/.<dump_id>.rejected/`.

`fmp validate` writes `reports/<vintage>/split_without_jump_candidates.csv`: splits whose raw closes do not move
by the split ratio (`verdict=flat`: the bars before the split are probably already adjusted). Nothing is
corrected automatically. A reviewed correction, with its evidence, goes into `<data_root>/overrides/price_basis.csv`
(columns and semantics in `datamanager_adapter/price_overrides.py`); `FmpStore` applies it on read, so no rebuild
is needed, and V-OVERRIDES reports what is applied.

A typical dump: `plan` → `run` → `plan` (profiles resolve the universe) → `run` → `plan` (intraday windows from
the daily bars) → `run`, until `budget_plan.md` shows no deferred tasks. Exit codes: 0 done, 1 stopped (quota,
auth, bandwidth, archive), 2 usage/lock/not planned, 3 not enough disk.

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
- `docs/runbook.md` (owner steps: purchase, download, G3, routine, repeat dump), `docs/storage_schema.md` (tables,
  columns and read semantics), `docs/decisions.md`, `docs/fmp_probe_report.md` (repository level)
