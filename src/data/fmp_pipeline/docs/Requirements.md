# Requirements

## Python Dependencies
- `pandas` >= 3.0.3, `numpy` >= 2.3.2 — `adjust/` (already in `requirements.txt`)
- `httpx` >= 0.28.1 — `probe/`, `client/` (in `requirements.txt`)
- `sqlite3` (stdlib) — `manifest/`; `gzip`, `zlib` (stdlib) — `archive/`
- `PyYAML` >= 6.0.3 — `planner/config.py` (in `requirements.txt`)
- `pyarrow` >= 25.0.1 — `builders/` (Parquet), `tzdata` — zoneinfo rules for intraday time on Windows
- `duckdb` >= 1.5.6 — `datamanager_adapter/fmp_store.bulk_daily` (imported lazily; owner decision Q6, D55)
- Tests: `pytest-asyncio` (in `requirements-dev.txt`)
- Profiles (plan §9.2): `requirements-fmp-reader.txt` (Pi: numpy, pandas, pyarrow, tzdata) and
  `requirements-fmp-pipeline.txt` (laptop: reader + httpx, PyYAML). The reader profile includes `duckdb`.

## External Dependencies
- `src.config.provider_config` — FMP API key resolution (env var, then `donotshare`).
- `config.donotshare.donotshare` — `DATA_CACHE_DIR`.
- `src.notification.logger` — `setup_logger`.
- `src.data.data_manager` — hooks the adapter in front of the legacy flow (`_load_fmp_adapter`, D50); the golden
  snapshot drives `DataManager.get_ohlcv` offline.
- Planned: `src.data.downloader.yahoo_data_downloader` (WP-9 increments), `src.data.pipeline` (job registration).

## External Services
- Financial Modeling Prep `/stable` API. Free key now (probe, ≤ 200 calls); Premium for one month after gate G1.
  Free key: 250 calls/day, per-symbol entitlement list, 500 MB/30 days. Premium: 750 calls/min, 50 GB/30 days.

## System Requirements
- Writer: Windows 11 laptop. Probe cache is a few MB. The full dump needs ~30–47 GB under `data_root` (spec §6.8).
- The raw archive pauses the run below 2 GB free; `fsutil.ensure_free_space` (need + 20%) is the per-phase check
  that `fmp plan`/`run`/`build` call.
- One writer per dump: `RawArchive` is not safe for two processes appending to the same `dump_id`; the CLI holds
  an OS lock on `<data_root>/locks/<dump_id>.lock` (D40).
- Reader: Raspberry Pi 5 (aarch64), `reader` profile only.

## Security Requirements
- The API key travels only in the `apikey` header and is redacted if a response echoes it.
- Nothing under `data_root` (raw responses, archive, vintages) is ever committed or uploaded (spec D5). Test
  fixtures are synthetic.
- Archive lines never contain the key: it is not a query parameter, and any `apikey` parameter is dropped anyway.

## Performance Requirements
- `adjust()` is vectorized (`numpy.searchsorted` + suffix products): O((bars + events) log events).
- `fmp build` holds one security's bars at a time; the events tables and the securities list (~100k rows) are the
  only whole tables in memory. Record tables stream in batches of `build.batch_rows` (50k). Spec §13: < 4 GB RSS.
- Adapter reads (spec §9.5): one security's whole history ~40 ms first read, ~22 ms cached; 50 securities × 2 years
  in one batch ~1.2 s (measured on a synthetic vintage, `docs/datamanager_integration.md` §4).
