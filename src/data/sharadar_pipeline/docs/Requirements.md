# Requirements

## Goal

Use a time-limited Sharadar subscription to answer research questions that the FMP vintage cannot answer because
it lacks stocks delisted before 2021, and keep only results that the licence allows keeping.

## Functional requirements

| Id | Requirement |
|---|---|
| R1 | Download every Bundle table in full history through the bulk endpoint, resumable, with file name, size, hash and server last-modified time recorded |
| R2 | Keep the downloaded zips unchanged; build everything else from them |
| R3 | Convert `stocks` to raw (as-traded) OHLCV using `closeunadj / close` |
| R4 | Convert `actions` to splits and raw-basis dividends in the FMP vintage event layout |
| R5 | Keep Sharadar's `permaticker` as `security_id`; record each security's FMP vintage `security_id` (same CIK, same symbol first, then longest date overlap) in `fmp_security_id` |
| R6 | Expose delisting date, reason and acquisition consideration so a delisting return can be computed |
| R7 | Serve fundamentals point-in-time: AR dimensions only, available from the day after `date` (the filing date) |
| R8 | Provide a reader with the same output shape as `FmpStore.bulk_daily` and `events` |
| R9 | Cross-check Sharadar against the FMP vintage on the overlap and write a report to the data root (not to git) |
| R10 | Purge: delete the whole data root on every machine, list what was deleted, and fail loudly if anything remains |
| R11 | Every kept result must be an aggregate or model output that cannot reproduce Sharadar tables |

## Licence requirements

| Id | Requirement |
|---|---|
| L1 | Personal use by the account owner only |
| L2 | All Sharadar data deleted within 30 days of the subscription ending, including the Pi and backups |
| L3 | No Sharadar data, key, or evaluation of the data in the public repository |
| L4 | No sharing of data or access |

## Python Dependencies

- `httpx` >= 0.28.1
- `pandas`, `pyarrow`, `numpy` (project versions)
- `duckdb` >= 1.5 (optional, for large joins)

## External Dependencies

- `src.data.fmp_pipeline` — vintage securities table, `adjust()`, reader interface
- `src.notification.logger` — logging
- `src.ml.pipeline.p24_falling_knife`, `src.ml.research.event_bench` — research helpers used by the screens

## External Services

- Sharadar API `https://api.sharadar.com/v1.0`, Bundle Full History subscription
- Free sample key `test-api-key` for development before purchase

## System Requirements

- Disk: about 10 GB for zips, extracted CSV and parquet (estimate; confirm with bulk status on purchase day)
- Memory: `stocks` is the largest table; convert in chunks

## Security Requirements

- `SHARADAR_API_KEY` only in `.env`
- Data root outside the repository and excluded from all backup jobs

## Performance Requirements

- Full download and conversion within one day, so the remaining window is research time
