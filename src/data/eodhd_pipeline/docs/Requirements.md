# Requirements

## Python Dependencies
- `httpx` — API client (mock transport in the tests)
- `pandas`, `numpy`
- `pyarrow` — Parquet store
- `duckdb` — price reads in `EodhdStore.bulk_daily` (as in `FmpStore`)

## External Dependencies
- `src.config.provider_config` — `EODHD_API_KEY`
- `config.donotshare.donotshare` — `DATA_CACHE_DIR`
- `src.data.fmp_pipeline` — `adjust()` (cross-check), vintage securities (`builders.validate.SECURITIES`) and
  `FmpStore` (mapping and vendor check; optional, every command runs without it)
- `src.notification.logger` — logging
- Used by: `src.data.sharadar_pipeline.eodhd_check` (client and `name_similarity`)

## External Services
- EODHD API `https://eodhd.com/api`: `/exchange-symbol-list/US` (with `delisted=1`), `/eod/{code}.US`,
  `/splits/{code}.US`, `/div/{code}.US`, `/user`
- Plan: "EOD Historical Data — All World" ($19.99 a month on 2026-11-14). The free plan serves one year of bars
  and returns 403 on splits and dividends; the download refuses it unless `--allow-limited`.
- Limits: 100,000 calls a day on paid plans, 20 on the free plan; 1,000 a minute (the client paces at 900).
  Every request costs one call, 404 included; `/user` costs none.

## System Requirements
- Storage: raw gzip JSON about 1–2 GB for 22,600 codes [VERIFY on purchase day]; the store is smaller
- Memory: the build streams prices into 64 bucket files; splits and dividends are held in memory (small)
- Time: about 68,000 calls at up to 900 a minute, roughly 1.5–4 hours depending on response times

## Security Requirements
- The API key never goes into the repository or the data root
- No EODHD data in git: the repository is public and the data is licensed for private use

## Performance Requirements
- Resumable: an interrupted or quota-stopped run continues where it stopped, without spending calls again
- Builds are repeatable from `raw/` without any API call
