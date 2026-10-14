# Requirements

## Python Dependencies
- `pandas` >= 3.0.3, `numpy` >= 2.3.2 — `adjust/` (already in `requirements.txt`)
- `httpx` >= 0.28.1 — `probe/` (in `requirements.txt`)
- Planned (plan §9.2): `pyarrow`, `duckdb` (owner decision Q6), `tzdata`, optionally `orjson`, split into
  `requirements-fmp-reader.txt` (Pi) and `requirements-fmp-pipeline.txt` (laptop).

## External Dependencies
- `src.config.provider_config` — FMP API key resolution (env var, then `donotshare`).
- `config.donotshare.donotshare` — `DATA_CACHE_DIR`.
- `src.notification.logger` — `setup_logger`.
- Planned: `src.data.data_manager` (WP-8 adapter), `src.data.cache.unified_cache` (legacy layer, read only),
  `src.data.downloader.yahoo_data_downloader` (WP-9 increments), `src.data.pipeline` (job registration).

## External Services
- Financial Modeling Prep `/stable` API. Free key now (probe, ≤ 200 calls); Premium for one month after gate G1.
  Free key: 250 calls/day, per-symbol entitlement list, 500 MB/30 days. Premium: 750 calls/min, 50 GB/30 days.

## System Requirements
- Writer: Windows 11 laptop. Probe cache is a few MB. The full dump needs ~30–47 GB under `data_root` (spec §6.8).
- Reader: Raspberry Pi 5 (aarch64), `reader` profile only.

## Security Requirements
- The API key travels only in the `apikey` header and is redacted if a response echoes it.
- Nothing under `data_root` (raw responses, archive, vintages) is ever committed or uploaded (spec D5). Test
  fixtures are synthetic.

## Performance Requirements
- `adjust()` is vectorized (`numpy.searchsorted` + suffix products): O((bars + events) log events).
- Spec §13 targets (builder, Pi reads) apply from WP-6/WP-8 on.
