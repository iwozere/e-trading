# Requirements

## Python Dependencies
- `pandas`, `numpy`, `pyarrow` (project versions)
- `duckdb` (through `EodhdStore.bulk_daily`)
- `talib` (through the P24 indicators)

## External Dependencies
- `src.data.eodhd_pipeline` — `EodhdStore`, `config.default_data_root`, `config.PRICE_BUCKETS`
- `src.data.fmp_pipeline.adjust.adjust` — `adjust()` (total-return mode)
- `src.ml.pipeline.p24_falling_knife` — `run_security`, `RunSpec`, `trades_frame`, `live_params`, `load_daily_batch`,
  `load_securities`, `is_common_stock`, `CostModel`, `simulate_portfolio`
- `src.notification.logger` — logging

## External Services
- None. Reads the local EODHD store only (no API calls).

## System Requirements
- EODHD store built (`src.data.eodhd_pipeline.cli build`), about 2 GB
- Memory: one price bucket (~850k daily rows) per worker; the month-end panel is a few million rows
- CPU: process pool, one worker per core minus one

## Security Requirements
- No keys. Cached panel and trades stay in the EODHD data root; only aggregates go to `results/` (git-ignored).

## Performance Requirements
- Panel: about 2 minutes on 7 workers; P24 on all stocks: tens of minutes; monthly strategies: minutes.
