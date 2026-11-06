# Requirements

## Python Dependencies
- `numpy`, `pandas` (repo versions; pandas 3 compatible)
- `TA-Lib` 0.8.3 (ADX, +DI, −DI, CCI; **not** PPO — its default is an SMA)
- `pyarrow`, `duckdb` (via the FMP vintage reader)
- `pandas_market_calendars` (XNYS sessions for incomplete-week detection)

## External Dependencies
- `src.data.fmp_pipeline.datamanager_adapter.fmp_store` — `FmpStore` (`bulk_daily`, `events_many`, `resolve`)
- `src.data.fmp_pipeline.adjust.adjust` — split adjustment at read time
- `src.data.fmp_pipeline.builders.flags` — `OHLC_VIOLATION`, `SPIKE_REVERTED`, `JUMP_NO_SPLIT`
- `src.data.fmp_pipeline.builders.validate` — `SECURITIES` reference-table path
- `src.notification.logger` — `setup_logger`
- Phase 2 (live scan): `src.data.data_manager.DataManager` (`get_ohlcv_batch`), `src.ml.pipeline.shared.universe_downloader`
  (`create_universe_downloader`, `UniverseConfig`), `src.data.db.services.notification_service.NotificationService`,
  `src.data.db.services.users_service.UsersService`, `src.data.pipeline` (`PluginSpec`, registry),
  `src.scheduler.scheduler_service` (`--user-id` allowlist), `src.ml.pipeline.signal_ledger` (`harvest_p24`)
- `matplotlib` (Agg) for charts

## External Services
- None at runtime for Phase 1: the FMP vintage `2026-10` is read from local disk (`D:/data-cache/fmp-data`).
- Phase 2: NASDAQ Trader symbol directory (FTP/HTTPS), Yahoo Finance through `DataManager`, PostgreSQL notification
  queue (Telegram + SMTP delivered by the notification processor).

## System Requirements
- Full backtest: ~8,800 common stocks × 2 timeframes × 8 variants; ~5–10 min with 7 worker processes on the dev machine.
- Memory: one chunk of 150 securities' daily history per worker (well under 1 GB each).

## Security Requirements
- No secrets; read-only access to the vintage. Never write into `D:/data-cache/fmp-data` or `R:`.

## Performance Requirements
- `build_context` ~0.06 s and `detect` ~0.02 s per security-timeframe (daily, full history).
