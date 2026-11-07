# Requirements

## Python Dependencies
- `pandas` >= 3.0 (project-wide pin)
- `numpy`
- `yfinance` (via `YahooDataDownloader`; `market_agent.py` also imports it directly)
- `optuna` - only for `strategy_sim.py optimize`
- `pytest` - tests

All pinned in the project root `requirements*.txt`; nothing P17-specific.

## External Dependencies
- `src.ml.pipeline.shared.universe_downloader` - NASDAQ / NYSE American symbol universe (HTTPS with FTP fallback)
- `src.ml.pipeline.shared.config` - `FundamentalFilterConfig`, `UniverseConfig` base classes
- `src.ml.pipeline.shared.edgar_cik` - ticker → CIK map for dilution and catalyst lookups
- `src.ml.pipeline.shared.trf_downloader` - FINRA daily short-volume (optional)
- `src.data.downloader.yahoo_data_downloader` - fundamentals snapshot and batch OHLCV
- `src.data.downloader.edgar_downloader` - EDGAR submissions, the daily 8-K index, and the Form 4 / 13D/G cache
  (`DATA_CACHE_DIR/edgar/13f/{form4,13dg}`) written by the P15 daily job (P17 reads it, never downloads)
- `src.ml.pipeline.p20_kestrel` (`KestrelService.get_all_upcoming_catalysts`) - `k20_catalysts` table (optional;
  the stage degrades to no column when the DB is unavailable)
- `src.data.downloader.fear_greed_downloader` - market sentiment line in the report (optional)
- `src.data.cache.unified_cache` / `fundamentals_cache` - shared OHLCV and fundamentals caches
- `src.data.db.services.notification_service` / `users_service` - email queue and recipient resolution
- `src.notification.logger` - logging

Consumers of P17 output (must stay compatible with the candidates CSV columns `ticker`, `tier`,
`explosive_candidate`, `final_score`, `setup_candidate`, `fresh_13g`, `avg_volume_30d`, `price`):
- `src.ml.pipeline.p19_penny_intraday.watchlist_builder`
- `src.ml.pipeline.signal_ledger.harvesters.harvest_p17`

## External Services
- Yahoo Finance (rate-limited; ~7k `info` calls per run, cached for 1 day)
- SEC EDGAR (submissions JSON + daily 8-K index, the latter written by the P15 daily bundle)
- FINRA Reg SHO daily files (optional)
- PostgreSQL (notification queue, users) - only for the Stage 8 email

## System Requirements
- Runs on the Raspberry Pi scheduler (`scheduler.service`); job timeout 7200 s, typical run ~7 min
- Memory: well under 1 GB (≈150 candidates after filters)
- Storage: ~2 MB per daily results folder

## Security Requirements
- No credentials in the pipeline: email goes through the notification DB service, recipient resolved by `user_id`
- External text (company names, filing descriptions) is HTML-escaped in emails and sanitised in Markdown tables

## Performance Requirements
- Complete before the US open (scheduled 08:00 UTC; 7200 s hard timeout)
- Stage 1 (universe snapshot) dominates; checkpoint/resume survives interruptions
