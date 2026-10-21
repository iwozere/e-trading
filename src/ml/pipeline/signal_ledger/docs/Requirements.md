# Requirements

## Python Dependencies
- `pandas` >= 3.0 (price panels, cohort statistics)
- `numpy` (Newey-West variance)
- `pyarrow` (parquet price cache)
- `sqlalchemy` >= 2.0, `psycopg2` (Postgres upsert with `ON CONFLICT`)
- `yfinance` (through `src.data.downloader.yahoo_data_downloader`)

## External Dependencies
- `src.data.db` - `model_signal_ledger`, `repo_signal_ledger`, `signal_ledger_service`; `ReposBundle.signal_ledger`
- `src.data.downloader` - `DataDownloaderFactory` / `YahooDataDownloader.get_ohlcv_batch` (auto-adjusted daily bars)
- `src.data.pipeline` - `specs/signal_ledger_specs.py` registers the nightly job
- `src.api` - `signal_ledger_routes.py` imports `scorecard.build_scorecard` and `portfolio.summarize_book`
- `src.notification.logger` - logging
- Read-only inputs written by other pipelines: `results/p05_ai_selector`, `p06_emps2`, `p10_emps3`, `p17_penny_stocks`,
  `p19_penny_intraday/shadow.sqlite` (opened `mode=ro`), `p21_momentum`

## External Services
- Yahoo Finance chart API (daily bars; no crumb needed)
- PostgreSQL: migration `010_signal_ledger` must be applied before the first run

## System Requirements
- Backfill of ~25k signals / ~1.1k tickers: about 2-3 minutes, a few hundred MB RAM
- Price cache: a few MB per year of history

## Security Requirements
- No credentials of its own; uses the shared DB URL
- Read-only access to other pipelines' results

## Performance Requirements
- Nightly run well under the 3,600 s timeout (only recent folders are re-harvested; only tickers with pending signals are refetched)
