# Requirements

## Python Dependencies
- `pandas_market_calendars >= 5.4.0` — NYSE (XNYS) trading calendar; new dependency added for this module
  (`requirements.txt`), not used anywhere else in the repo prior to P21
- `yfinance >= 0.2.64` — already a project dependency; used directly (not via `YahooDataDownloader`) only in
  `data/earnings.py`, the one place P21 bypasses the shared downloader (no downloader wraps `get_earnings_dates()`)
- `pandas`, `numpy` — already project dependencies

## External Dependencies
- `src.data.downloader` (`DataDownloaderFactory`, `YahooDataDownloader`) — all OHLCV and fundamentals fetching
- `src.util.tickers_list` — `get_sp500_constituents_with_sector()` (added alongside this module)
- `src.notification.logger` — `setup_logger()`, used by every module in this package
- `src.notification.service.client` — `NotificationServiceClient`, used for ABORT-level gate alerts
- `src.data.db` (`session_scope`, `Schedule` model) — used only by `jobs/register_jobs.py`

## External Services
- Yahoo Finance (via `yfinance`/`YahooDataDownloader`) — equity OHLCV, fundamentals, earnings dates
- Wikipedia (`List_of_S%26P_500_companies`) — S&P 500 constituents + GICS sector, via `src.util.tickers_list`
- No API keys required — all data sources are the same free/unauthenticated sources the rest of the repo uses

## System Requirements
- No GPU, no special memory requirements — signal computation over ~500 tickers x ~400 daily bars is a
  lightweight pandas workload (well under a second per ticker)
- Disk: `results/p21_momentum/` grows by one dated folder per job run (~three per month) plus the append-only
  `_state/ledger.jsonl` and `_state/nav_daily.csv`; negligible for years of operation at this position count

## Security Requirements
- No API keys or secrets — all data sources are unauthenticated
- `config/pipeline/p21_exclusions.json` and `config/frozen_params.json` are checked into git (project
  convention for operator-facing config, matching P20's `activists.json`) — no credentials, so no
  `donotshare` handling needed

## Performance Requirements
- `monthly_rebalance` and `monthly_execute` run once a month each; `daily_mark` runs once per trading day —
  none are latency-sensitive, and none have a hard runtime SLA (unlike the backtest harness's own §14.9 B9
  30-minute full-history requirement, which is a `backtest/p21_momentum/` concern, not this package's)
- Every job is idempotent and safe to re-run (`--force` bypasses the idempotency skip) — see spec §3
