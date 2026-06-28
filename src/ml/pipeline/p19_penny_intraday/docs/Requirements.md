# Requirements — P19 Intraday Penny-Stock Monitor

## Python Dependencies
- `pandas`, `requests`
- `python-dotenv` (loads `config/donotshare/.env` keys)
- `optuna` (threshold calibration, Phase 4 — reuses P17 harness)
- `anthropic` (optional LLM alert summarizer, Phase 4)

## External Dependencies (cross-module)
- `src.ml.pipeline.p17_penny_stocks` — daily watchlist output + `CatalystAgent`,
  `ShortSqueezeAgent`, `DilutionAgent`, `TechnicalAgent`, `Candidate` model
- `src.data.data_manager` / `src.data.downloader.*` — Finnhub, Polygon, yfinance feeds
- `src.data.downloader.edgar_downloader` — fresh-8-K catalyst (EFTS / daily index)
- `src.common.sentiments` — Reddit/StockTwits/Trends/NewsAPI/FinBERT adapters (context)
- `src.data.db.services.notification_service` / `users_service` — alert delivery
- `src.notification.logger` — logging

## External Services / Keys (`config/donotshare/.env`)
- `FINNHUB_API_KEY` (free) — real-time `/quote` (price). Intraday candles are premium.
- `POLYGON_API_KEY` (free) — intraday `/aggs` (volume), ~15-min delayed, ~5 req/min.
- EDGAR (no key, fair-use headers).
- Optional: Reddit/StockTwits/NewsAPI keys for sentiment adapters.

## System Requirements
- A market-hours scheduler slot (intraday cron, UTC, DST-aware) running `run-once`.
- Persistent store for the shadow dataset (SQLite/Parquet) and per-day state files.

## Performance / Rate Requirements
- Watchlist cap × poll interval must fit provider limits (§13.1): Finnhub ~60/min
  → N≈60 @1-min or N≈300 @5-min; Polygon ~5/min → volume only for triggered subset.
- Price poll latency target: < a few seconds per sweep (Finnhub ~140 ms/call).

## Security
- API keys only from `config/donotshare/.env`; never hard-coded or logged.
