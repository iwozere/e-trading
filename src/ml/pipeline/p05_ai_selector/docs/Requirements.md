# Requirements — P05 AI Selector

## Python Dependencies

- `anthropic >= 0.25.0` — Claude API SDK (NEW; add to requirements.txt)
- `pandas >= 2.0.0` — data manipulation throughout all stages
- `numpy >= 1.24.0` — ATR computation, numerical operations
- `requests >= 2.31.0` — FMP earnings calendar fetch

## Internal Module Dependencies

- `src.data.downloader.russell3000_downloader` — Russell 3000 universe (shared with P15)
- `src.data.data_manager.DataManager` — OHLCV (Stage 1) and fundamentals (Stage 2)
- `src.config.provider_config` — API key resolution (FMP, Anthropic)
- `src.notification.logger` — all modules use `setup_logger(__name__)`
- `src.ml.pipeline.p15_hidden_deps.p15_weekly` — receives `russell3000_refresh` job

## External Services

- **Anthropic API** — Claude `claude-sonnet-4-6` for Stage 3 LLM synthesis (~$0.05/run)
- **Financial Modeling Prep (FMP)** — earnings calendar (`/stable/earning-calendar`), Russell 3000 constituents (`/stable/russell-index-constituents`), fundamentals
- **P18 output CSVs** — `results/p18_institutional_flow/{date}/signals.csv`, `consensus.csv`, `form4_sells.csv`

## Environment Variables / Config Keys

- `ANTHROPIC_API_KEY` — required for Stage 3; now in `src/config/provider_config._KNOWN_KEYS`
- `FMP_API_KEY` — optional (falls back to static CSV for Russell 3000; skips earnings on missing)
- `DATA_CACHE_DIR` — from `config/donotshare/donotshare.py`

## System Requirements

- Python 3.10+
- Cold-start disk: ~500 MB for Russell 3000 OHLCV cache (60-day × ~3,000 tickers)
- Network: requires internet access to Anthropic API and FMP (Stage 3 and optional stage 1/2 gap fills)
