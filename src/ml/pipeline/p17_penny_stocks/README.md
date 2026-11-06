# P17 Explosive Penny Stock Screener

## Overview
Daily screener for US penny stocks ($0.50–$10, $30M–$2B market cap, 5M–50M float) on NASDAQ and NYSE American.
Each weekday at 08:00 UTC it scores the previous session's universe on momentum, volume, technical structure,
fundamentals, SEC 8-K catalysts, short interest and accumulation, deducts dilution penalties, assigns tiers
A / B / C / W, and emails the actionable Tier A/B names. The full specification is
[docs/pipeline-specification.md](docs/pipeline-specification.md).

Since 2026-11-06 an **extension gate** keeps names whose move already happened out of Tier A/B (prod signals used to
fire at a median +43% 5-day run-up and lose ~9% over the next 10 sessions). A **pre-breakout setup score** is computed
in shadow mode: it is reported and tracked by the signal ledger but not emailed, because replay showed no edge yet.

## Features
- Universe download + hard filters (exchange, price, market cap, float, liquidity, cash runway / debt survival).
- Technical features from 90-day daily OHLCV: RVOL, 5/20/60d returns, SMA20/50, 20/50d breakouts, Bollinger squeeze,
  ATR, OBV slope, accumulation days, and setup features (distance to 50d high, ATR stretch over SMA20, 10d range).
- Short-squeeze enrichment (Yahoo SI % float + optional FINRA daily short-volume ratio).
- Dilution detection (S-3 shelf, 424B ATM, reverse split, convertible, warrants) from EDGAR submissions.
- Catalyst detection from the shared daily 8-K index (FDA, M&A, material agreements, contracts, guidance).
- Weighted composite score, tiers, EXPLOSIVE flag, extension gate, shadow setup flag.
- CSV / JSON / Markdown reports per day; Tier A/B email via the notification service.
- Research tools: best-case backtest, stop/trailing strategy simulator + Optuna, point-in-time replay.

## Quick Start
```bash
# Daily run (defaults to yesterday's session)
.venv/Scripts/python.exe src/ml/pipeline/p17_penny_stocks/run_p17.py

# Specific date / tickers / force re-download
.venv/Scripts/python.exe src/ml/pipeline/p17_penny_stocks/run_p17.py --target-date 2026-11-05 --tickers CRCT,ANGI
.venv/Scripts/python.exe src/ml/pipeline/p17_penny_stocks/run_p17.py --force-refresh

# Point-in-time replay of every saved run with forward returns (evaluate a scoring change before shipping)
.venv/Scripts/python.exe -m src.ml.pipeline.p17_penny_stocks.replay --results-dir R:/apps/e-trading/results/p17_penny_stocks --out-dir results/p17_replay

# Best-case backtest and strategy simulator
.venv/Scripts/python.exe -m src.ml.pipeline.p17_penny_stocks.backtest --since 2026-05-15
.venv/Scripts/python.exe -m src.ml.pipeline.p17_penny_stocks.strategy_sim run --entry next_open
```

```python
from src.ml.pipeline.p17_penny_stocks.config import P17PipelineConfig
from src.ml.pipeline.p17_penny_stocks.p17_pipeline import P17Pipeline

result = P17Pipeline(P17PipelineConfig.create_default(), target_date="2026-11-05").run()
print(result["tier_a"], result["tier_b"], result["extended"], result["setup"])
```

Outputs go to `results/p17_penny_stocks/{date}/`: `{date}_candidates.csv`, `{date}_report.json`,
`{date}_report.md`, intermediate `00_`–`04_` files and `pipeline.log`.

## Integration
This module integrates with:
- `src.ml.pipeline.shared` - NASDAQ universe download, filter config base classes, EDGAR CIK map, FINRA TRF
- `src.data.downloader` / `src.data.cache` - Yahoo OHLCV + fundamentals, EDGAR, Fear & Greed; shared caches
- `src.data.db.services` - notification queue and user channel lookup for the Tier A/B email
- `src.data.pipeline.specs.p17_specs` - scheduler job definition (cron `0 8 * * 1-5`)
- `src.ml.pipeline.p19_penny_intraday` - reads the latest candidates CSV (Tier A/B/C + explosive) as its watchlist
- `src.ml.pipeline.signal_ledger` - harvests the candidates CSV (`p17_candidates`, `p17_tier_ab`, `p17_setup` streams)

## Configuration
All thresholds live in `config.py` dataclasses (`P17FilterConfig`, `P17TechnicalConfig`, `P17ShortSqueezeConfig`,
`P17ScoringConfig`, `P17CatalystConfig`, `P17AlertConfig`). The scheduler injects `--user-id` for the email.
FINRA credentials are optional (the short-squeeze agent degrades to Yahoo-only). `DATA_CACHE_DIR` controls the shared
OHLCV / fundamentals / EDGAR caches.

## Related Documentation
- [Requirements](docs/Requirements.md) - Technical requirements
- [Design](docs/Design.md) - Architecture and design
- [Tasks](docs/Tasks.md) - Implementation roadmap
- [Specification](docs/pipeline-specification.md) - Full pipeline spec, findings and calibration backlog
