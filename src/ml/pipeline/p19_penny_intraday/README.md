# P19 — Intraday Penny-Stock Spike Monitor

## Overview
Detects **explosive intraday moves in penny stocks while they are happening**
(minute/few-minute cadence) and emits a single, de-duplicated, human-readable
alert per name per day at breakout — the gap P17's daily batch screener cannot
fill. P19 is a **signal producer**; P17 is its watchlist source and conviction
prior (feed-forward, never merged).

## Status
**Phase 0 — scaffold.** Spec, config, model, CLI stub, and a feed-latency probe
exist; the live loop, triggers, and alerting are not yet implemented.

## Features (planned, by phase)
- Phase 1: watchlist builder (P17 + gappers) + **shadow-mode logger** (no alerts).
- Phase 2: Finnhub real-time price trigger + delayed-volume context + Telegram alerts.
- Phase 3: catalyst (fresh 8-K) / short-squeeze / dilution enrichment via P17 agents.
- Phase 4: Optuna threshold calibration on the shadow dataset; optional halt detection.

## Quick start (scaffold)
```bash
# measure free-tier feed capability / rate limits (run during market hours for latency)
python -m src.ml.pipeline.p19_penny_intraday.tools.latency_probe

# CLI run modes (stubs until Phase 1/2)
python src/ml/pipeline/p19_penny_intraday/run_p19.py run-once --mode shadow
```

## Integration
- `src.ml.pipeline.p17_penny_stocks` — daily watchlist + catalyst/squeeze/dilution agents
- `src.data.data_manager` / `src.data.downloader.*` — Finnhub/Polygon/yfinance feeds
- `src.common.sentiments` — social/news/FinBERT context
- `src.data.downloader.edgar_downloader` — fresh-8-K catalyst
- `src.data.db.services` — NotificationService alert delivery

## Key constraint (measured 2026-06-28)
No free tier provides real-time intraday **volume**: Finnhub `/quote` gives
real-time **price** (no candles), Polygon `/aggs` gives volume but ~15-min delayed
and ~5 req/min. Trigger on price; treat RVOL as delayed context. See spec §13.1.

## Related Documentation
- [Pipeline Specification](docs/pipeline-specification.md) — full design (start here)
- [Requirements](docs/Requirements.md) · [Design](docs/Design.md) · [Tasks](docs/Tasks.md)
- [Brainstorming notes](docs/brainstorming1.md) — original free-data research
