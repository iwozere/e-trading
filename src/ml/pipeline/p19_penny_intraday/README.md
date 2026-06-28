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
# REST feed probe (Finnhub/Polygon capability + rate limits)
python -m src.ml.pipeline.p19_penny_intraday.tools.latency_probe --rest

# IBKR Gateway probe — RUN ON THE PI during market hours (confirms delayed bars + volume)
python -m src.ml.pipeline.p19_penny_intraday.tools.latency_probe --ibkr

# CLI run modes (stubs until Phase 1/2)
python src/ml/pipeline/p19_penny_intraday/run_p19.py run-once --mode shadow
```

## Integration
- `src.ml.pipeline.p17_penny_stocks` — daily watchlist + catalyst/squeeze/dilution agents
- `src.data.data_manager` / `src.data.downloader.*` — Finnhub/Polygon/yfinance feeds
- `src.common.sentiments` — social/news/FinBERT context
- `src.data.downloader.edgar_downloader` — fresh-8-K catalyst
- `src.data.db.services` — NotificationService alert delivery

## Feed decision (2026-06-28)
Free REST tiers lack real-time intraday **volume** (Finnhub `/quote` = price only;
Polygon `/aggs` = volume but ~15-min delayed + ~5 req/min — spec §13.1). So the
**primary feed is the IBKR Gateway** (delayed, free): its 5m bars **carry volume**,
giving real RVOL-so-far at ~15-min delay (acceptable). Binding limits become IBKR's
~100 market-data lines and historical pacing (spec §13.2). Connects to the same-Pi
paper Gateway (`raspberrypi:4002`).

## Related Documentation
- [Pipeline Specification](docs/pipeline-specification.md) — full design (start here)
- [Requirements](docs/Requirements.md) · [Design](docs/Design.md) · [Tasks](docs/Tasks.md)
- [Brainstorming notes](docs/brainstorming1.md) — original free-data research
