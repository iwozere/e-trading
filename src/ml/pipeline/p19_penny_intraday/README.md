# P19 — Intraday Penny-Stock Spike Monitor

## Overview
Detects **explosive intraday moves in penny stocks while they are happening**
(minute/few-minute cadence) and emits a single, de-duplicated, human-readable
alert per name per day at breakout — the gap P17's daily batch screener cannot
fill. P19 is a **signal producer**; P17 is its watchlist source and conviction
prior (feed-forward, never merged).

## Status
**Phase 1.5 + Phase 3 — code complete (2026-08-18), not yet deployed.** v2
rework (`docs/pipeline-specification-v2.md`): a two-axis model replaces the
old severity score — a **structural integrity grade** (A–D, from EDGAR
filings) stays orthogonal to **momentum evidence** (RVOL/price-thrust).
Watchlist builder, shadow-mode logger, EOD backfill, the Layer 0 structural
profiler (`structural/`, now 26 of 28 N/P signals — N12 and P10 still
deferred, N9 above $75M float stays boolean per the spec's own fallback, see
`docs/design-v2.md` §9.10), momentum-tier classification,
T+10 outcome-label backfill, the intraday EDGAR filings poll, and sentiment
context attach are all built and tested (0 pyright/mypy, 148 P19 + 43 EDGAR
tests). **Not yet done**: applying `bin/scheduler/insert_p19_v2_schedules.sql`
on the Pi and confirming a real run against live EDGAR — see
`docs/tasks-v2.md`'s carry-over lists. **Phase 2 (alerting) is deliberately
sequenced after Phase 3 and calibration**, not because it's unbuilt by
oversight — thresholds are meant to be fit from shadow data, not hand-set
(decision #3). This remains shadow-mode only.

## Features (by phase)
- Phase 1 ✅: watchlist builder (P17 + gappers) + **shadow-mode logger** (no alerts).
- Phase 1.5 ✅ (code): Structural Integrity Profiler (`structural/`) — EDGAR
  XBRL/Form4/filings + yfinance splits → grade/dilution_urgency/insider_conviction,
  denormalised onto every shadow row; momentum-tier classification (log-only);
  outcome labels (`close_retention`, MAE/MFE, forward returns, dilution/split
  decay labels). See `docs/design-v2.md`.
- Phase 3 ✅ (code): N5/N6 (EFTS phrase search), N15 (recent IPO + micro
  float + FPI), N16 (auditor quality via the EX-23.1 consent exhibit), P8
  (13D/G presence, P18's daily cache), P9 (debt maturity), P11 (short
  interest, conditional on grade); intraday EDGAR filings poll
  (`filings_poll.py`, spec §9, log-only); sentiment context attach
  (`sentiment_cache.py`, throttled). See `docs/design-v2.md` §9.
- Phase 2: Disposition Engine (2-axis matrix, spec v2 §8) + Telegram alerts —
  sequenced after Phase 3/calibration, not skipped.
- Phase 4: Optuna threshold calibration on the shadow dataset; optional halt detection.

## Quick start (scaffold)
```bash
# REST feed probe (Finnhub/Polygon capability + rate limits)
python -m src.ml.pipeline.p19_penny_intraday.tools.latency_probe --rest

# IBKR Gateway probe — RUN ON THE PI during market hours (confirms delayed bars + volume)
python -m src.ml.pipeline.p19_penny_intraday.tools.latency_probe --ibkr

# CLI run modes
python src/ml/pipeline/p19_penny_intraday/run_p19.py build-watchlist
python src/ml/pipeline/p19_penny_intraday/run_p19.py profile-structural   # Layer 0, pre-market
python src/ml/pipeline/p19_penny_intraday/run_p19.py run-once --mode shadow
python src/ml/pipeline/p19_penny_intraday/run_p19.py eod-backfill
python src/ml/pipeline/p19_penny_intraday/run_p19.py label-backfill       # T+10
python src/ml/pipeline/p19_penny_intraday/run_p19.py filings-poll        # intraday, spec §9
```

## Integration
- `src.ml.pipeline.p17_penny_stocks` — daily watchlist + catalyst/squeeze/dilution agents
- `src.data.data_manager` / `src.data.downloader.*` — Finnhub/Polygon/yfinance feeds
- `src.common.sentiments` — social/news/FinBERT context, throttled via `sentiment_cache.py`
  (Phase 3, spec §10) — same providers every other pipeline uses
- `src.data.downloader.edgar_downloader` — fresh-8-K catalyst; XBRL companyfacts,
  submissions, the daily Form 4 cache, and (Phase 3) the daily 13D/G cache
  (P18's scan) all feed the Layer 0 structural profiler without new network
  surface; EFTS text/multi-CIK search (Phase 3) is the one genuinely new
  per-ticker network path Layer 0 adds
- `src.data.db.services` — NotificationService alert delivery (Phase 2)

## Feed decision (2026-06-28)
Free REST tiers lack real-time intraday **volume** (Finnhub `/quote` = price only;
Polygon `/aggs` = volume but ~15-min delayed + ~5 req/min — spec §13.1). So the
**primary feed is the IBKR Gateway** (delayed, free): its 5m bars **carry volume**,
giving real RVOL-so-far at ~15-min delay (acceptable). Binding limits become IBKR's
~100 market-data lines and historical pacing (spec §13.2). Connects to the same-Pi
paper Gateway (`raspberrypi:4002`).

## Related Documentation
- [Pipeline Specification v2](docs/pipeline-specification-v2.md) — current design, start here
  ([StructuralSignals reference](docs/StructuralSignals.md) for every N/P signal)
- [Requirements v2](docs/requirements-v2.md) · [Design v2](docs/design-v2.md) · [Tasks v2](docs/tasks-v2.md)
- v1 (superseded, Phase 0/1 history): [Pipeline Specification](docs/pipeline-specification.md) ·
  [Requirements](docs/Requirements.md) · [Design](docs/Design.md) · [Tasks](docs/Tasks.md)
- [Brainstorming notes](docs/brainstorming1.md) — original free-data research
