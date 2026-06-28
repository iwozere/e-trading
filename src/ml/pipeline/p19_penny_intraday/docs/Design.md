# Design — P19 Intraday Penny-Stock Monitor

> The authoritative design is **[pipeline-specification.md](pipeline-specification.md)**.
> This file summarises the architecture and the key decisions; see the spec for detail.

## Purpose
Catch same-session penny-stock spikes that the daily P17 batch screener structurally
misses, and emit one de-duplicated, human-readable alert at breakout.

## Architecture (summary)
Daily **Watchlist Builder** (P17 output + pre-market gappers < $5, hard-filtered,
capped) → intraday **stateless `run-once`** loop on a short cron: **Intraday Feed**
→ **Trigger Engine** (stateful, dedup) → **Enrichment** (P17 agents) → **Alert
Manager**. A **Shadow Logger** records every poll for every name to build the
calibration dataset. Full diagram: spec §4.

## Key decisions
1. **Separate pipeline, feed-forward from P17** — opposite cadence/shape; P17 ranks,
   P19 watches. P17 score is alert *context*, never a gate (spec §3).
2. **Finnhub real-time price trigger; volume is delayed context** — measured: no free
   tier gives real-time intraday volume (spec §13.1).
3. **Shadow-mode first** — accumulate data, then calibrate thresholds with Optuna
   before alerting (spec §15–16).
4. **Sentiment is context only**, not a trigger (spec §10).
5. **Stateless `run-once` on intraday cron**, not a daemon — crash-safe via state file.

## Component design
- `config.py` — `P19Config` and sub-configs (filters, feed, triggers, alerts).
- `models/intraday_signal.py` — `IntradaySignal` (detection + shadow-log row).
- `run_p19.py` — CLI run modes (`build-watchlist` / `run-once` / `eod-backfill`).
- `tools/latency_probe.py` — free-tier feed capability / rate probe.
- *(later)* watchlist builder, feed, trigger engine, state store, alert manager,
  shadow logger.

## Error handling
- Provider failures: DataManager failover + per-poll tolerance (never abort the loop).
- Crash safety: state persisted to `alerted.json`; `run-once` is idempotent per poll.
- Alerting failures logged, never block the loop (mirrors P17 Stage-8 / P15 jobs).
