# Design

## Purpose

P21 exists to answer whether a disciplined, low-cost DIY 20-stock momentum portfolio adds anything over
simply buying QDVA (MSCI USA Momentum ETF) and applying the same regime overlay to it — see
`pipeline-specification.md` §0 and §9 for the full framing. This document is deliberately thin: it condenses
the spec and points back into it for anything more detailed than a paragraph, per the plan's own convention
(`docs/implementation-plan.md` §6.1).

## Architecture

Three scheduled jobs, each idempotent and safe to re-run (spec §1, §3):

| Job | Trigger (self-guarded, daily cron) | Reads | Writes |
|---|---|---|---|
| `monthly_rebalance` | Last NYSE trading day of month | universe, prices, fundamentals, exclusions, earnings | `universe.json`, `signals.json`, `targets.json`, `_state/regime_history.json` |
| `monthly_execute` | First NYSE trading day of month | prior `targets.json`, opens | `positions.json`, `report.md`, `_state/ledger.jsonl`, `_state/current_positions.json` |
| `daily_mark` | Every NYSE trading day | closes, `_state/current_positions.json` | `daily_mark.json`, `_state/nav_daily.csv` |

Every job's `run()` is a thin no-op guard (idempotency + calendar check) wrapping `_run_*()`, the actual
orchestration — separated so tests can call `_run_*()` directly with fixed dates while `run()` handles the
"is today the right day" self-guarding (Open Decision #2 of `docs/implementation-plan.md`, resolved: the
scheduler's cron has no native "last/first trading day of month" concept, so all three jobs run on a daily
cron and no-op internally except on their actual trigger day).

## Data Flow

```
fetch (universe, prices, fundamentals, exclusions)
  -> validate (§13 gates; ABORT halts the run, leaves state untouched)
  -> signal (§4: 12-1 month risk-adjusted momentum)
  -> filter (§5: F1-F6, cheap-first)
  -> rank -> select (§6: hysteresis + sector cap)
  -> size (§7: inverse-vol, capped at 1% of NAV)
  -> regime (§8: bear/high-vol overlay, applied to sizing)
  -> write TARGET
     [month boundary]
  -> read TARGET -> fetch opens -> simulate fills (§10.1) -> write POSITIONS + LEDGER
  -> update 5 tracks (§9) -> generate REPORT (§12)
```

`strategy/`, `execution/`, `quality/`, `results/` are written once and imported by both the live pipeline and
(eventually) the backtest harness under `backtest/p21_momentum/` — only the outer loop differs (real time vs.
a frozen historical panel). See `docs/implementation-plan.md` §1, §8.

## Design Decisions

- **Adjusted-close-only bookkeeping (spec §10.1) — provisional.** `YahooDataDownloader.get_ohlcv_batch()`
  returns only a split/dividend-adjusted `close`/`open`. Rather than fork the downloader to add a raw-price +
  `actions` feed, the entire position/ledger model works off the one adjusted series: no manual split
  adjustment, no separate dividend crediting, `shares` is a notional (`dollars / adjusted_price`) rather than
  a broker-reconcilable quantity. Explicitly marked for revisit if Phase 0/1 shows this distorts the
  audit trail — see the spec's own blockquote at the top of §10.1 before changing this.
- **PaperTradingMixin rejected.** `src/trading/broker/paper_trading_mixin.py` was considered and rejected for
  execution simulation: it's async, uses `random.random()` for fill probabilities (violates the §14.9 B10
  bit-identical-rerun requirement), and has a flat commission model where P21 needs IBKR's tiered per-share +
  per-order-minimum model. `execution/fills.py`'s `simulate_fill()` is a small deterministic function instead.
- **F6 (earnings blackout) evaluated post-selection, not pre-ranking.** The literal spec order runs all six
  filters before ranking. This implementation runs F1-F5 before ranking (as specified) but defers F6 until
  after `select_portfolio()` has determined which survivors are actually *new* entries — F6 only matters for
  new entries (spec: "holdings unaffected"), so this means fewer earnings-calendar network calls, at the cost
  of not backfilling a replacement when a new entry fails F6 (the position count simply comes in lower, which
  is exactly the existing `WARN_UNDERFILLED` path). See `jobs/run_monthly_rebalance.py`'s module docstring.
- **Dependency-injected paths, not hard-coded module constants.** Every job's `run()` accepts optional
  `results_dir` / `state_dir` / path overrides (defaulting to the real `config.py` paths). This exists
  specifically so `tests/test_integration_monthly_cycle.py` can exercise real file I/O against a tmp directory
  — including the §14.9 B10 determinism check (two identical `monthly_rebalance` runs must produce
  byte-identical `targets.json`) — without mocking the read/write layer away entirely.
- **Dataclasses, not pydantic, for `schemas.py`.** Matches this repo's convention: pydantic is reserved for
  API-boundary schemas; plain dataclasses (`slots=True`, `to_dict()`/`from_dict()`) are used for file-I/O-only
  models.
- **Known gap in `monthly_execute`'s report generation.** Full multi-track (B/C/D/E) NAV compounding and the
  §12.5/§12.6 realized-metrics rollups need a walk over the complete `_state/nav_daily.csv` and
  `_state/ledger.jsonl` history — `strategy/tracks.py` provides the building blocks
  (`apply_ter_drag`, `build_nav_series`, `compute_attribution`) but `jobs/run_monthly_execute.py` does not yet
  wire them into a full history aggregation; the report currently renders `0.0` placeholders for those fields,
  explicitly, not silently. Track A's own state (positions, ledger, cash, NAV) is real. This is flagged in the
  module's own docstring, not hidden.

## Integration Patterns

- All price/fundamentals access goes through `data/prices.py`, which wraps `YahooDataDownloader` — no module
  in this package calls `yfinance` directly except `data/earnings.py` (spec §2's one documented exception).
- `quality/gates.py`'s `PipelineAbort` is the only exception a job's `main()` catches specially: it logs,
  sends an admin alert via `NotificationServiceClient`, and returns a structured `{"aborted": True, ...}`
  result rather than raising out of `main()` — the scheduler still sees a clean subprocess exit with a
  `__SCHEDULER_RESULT__` payload it can inspect.
- Backtest harness (`backtest/p21_momentum/`, not yet built — see `docs/implementation-plan.md` §8) is
  deliberately outside `results/p21_momentum/`: it is a one-time frozen research snapshot, never regenerated
  by the three live jobs, and must not be confused with live run history.
