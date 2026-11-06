# Design

## Purpose
Find small-float US penny stocks with the ingredients of an explosive move (volume, structure, catalyst, short-squeeze
fuel) while filtering out the usual penny-stock traps (dilution, insolvency, illiquidity), and deliver the actionable
names before the US open.

## Architecture

### High-Level Architecture
`P17Pipeline` (`p17_pipeline.py`) runs agents in sequence, each wrapped in `_run_job()` so one failing stage degrades
the run instead of aborting it:

```
UniverseAgent ─► MarketAgent ─► TechnicalAgent ─► ShortSqueezeAgent ─► DilutionAgent ─► CatalystAgent
   (stage 1)       (stage 2)       (stage 3)          (stage 4)           (stage 5)       (stage 5.5)
                                                                                              │
         NotificationAgent ◄── ReportingAgent ◄── ScoringAgent ◄──────────────────────────────┘
            (stage 8)            (stage 7)          (stage 6)
```

### Component Design
- **UniverseAgent** - symbol universe → Yahoo fundamentals snapshot (1-day cache, checkpointed) → hard filters.
- **MarketAgent** - 90-day daily OHLCV (shared unified cache, batch download for misses) and fundamentals;
  `apply_survival_filter` drops names failing cash-runway / debt-to-cash hard stops.
- **TechnicalAgent** - pure pandas/numpy indicators per candidate, fault-isolated per ticker.
- **ShortSqueezeAgent** - SI % float, days to cover, optional FINRA short-volume ratio; `compute_score()`.
- **DilutionAgent** - EDGAR filing history → summed hard penalty.
- **CatalystAgent** - 8-K items + description keywords → recency-weighted `catalyst_score`.
- **ScoringAgent** - sub-scores normalised to 0–100, weighted sum, dilution deduction, tier, EXPLOSIVE flag,
  extension gate, setup score.
- **ReportingAgent** - CSV / JSON / Markdown artefacts.
- **NotificationAgent** - queues one Tier A/B email via `NotificationService`; never raises.
- **Research tools** - `backtest.py` (sell-at-peak upper bound), `strategy_sim.py` (stop/trail simulator + Optuna),
  `replay.py` (point-in-time re-scoring + forward returns).

## Data Flow
- Input: symbol universe, Yahoo snapshot/OHLCV, EDGAR, FINRA, Fear & Greed.
- Internal: a list of `Candidate` dataclasses (`models/candidate.py`) enriched in place by each agent.
- Output: `results/p17_penny_stocks/{date}/` artefacts + an email; `__SCHEDULER_RESULT__` JSON on stdout carries
  `tier_a_count`, `explosive_count`, `extended_count`, `setup_count` for scheduler notification rules.

## Design Decisions
- **Agents mutate shared `Candidate` objects** - simple, no copying; each agent owns a disjoint set of fields.
- **Hard filters before scoring, penalties after** - dilution is a deduction, not a weight, so a strong setup cannot
  outvote a toxic financing.
- **Extension gate (2026-11-06)** - prod history showed A/B signals fired at a median +43% 5-day run-up and lost ~9%
  over 10 sessions; the mandatory EXPLOSIVE conditions describe the breakout day itself. Names up > 30% in 5 days or
  > 3 ATR above SMA20 are capped at Tier C (still on the P19 watchlist) and never EXPLOSIVE. Replay: fwd10 median
  −8.0% → +0.4% (n 59 → 20), consistent in both halves.
- **Setup score in shadow mode** - the "coiled, not yet moved" score showed no edge in replay (sign flipped between
  halves; only 7/59 jumps were preceded by a flag), so it is reported and measured by the signal ledger but not
  emailed. Thresholds were deliberately not tuned on the replay sample.
- **Replay compares like with like** - `replay_legacy` (gate off) vs `replay_ab` (gate on) use the same fresh OHLCV;
  `prod_ab` differs because prod scored some rows on stale cached bars.
- **Caches shared across pipelines** - OHLCV and fundamentals caches are reused by P19 and others.

## Integration Patterns
- Scheduler: `src/data/pipeline/specs/p17_specs.py` (cron `0 8 * * 1-5`, timeout 7200 s, rules on
  `tier_a_count` / `explosive_count`).
- Downstream readers use the candidates CSV only (P19 watchlist, signal ledger); column additions are backward
  compatible, renames are not.
- Notifications are fire-and-forget through the DB queue; failures are logged and swallowed.
