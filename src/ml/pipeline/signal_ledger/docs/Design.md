# Design

## Purpose
The pipelines produce picks every day, but none of them were tracked after the fact. The 2026-10-21 review showed that only
P05 had a measurable edge, P17 was clearly negative and the rest were flat. The ledger makes that check continuous and
automatic: every pick is recorded once, measured when its horizons mature, and summarized per stream. The user decided that a
recorded, measured buy list is enough for now — no broker orders (IBKR paper execution can be added later on top of this table).

## Architecture

### High-Level Architecture
```
results/<pipeline>/YYYY-MM-DD/*  --harvesters-->  signal_ledger (pending)
                                                   |
Yahoo daily bars --> PriceCache (parquet) ---------+--> evaluator --> signal_ledger (complete / no_data, returns, data_flag)
                                                   |
                                                   +--> portfolio.simulate_book --> signal_book_nav
                                                   +--> scorecard.build_scorecard --> scorecard.json, /api/signals/scorecard
```

### Component Design
- `harvesters.py` — one function per pipeline output; returns `SignalRecord`s. A failing harvester is logged and skipped.
- `prices.py` — `PriceCache`: wide open/close panel. A ticker's columns are always replaced from one fetch, because adjusted
  history changes after every split/dividend and stitched fetches would give wrong returns.
- `evaluator.py` — entry/exit logic and the price-jump guard.
- `scorecard.py` — per-stream statistics; sub-streams (e.g. `p17_tier_ab`) are predicates on a parent source.
- `portfolio.py` — daily paper-book simulation.
- `jobs/run_signal_ledger.py` — orchestration; prints `__SCHEDULER_RESULT__` like the other job scripts.

## Data Flow
- `signal_date` = the results folder date. Entry = open of the first session strictly after it. For P05 (runs before the US
  open of its folder date) this is one day conservative; for P06/P10/P17 (folder = previous session) it is the next open.
- P19 alerts fire intraday; the ledger still enters at the next open ("buy the day after the alert").
- Outcomes freeze once all horizons are known (`status = complete`); `no_data` after 10 days without an entry price.
- Benchmarks are measured over the identical window; excess = side × (ticker − benchmark) − round-trip cost.

## Design Decisions
- **Harvest, don't instrument.** Reading existing outputs needs no change in six pipelines and backfills history for free.
- **Next-open entry, close exit.** Avoids look-ahead (P17's own backtest used the later peak `max_high`).
- **Cohort Newey-West t.** Same-day picks are correlated and 20-day windows overlap; per-pick t-stats would overstate significance.
- **Halts/delistings exit at the last traded close** instead of dropping out (dropping losers biases results up).
- **5× jump guard.** Penny-stock reverse splits that Yahoo never adjusted produced +5,000 % "returns" (IPDN, GMEX); 48 such
  jumps in 28 tickers over the archive. Flagged rows stay visible but do not count.
- **Long-only book.** Short recommendations (rare P05 `bias=short`) are scored but not traded in the book.
- **Costs per stream.** 10 bps round trip for liquid names, 30 bps for P06/P10 small caps, 100 bps for P17/P19 penny stocks.
- **Book NAV fully rebuilt nightly.** Cheap (seconds) and keeps it consistent with any re-harvested signal.

## Integration Patterns
- DB access only through `SignalLedgerService` (UoW per call); the API recomputes the scorecard from ledger rows on request,
  so changing the scorecard logic needs no data migration.
- Scheduler: `Signal Ledger Nightly`, `0 6 * * 2-6` UTC, after the previous US close is final.
