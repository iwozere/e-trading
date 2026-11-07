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
                                                   +--> portfolio.simulate_book x exits.POLICIES --> signal_book_nav,
                                                   |        signal_book_trades, signal_book_stats (+ simulate_uncapped stats)
                                                   +--> scorecard.build_scorecard --> scorecard.json, /api/signals/scorecard
```

### Component Design
- `harvesters.py` — one function per pipeline output; returns `SignalRecord`s. A failing harvester is logged and skipped.
- `prices.py` — `PriceCache`: wide OHLC panel (old caches without high/low fall back to the open/close range). A ticker's columns are always replaced from one fetch, because adjusted
  history changes after every split/dividend and stitched fetches would give wrong returns.
- `evaluator.py` — entry/exit logic, price-jump guard and MAE/MFE, on numpy arrays built once per run (`PricePanels`).
- `exits.py` — `ExitPolicy` definitions and which streams get a `native` exit.
- `scorecard.py` — per-stream statistics; sub-streams (e.g. `p17_tier_ab`) are predicates on a parent source.
- `portfolio.py` — daily paper-book simulation under one exit policy (`simulate_book` -> `BookResult` with NAV and trades;
  `simulate_uncapped` trades every signal); `summarize_book` / `summarize_trades`.
- `jobs/run_signal_ledger.py` — orchestration; prints `__SCHEDULER_RESULT__` like the other job scripts.

## Data Flow
- `signal_date` = the results folder date. Entry = open of the first session strictly after it. For P05 (runs before the US
  open of its folder date) this is one day conservative; for P06/P10/P17 (folder = previous session) it is the next open.
- P19 alerts fire intraday; the ledger still enters at the next open ("buy the day after the alert").
- P22 republishes the same ~500-name watchlist daily, so only **tier entries** are signals: a ticker in tiers 1–3 that was
  not tiered in the previous readable watchlist (re-entry after dropping out counts again). The harvester always scans the
  whole (small) P22 archive to know the previous set; `since` only filters the output. Book horizon 250 sessions.
- P15 is not harvested: it is a data-download bundle (prices, options, GDELT, EDGAR, FRED) with no buy signals.
- Outcomes freeze once all horizons (up to 250 sessions) are known (`status = complete`); `no_data` after 10 days without
  an entry price. Pending signals are re-evaluated nightly.
- Every ticker with a pending signal or a signal in the last `ACTIVE_PRICE_DAYS` is re-fetched nightly. Before 2026-11-07
  only pending tickers were, so book positions held past 20 sessions (P24, P22) were marked at a stale price.
- Native exits: harvesters put the pipeline's exit into `meta["exit_signal_date"]` (P22: first watchlist without the
  ticker in tiers 1-3; P24: its paper state). Those streams re-emit their whole archive nightly so the upsert refreshes meta.
  Republishing streams instead exit after `DECAY_SESSIONS` sessions without a new signal for the ticker.
- Order inside a book session: open exits (native/decay, gapped stops) -> entries -> intraday stop/target (both touched =
  stop) -> time exit at the close. Stops fill at the stop level, or at the open if gapped through.
- Benchmarks are measured over the identical window; excess = side × (ticker − benchmark) − round-trip cost.

## Design Decisions
- **Harvest, don't instrument.** Reading existing outputs needs no change in six pipelines and backfills history for free.
- **Next-open entry, close exit.** Avoids look-ahead (P17's own backtest used the later peak `max_high`).
- **Cohort Newey-West t.** Same-day picks are correlated and 20-day windows overlap; per-pick t-stats would overstate significance.
- **Halts/delistings exit at the last traded close** instead of dropping out (dropping losers biases results up).
- **5× jump guard.** Penny-stock reverse splits that Yahoo never adjusted produced +5,000 % "returns" (IPDN, GMEX); 48 such
  jumps in 28 tickers over the archive. Flagged rows stay visible but do not count.
- **Long-only book.** Short recommendations (rare P05 `bias=short`) are scored but not traded in the book.
- **Costs per stream.** 10 bps round trip for liquid names, 30 bps for P06/P10 small caps, 60 bps for P22/P24,
  100 bps for P17/P19 penny stocks.
- **Book NAV fully rebuilt nightly.** Keeps it consistent with any re-harvested signal (~3 min on the dev machine for
  27k signals x ~8 policies x capped + uncapped).
- **Exit policies in shadow books, not in IBKR.** One account can run one exit per signal; the simulation runs all of them
  on the same entries. MAE/MFE per trade show how deep winners dip before working, which is the evidence for stop widths.
- **Capped and uncapped.** With a 40-position cap, long holds fill all slots within days and later signals are never taken,
  so the capped book compares capacity rather than exits. Exit rules are compared on the uncapped run (`all_signals`);
  the capped book stays the realistic portfolio view.

## Integration Patterns
- DB access only through `SignalLedgerService` (UoW per call); the API recomputes the scorecard from ledger rows on request,
  so changing the scorecard logic needs no data migration. The policy comparison is read from `signal_book_stats` (written
  nightly) because recomputing it means replaying every book.
- Scheduler: `Signal Ledger Nightly`, `0 6 * * 2-6` UTC, after the previous US close is final.
