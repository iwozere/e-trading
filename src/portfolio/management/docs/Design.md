# Design

## Purpose
Remind the user of two things together, only when it matters: a held
ticker's earnings date is imminent, and whether that position currently has
a live protective stop order. See `docs/brainstorm.md` for the full scoping
discussion (including the Flex-Query-can't-report-open-orders finding and
what was deliberately cut from the original, broader proposal).

## Architecture

### High-Level Architecture
```
APScheduler (*/15 12-22 * * 1-5, UTC)
   -> runner.run_once(cfg)
        |
        +--> flex_downloader.download_open_positions_xml  (reused from pnl_alert)
        +--> ibkr_xml_loader.load_ibkr_xml                (reused from pnl_alert)
        +--> position_aggregator.merge_holdings           (reused from pnl_alert)
        |       -> {ticker: quantity}
        |
        +--> earnings_source.EarningsSource
        |       -> wraps p05_ai_selector.EarningsCalendar
        |       -> [EarningsEvent(ticker, earnings_date, session)]
        |
        +--> earnings_window.resolve_anchor_utc + matched_trigger  (pure)
        |       -> which tickers have a T-1day/T-1hour trigger firing *now*
        |
        +--> open_orders.fetch_protective_qty                (live Gateway, read-only)
        |       -> only called if >=1 ticker triggered this run
        |       -> {ticker: working protective-order qty}
        |
        +--> coverage.evaluate                                (pure)
        |       -> CoverageRow per triggered ticker
        |
        +--> notifier.send_reminder  (NotificationServiceClient)
```

### Component Design
- **config.py** — `ManagementConfig` dataclass + `load_config(path)` YAML
  loader. IBKR live host/port/clientId fall back to `IBKR_HOST` / `IBKR_PORT`
  / `IBKR_LIVE_STOP_GUARD_CLIENT_ID`.
- **earnings_window.py** — pure: `EarningsEvent`, `resolve_anchor_utc`,
  `matched_trigger`. No I/O, the core trigger-timing logic.
- **earnings_source.py** — thin wrapper around `EarningsCalendar` (P05),
  mapping its `{ticker: date}` result into `EarningsEvent`s.
- **coverage.py** — pure: `classify` / `evaluate`, position qty vs. working
  protective-order qty → `covered` / `partially_covered` / `uncovered`.
- **open_orders.py** — `IBKROpenOrdersFeed` (connect/fetch/disconnect against
  the live Gateway, read-only) + `fetch_protective_qty` (the one function
  that bundles all three into a single call, so the whole `ib_insync`
  session lifecycle runs on one thread — see its docstring).
- **notifier.py** — `TriggeredReminder`, plain-text + HTML formatting,
  `send_reminder` via `NotificationServiceClient`.
- **runner.py** — orchestrates the steps above; usable from the CLI, tests,
  and the scheduler. Accepts an injectable `now` so trigger-matching tests
  are deterministic instead of racing the real clock.
- **cli.py** / **__main__.py** — `python -m src.portfolio.management` with
  `--dry-run`, `--as-of-date`, `--config` flags.
- **seed_schedule.py** — idempotent inserter/updater of the row in
  `job_schedules`.

## Data Flow
- Input: IBKR holdings (symbol, quantity) from the Flex Query XML export;
  earnings dates for those tickers from FMP; live working protective orders
  for tickers whose trigger is currently in-window.
- Output: one notification message per run that has >=1 triggered ticker
  (never one message per ticker — see "No dedup" below for why batching
  doesn't conflict with that), plus a small `RunSummary` dict (stored by the
  scheduler as run-result JSON).
- No new persisted state for the MVP — every run recomputes everything from
  live/near-live sources. See "Phase 2 (optional)" in `Tasks.md` for an
  audit-log table if that's wanted later.

## Design Decisions

### Earnings-triggered only, no baseline poll
Confirmed by the user: stop coverage is checked *only* at T-1 day and T-1
hour before each held ticker's own earnings date — not on an independent
always-on cadence. A ticker with no earnings in the lookahead window simply
isn't checked. This keeps the design to one job instead of two, at the cost
of not catching a missing stop on a quiet ticker with no upcoming earnings —
an accepted tradeoff, not an oversight.

### No dedup — alert every time the condition is met
Per the user's explicit choice, same rationale as `pnl_alert`'s original "no
dedup" decision: simple and loud for now, revisit if it proves too noisy.
In practice this rarely means literal repeat spam, because:
- The T-1day and T-1hour windows are narrow (`trigger_window_minutes`,
  default 15) relative to the 15-minute poll cadence, so each trigger
  typically matches on exactly one poll.
- The two triggers are themselves ~23 hours apart.

### Window-match triggers, not literal per-ticker dynamic scheduling
The original brainstorm gestured at "dynamic reminders" (bespoke per-ticker
APScheduler jobs). This does something simpler and closer to this
codebase's existing convention (P20 Kestrel's T-10/T-3 countdown: a single
periodic job that checks a relative-time condition on every run): one fixed
cron job polls every 15 minutes, and each run asks "is `now` within
`trigger_window_minutes` of this ticker's T-1day or T-1hour mark?" No
per-ticker scheduler entries, no state to keep in sync as calendar data
changes.

### Holdings: XML only, no live top-up
`pnl_alert` tops up its XML-derived holdings with same-day live positions
from the **paper** Gateway (`IBKR_PAPER_PORT`) — irrelevant here, since this
module only ever touches the live account. Adding a second *live* connection
just for same-day holdings freshness wasn't judged worth the complexity for
a reminder that only fires once around each earnings date; the Flex Query
XML (refreshed at the top of every run, same as `pnl_alert`) is precise
enough. `Holding.source` from `position_aggregator` is unused here as a
result — only `{ticker: quantity}} is extracted.

### Live Gateway connection is best-effort, but always reported
If the live Gateway is unreachable when a trigger fires, the run doesn't
abort — it still sends the reminder (with coverage defaulting to
`uncovered`, since 0 protective qty was found) and records
`"live_ibkr_unreachable"` in `RunSummary.errors`. Silently skipping the
notification when IBKR is down would be exactly wrong for a coverage
reminder — the day the Gateway happens to be down is not a day to go quiet.

### Read-only by construction, not by convention
`IBKROpenOrdersFeed.connect()` always passes `readonly=True` to `ib.connect()`
— this module cannot place, modify, or cancel an order even if some future
change called it wrong. This is deliberately stronger than "the code
happens not to call placeOrder" for a module whose entire premise is "don't
touch orders, just report on them."

### Reuse over rebuild (see brainstorm.md for the full comparison)
- Holdings: `pnl_alert`'s Flex Query XML pipeline, not a second parser.
- Earnings dates: P05's `EarningsCalendar` (already FMP-backed, already
  parameterized by an arbitrary ticker list), not P20 Kestrel's
  watchlist-bound calendar and not a new SEC EDGAR 8-K pipeline.
- Notification delivery, scheduler wiring, DB engine: all the existing
  shared plumbing, no new infrastructure.

## Integration Patterns
- Scheduler integration is a second branch inside the existing
  `_execute_portfolio_job` handler: `target == "portfolio.management"`
  dispatches to this module's `runner.run_once`, alongside the existing
  `"portfolio.pnl_alert"` branch.
- Notifications go through the same `NotificationServiceClient` /
  Telegram+SMTP plumbing the rest of the app uses.
- `job_type = "alert"` is reused (same reasoning as `pnl_alert`: the
  `job_schedules` table has a hard DB check constraint on `job_type`, and
  `target` is already the semantically-correct dispatch key).

## Error Handling
- Flex Query XML unreadable/missing: logged, added to `RunSummary.errors`;
  run exits early with `holdings_count = 0` (no earnings lookup performed).
- Earnings calendar lookup failure: logged, added to `RunSummary.errors`,
  run exits early (no triggers can be evaluated without earnings dates).
- No ticker's trigger is in-window this run: logged at INFO, run exits
  without opening an IBKR connection at all.
- Live IBKR unreachable when a trigger *is* in-window: WARNING, added to
  `RunSummary.errors`, reminder still sent with coverage defaulting to
  `uncovered` (see "Live Gateway connection is best-effort" above).
- Per-trade parse failure inside `reqAllOpenOrders()`'s result: logged at
  DEBUG, that trade skipped, run continues.
