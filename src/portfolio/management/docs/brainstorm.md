# Portfolio Management — Stop-Loss & Earnings Reminder

## Goal

Strict, hands-off *visibility* into one thing: **every open IBKR position must
have a live protective stop order.** The user sets stop-loss orders manually;
this system's job is to notice — immediately and reliably — when a position
doesn't have one, and to surface upcoming earnings dates for held tickers so
that a missing/removed stop ahead of an earnings gap isn't a surprise.

This is a **reminder system, not an execution system**. It never places,
modifies, or cancels an order. That's a deliberate scope cut from the earlier
draft of this document (see "Rejected from the original brainstorm" below).

---

## Key finding: IBKR Flex Query cannot report open orders

Activity Flex Query sections cover account *activity* — Open Positions,
Trade Confirms, Cash Report, Corporate Actions, Dividend/Interest Accruals,
Transfers — all point-in-time or historical. There is no "Open Orders"
section; Flex statements are EOD/batch reports, not a live order-book
snapshot. Working orders (including manually-placed stops) are only visible
through a **live Gateway/TWS API session**.

`ib_insync`/`ib_async` exposes this as `ib.reqAllOpenOrders()` (blocking,
queries IBKR fresh) or the faster local-cache reads `ib.openOrders()` /
`ib.openTrades()`. This codebase already maintains exactly this kind of live
session elsewhere (`src/ml/pipeline/p19_penny_intraday/intraday_feed.py`,
`IBKRBroker` in `src/trading/broker/ibkr_broker.py`), so this is a small
addition, not a new integration.

**Setup requirement — Master API Client ID:** by default, the API only sees
orders placed by *that same API client*. A stop-loss placed manually in the
TWS/Gateway GUI is invisible to `reqAllOpenOrders()` unless the connecting
client ID is configured as the account's **Master API client ID**
(Gateway/TWS → Configure → API → Settings → "Master API client ID"). Without
this, the guard will report every manually-set stop as missing. This must be
set once on the **live** Gateway (see "Decisions" below), and documented in
`Requirements.md` as a precondition, not assumed.

**Live account only.** There's no point checking stop coverage on the paper
account — real positions and real stops only exist on live. This connects to
the **live** Gateway (port `4001`, vs. `4002` for paper — confirmed against
`src/trading/broker/ibkr_utils.py`'s `trading_mode` port mapping), read-only,
with its own dedicated `clientId` distinct from `IBKR_CLIENT_ID` /
`IBKR_PAPER_CLIENT_ID` / P19's `19`/`20`. This is the opposite direction from
the last two fixes on this branch (`bf08b06`, `ee89c9c`), which moved
`pnl_alert`/`IBKRDownloader` *onto* the paper port — worth double-checking at
review time that this module's config isn't accidentally copy-pasted from
one of those.

---

## Reuse plan

Everything below already exists in this codebase and should be imported, not
rebuilt:

| Need | Reuse |
|---|---|
| Current holdings (qty, avg cost, symbol-collision-safe) | `src.portfolio.pnl_alert.ibkr_xml_loader.load_ibkr_xml` + `position_aggregator` (same-day XML refreshed by `flex_downloader.py`), optionally topped up live via `IBKRBroker.get_positions()` — same pattern `pnl_alert` already uses |
| Live IBKR session (for open orders) | Same connect pattern as `p19_penny_intraday/intraday_feed.py`, but pointed at the **live** Gateway (port `4001`, read-only), not paper — retry-on-connect logic reused, target port/account is not. Needs its own `clientId` (next free one after 19/20) |
| Earnings dates for arbitrary held tickers | `src.ml.pipeline.p05_ai_selector.signals.earnings_calendar.EarningsCalendar.get_earnings_within_days(tickers, as_of_date, window_days)` — already takes an arbitrary ticker list, FMP-backed, monthly cache. (P20 Kestrel's calendar was considered too — rejected because it's bound to Kestrel's own watchlist tables, not a fit for "whatever I currently hold") |
| Notification delivery | `src.notification.service.client` — same combined Telegram + Email digest pattern as `pnl_alert/notifier.py` |
| Scheduling | Existing APScheduler `job_schedules` table + `seed_schedule.py` convention (`pnl_alert.seed_schedule` is the template) — no new orchestration tech |
| DB (only if a history table is wanted — see Phase 2) | `src/data/db` (SQLAlchemy + Alembic, already supports the deploy's DB URL) — no new database decision to make |

No PostgreSQL setup, no Celery, no new Telegram bot wiring, no new Flex
downloader/XML parser, no new earnings-calendar source.

---

## Design

### Decisions

- **Re-alert policy:** alert every run the condition is still true — no
  dedup/snooze/first-crossing-only logic for now. Simple and loud by design;
  revisit once it's clear whether that's too noisy in practice.
- **Account:** live only, read-only connection (see "Live account only"
  above). Never connects to paper for this module.
- **Cadence — earnings-triggered only, no baseline poll.** Confirmed:
  there's no independent always-on stop-coverage check. The system checks
  stop coverage *only* at **T-1 day** and **T-1 hour** before each held
  ticker's own earnings date/time (BMO/AMC). A ticker with no earnings in
  the lookahead window simply isn't checked — coverage matters most right
  before a volatility event, and this keeps the design to one job instead
  of two.

### Phase 1 (MVP) — Earnings-triggered stop-coverage reminder

1. Load current holdings (reuse `pnl_alert`'s loader).
2. For each held ticker, get its earnings date/time via
   `EarningsCalendar.get_earnings_within_days()`.
3. Schedule two triggers per ticker with an upcoming earnings date: T-1 day
   and T-1 hour before that date/time. (Dynamic per-ticker scheduling, not a
   fixed cron — dates move as new calendar data comes in, so this needs to
   re-resolve trigger times each time the calendar is refreshed, the same
   way the original brainstorm's "Dynamic Reminders" section already
   described for its own T-1h/T+0/T+1h triggers.)
4. When a trigger fires: connect to the **live** Gateway read-only, call
   `reqAllOpenOrders()`, filter to `STP` / `STP LMT` / `TRAIL` order types,
   sum working quantity for that ticker, and compare to position quantity →
   `covered` / `partially_covered` / `uncovered`.
5. Send one Telegram + Email message (same channel as `pnl_alert`) —
   *"XYZ has earnings in 1 hour (AMC); currently uncovered — no live stop
   order found."* Fires every time the trigger condition is met, no dedup
   (per "Decisions" above). Purely informational; the system never touches
   orders around it.

This needs two new pieces of code: the open-orders fetch + matcher, and the
per-ticker T-1day/T-1hour trigger scheduling. Holdings, earnings dates, and
notification delivery are all reused as-is.

### Phase 2 (optional) — Persisted compliance history

Only if you later want to audit "was I covered on this day" retroactively:
one lightweight table, `stop_compliance_log(ticker, checked_at, position_qty,
covered_qty, status)`, appended each run via the existing `src/data/db`
engine/Alembic migration flow. Not needed for the reminder itself, which is
stateless (poll live state, diff, alert).

---

## Rejected from the original brainstorm

- **PostgreSQL / new schema for snapshots & trade history** — `pnl_alert`
  already archives dated `Open_Positions-YYYY-MM-DD.xml` snapshots daily;
  a full transaction ledger isn't needed for a reminder system and can be
  revisited separately if a real audit/tax use case shows up.
- **SEC EDGAR 8-K + LLM date extraction** — real engineering cost for a
  problem the FMP calendar already solves adequately; keep as a possible
  future cross-check, not a Phase 1/2 dependency.
- **LLM-driven "disable/reactivate trailing stop" automation** — dropped
  entirely per the scope above: the user sets stops manually and wants to be
  *told*, not have orders touched automatically. Letting an LLM call decide
  to modify a live protective order was also a real safety concern worth
  flagging even before the scope changed: if this ever becomes automated,
  the disable/reinstate timing should be a deterministic rule off the
  earnings date, never an LLM judgment call, and any post-earnings summary
  text should stay firmly separate from anything that can place an order.
- **Celery** — APScheduler is already the standard here; no reason to run
  two schedulers.
