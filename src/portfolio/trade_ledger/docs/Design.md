# Design

Status: spec (2026-11-04). Requirements: [Requirements.md](Requirements.md).

## Purpose
IBKR shows cost basis per tax lot (FIFO): after partial sells the displayed
average price is the average of the remaining lots, not of everything bought.
This module rebuilds the full trade history from Flex Query XML, computes a
true weighted average cost (WAC) per position, and emails a monthly report of
every execution with its commission.

## Architecture

### High-Level Architecture
```
APScheduler (cron 0 6 2 * *, target "portfolio.trade_ledger")
   │
   ▼
runner.run_once(cfg, month)  → asyncio.to_thread(run_month, ...)
   ├─► flex_downloader.download_flex_statement(token, query_id)   (pnl_alert, generalized)
   │      └─ archive raw/monthly/My_Trades_YYYY-MM.xml
   ├─► xml_parser.parse_flex_file(path) ─► ParsedStatement (executions, corporate actions, transfers, open positions)
   ├─► repository.upsert(...)          ─► trade_ledger_* tables (idempotent)
   ├─► repository.load_all()           ─► all events
   ├─► ledger_engine.build_ledger(events, as_of=month_end) ─► LedgerResult (pure)
   ├─► reconciler.reconcile(ledger, ibkr_open_positions)  ─► warnings (pure)
   ├─► report_builder.build_html(...) / build_xlsx(...)
   └─► notifier.send_report(view, xlsx_path)  ─► NotificationService.create_message (email)
```

### Component Design

| File | Responsibility |
|---|---|
| `config.py` + `config/trade_ledger.yaml` | `TradeLedgerConfig` dataclass: recipient_id, raw/output dirs, cron, channels |
| `models.py` | Frozen dataclasses: `Execution`, `CorporateActionLeg`, `TransferEvent`, `IbkrOpenPosition`, `PositionState`, `ClosedCycle`, `LedgerResult` |
| `xml_parser.py` | Flex XML → dataclasses. Parses every `FlexStatement`; skips non-STK; `Decimal` for all numbers; `yyyyMMdd;HHmmss` dates |
| `repository.py` | Adapter: dataclasses ↔ DB rows via `TradeLedgerService`; no SQL, no business logic |
| `ledger_engine.py` | Pure replay: WAC, cycles, realized P&L, corporate actions, transfers |
| `reconciler.py` | Pure: ledger month-end quantities vs IBKR open positions |
| `report_builder.py` | HTML body + XLSX (openpyxl) |
| `notifier.py` | Queues the email via `NotificationService.create_message`: `content.html` (body fragment) + `content.attachments` (XLSX as base64); failure email |
| `runner.py` | Orchestration + `RunSummary` |
| `cli.py` / `__main__.py` | `run [--month YYYY-MM] [--no-download] [--dry-run]`, `import PATH...`, `reconcile` |
| `seed_schedule.py` | Idempotent `job_schedules` row (same pattern as `pnl_alert`) |

DB layer (project pattern, same as the signal ledger):
- model `src/data/db/models/model_trade_ledger.py`
- migration `src/data/db/migrations/versions/014_trade_ledger.py` (revises 013)
- repo `src/data/db/repos/repo_trade_ledger.py` (`TradeLedgerRepo`, registered as
  `ReposBundle.trade_ledger`)
- service `src/data/db/services/trade_ledger_service.py` (`TradeLedgerService`,
  one transaction per imported file)

## Data Model

Column names equal the dataclass field names in `models.py`, so conversion is
`asdict()` / `Model(**row)`.

```
trade_ledger_executions
  id PK, account_id, trade_id, transaction_id, ib_order_id, ib_exec_id,
  conid, isin, symbol, description, listing_exchange, sub_category,
  currency, fx_rate_to_base, trade_date, trade_datetime,
  buy_sell, quantity, trade_price, proceeds, ib_commission,
  ib_commission_currency, taxes, net_cash, cost, fifo_pnl_realized,
  open_close, notes, source_file, imported_at
  UNIQUE (account_id, trade_id)

trade_ledger_corporate_actions
  id PK, account_id, transaction_id, action_id, action_type, conid, isin, symbol,
  action_description, report_date, action_datetime, quantity, proceeds,
  value, cost_basis, currency, source_file, imported_at
  UNIQUE (account_id, transaction_id)

trade_ledger_transfers
  id PK, account_id, transaction_id, transfer_type, direction, counterparty_account,
  conid, isin, symbol, transfer_datetime, quantity, transfer_price,
  position_amount, currency, source_file, imported_at
  UNIQUE (account_id, transaction_id)

trade_ledger_open_positions        (month-end IBKR snapshot, for reconciliation)
  id PK, account_id, report_date, conid, isin, symbol, position,
  cost_basis_price, cost_basis_money, currency, source_file, imported_at
  UNIQUE (account_id, report_date, conid)
```
Numeric columns are `NUMERIC` (no float). Only fields the engine or report
uses are columns; the raw XML stays the complete source of truth, so a new
column can always be back-filled by re-importing.

## Ledger Engine

### Position identity
A position is keyed by an internal `position_key`, initially the first
`conid` seen. A corporate action whose legs share an `actionID` and move
shares from conid A to conid B (reverse split with new ISIN, ticker change)
re-points B to A's key. Display symbol = symbol of the latest event.

Example from the data — MULN 1-for-100 reverse split in 2025:
```
leg 1: 20250530203741MULN  quantity -1000   (old line removed)
leg 2: MULN                quantity   +10   (new line added)
→ same position; shares 1000 → 10; total cost unchanged; WAC × 100
```

Verified on the backfill (2026-11-04): all 7 reverse splits come as two legs
sharing an `actionID`, the new leg always has a **new conid**, and chains
work across years and symbols — YQ conid 458895907 → 524662771 (2021) →
672370108 (2023); MULN 1000 → 10, then renamed BINI 10 → 0.04 → 0.0002.
All have `proceeds=0`; fractional shares are kept, never cashed out.
Commissions: 385 negative, 4 zero, none positive.

### State per position
```
shares            Decimal
cost_excl         Decimal   Σ qty × trade_price of open shares
cost_incl         Decimal   cost_excl + Σ |buy commission|
WAC_excl = cost_excl / shares,   WAC_incl = cost_incl / shares
cycle_no, cycle_open_date, realized (month / YTD / cycle)
```

### Event rules
| Event | Effect |
|---|---|
| BUY q @ p, commission c | shares += q; cost_excl += q·p; cost_incl += q·p + abs(c) |
| SELL q @ p, commission c | realized_incl = q·p − abs(c) − q·WAC_incl; cost_* −= q·WAC_*; shares −= q |
| shares reach 0 | close cycle (record `ClosedCycle`), reset cost to 0 |
| split / ticker change (linked legs, no cash) | shares = Σ new legs; cost carried unchanged |
| corporate action with cash | sell of the cashed quantity at proceeds |
| internal transfer (own accounts) | no-op (both accounts are one ledger) |
| commission adjustment (refund) | buy still held → cost_incl −= amount; otherwise realized += amount |
| action type not FS/RS/TC/IC | applied generically (conversion / cash sale / zero-cost receipt) + warning |
| sell below zero | warning; position marked short, WAC not computed |

Commission sign: IBKR reports commissions as negative numbers; the engine
always uses `abs()`.

Ordering: events sorted by `(datetime, kind_rank, id)`; corporate actions
apply after trades with the same timestamp. All legs sharing an `actionID`
are applied **together as one group** at the action's timestamp — before any
later event. (A prototype that deferred legs let a NUTX sell on the new conid,
two months after the split, open a separate negative position.)

### Validation on real history (prototype, 2026-11-04)
A throwaway share-count replay (trades + linked corporate-action legs, both
accounts combined, internal transfers ignored) matched IBKR's SUMMARY open
positions **exactly at all 7 checkpoints** (31 Dec 2020–2025, 2026-10-30).
This confirms: no positions before the first trade (2020-10-20), complete
reverse-split chains, nothing missing in the downloads. The real engine must
reproduce this as a regression test on the backfill (manual, Tasks step 11).

### Validation of the real engine (2026-11-04)
`ledger_engine.build_ledger` on the full backfill:
- share counts match IBKR at **all 7 checkpoints**, both accounts combined;
  0 warnings; 67 closed cycles; 7 reverse splits applied.
- for every position with buys only in its current cycle, WAC incl.
  commission equals IBKR's `costBasisPrice` exactly — independent proof of
  the commission handling. One case (BUG 2023, $1.00 off) exposed IBKR's
  **Commission Adjustments** refunds → added as a fifth event type.
- remaining differences are all explained: MSFT, PINS, CRSP, KRMN — a FIFO
  replay of the same events reproduces IBKR's figure exactly (partial sells,
  IBKR closes oldest lots first); VT — IBKR moves transferred shares between
  accounts at an assigned `cost` and runs FIFO per account (account U1…47's
  basis reproduced from the transfer `cost` field: 2278.44 / 4582.72 /
  6813.62). Our combined WAC ignores internal transfers by design.

### Outputs (`LedgerResult`)
- per-execution rows annotated with WAC after the event and realized P&L
  (for the XLSX `Executions` sheet);
- positions at `as_of` (month end);
- closed cycles;
- month / YTD / lifetime aggregates per position (`PeriodTotals`, incl. net
  dividends) and per currency;
- warnings.

### Deposits
Account-level `CashTransaction` rows of type `Deposits/Withdrawals` (empty
`assetCategory`, so they bypass the STK filter via `_collect(stk_only=False)`)
are stored in `trade_ledger_cash_transactions` with `conid = symbol = ""`.
The engine sums them per currency into `LedgerResult.deposits`
(`DepositTotals`: month, YTD, lifetime, first date) and keeps the rows for the
XLSX; they never touch positions or P&L. `CashTransaction.is_deposit` drops
booking corrections IBKR files under the same type ("ADJUSTMENT: ..."). A
cash transfer between the user's own accounts would appear as -X / +X and
nets to zero, consistent with the combined-accounts view.

### P&L per ticker (report)
`report_builder.build_ticker_pnl` combines `lifetime_totals` (realized incl.
commission over all cycles + net dividends) with unrealized P&L at IBKR's
`markPrice` from the reconciliation snapshot (`ReconciliationReport.mark_price_by_key`).
Prices therefore come from the same Flex statement as the share check, with
no extra market-data dependency.

Checked on real history (2026-10-30): our lifetime USD realized + unrealized
= 91,083.96 vs IBKR FIFO realized + `fifoPnlUnrealized` = 91,083.97. The
split between realized and unrealized differs by ~1,197 (WAC vs FIFO lot
allocation, per-account transfer cost), but the sum agrees.

## Design Decisions

### Recompute everything, store only events
The engine replays the whole history on every run (~400 events, sub-second)
instead of storing running WAC. No derived state to drift or migrate; a fix
in the engine fixes all history on the next run.

### Postgres for events, XML as source of truth
Raw XML is archived verbatim. The DB tables give idempotent upserts on IBKR's
ids and make ad-hoc queries easy. Re-importing the XML directory rebuilds the
tables from scratch if needed.

### Combined accounts
Decided 2026-11-04: one ledger per position across both accounts. Account is
a column in the execution rows. Internal transfers are therefore no-ops.

### Two WAC columns
Decided 2026-11-04: show both. Realized P&L uses the incl.-commission WAC
(true economic result); the excl. variant is in the XLSX.

### Cycle reset at zero
Decided 2026-11-04: when shares reach exactly 0 the position cycle closes and
the next buy starts fresh.

### Reuse and generalize the Flex downloader
`pnl_alert/flex_downloader.py` already implements SendRequest / GetStatement.
Its internals become a public `download_flex_statement(token, query_id) ->
bytes`; `download_open_positions_xml` keeps its current behaviour on top of
it. `trade_ledger` imports the public function. Same token for both queries.

### Reconciliation against IBKR month-end positions
Adding an **Open Positions** section (level: Summary) to `My Trades` makes the
monthly statement include IBKR's positions *as of month end* — the exact
moment the ledger is evaluated. Using `pnl_alert`'s daily `Open_Positions.xml`
instead would mismatch whenever the user trades on the 1st or 2nd of the month.
Re-downloaded backfill files (with this section) give a checkpoint at each
range end (31 Dec of 2020–2025, 2026-10-30) to validate the whole history.

Implementation (`reconciler.py`): IBKR dates the snapshot on the last
*business* day (2022-12-30, 2026-10-30), so `reconcile_period` takes the
latest snapshot inside the report month and rebuilds the ledger **as of that
date** before comparing — trades on the calendar month end can never fall
between the two. Shares must match exactly (accounts combined, IBKR conids
mapped through corporate-action links). IBKR's combined cost-basis price is
carried for display only. Months without a snapshot report `no_snapshot`.
`reconcile_all` checks every snapshot in the data (full-history regression):
7/7 `ok` on the backfill, 2026-11-04.

### Scheduler integration
`job_type = "alert"`, `target = "portfolio.trade_ledger"` — one more branch in
`SchedulerService`'s existing `portfolio.*` dispatcher, same as `pnl_alert`
and `management`. No change to the `job_schedules` check constraint.
- The branch returns `{"target", "success", "summary"}` plus `error` when
  `RunSummary.ok` is False; `is_failed_job_result` then records the run as
  FAILED with that error (the scheduler now prefers a result's `error` over
  the script-style exit-code message).
- The row is created by `python -m src.portfolio.trade_ledger.seed_schedule`
  (idempotent, user 1, name `portfolio_trade_ledger`, cron from the YAML).
  The data-pipeline registry (`src/data/pipeline`) is not used: it is for
  `run_*.py` data scripts, while portfolio jobs dispatch by target.

### Email only
The report carries an XLSX attachment and a wide table; Telegram is not used.

### Email transport
- The message is created with `NotificationService.create_message` (the P05 /
  P17 pattern). `NotificationServiceClient.send_notification(data={"html":...})`
  is not used: `data` lands in `message_metadata`, while the email channel
  reads the HTML body from `content.html` only.
- The XLSX goes in `content.attachments` as base64, so the notification bot
  needs no read access to the scheduler's `data/` folder.
- Gmail clips HTML bodies above ~102 KB. Cell styling is therefore set on the
  `<table>` (inherited) and via attributes (`align`, `cellpadding`, zebra
  `bgcolor`) instead of per-`<td>` inline styles: October 2026 shrank from
  192 KB to 58 KB. The notifier logs a warning above 102 KB.
- `format_html_body` is the email body (one `<div>`; the email channel adds
  `<html>`/`<body>`); `format_html` wraps it as a standalone document for the
  copy saved next to the XLSX.

### Runner
- `run_month` is synchronous; `run_once` runs it in a worker thread so the
  Flex polling does not block the scheduler's event loop.
- The downloaded statement is archived as
  `raw/monthly/My_Trades_<month of its toDate>.xml`; a statement that does not
  parse is kept under `raw/monthly/failed/`.
- `RunSummary.ok` is False when `errors` is non-empty (no statement for the
  month, unexpected exception, email not queued). The failure email is queued
  in those cases (not on `--dry-run`); the scheduler branch raises on `not ok`.
- End-to-end on a scratch DB (2026-11-04): `import` backfill → 389 executions /
  179 cash rows; `reconcile` 7/7 ok; `run --month 2026-10 --dry-run` with a real
  Flex download → archived `My_Trades_2026-10.xml`, 0 new rows (same IBKR ids
  as the backfill), 9 executions, reconciliation ok.

## Error Handling
- Download fails → use the archived file for that month if present; else
  run fails, short failure email, non-zero exit.
- Malformed row → logged with masked account id, skipped, counted in
  `RunSummary.skipped`.
- Unhandled corporate action / short position / reconciliation mismatch →
  report still sent, listed under Warnings.
- Email send fails → `RunSummary.email_sent = False`, scheduler marks failure;
  XLSX stays on disk in `data/portfolio/trade_ledger/reports/`.

## Integration Patterns
- Same structure as `src/portfolio/pnl_alert` (config YAML next to code,
  runner + CLI + seed_schedule).
- CLI:
  ```
  python -m src.portfolio.trade_ledger import --dir data/portfolio/trade_ledger/raw/backfill
  python -m src.portfolio.trade_ledger run --month 2026-10 [--dry-run] [--no-download]
  ```
