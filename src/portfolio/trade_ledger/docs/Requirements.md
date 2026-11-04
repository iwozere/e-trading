# Requirements

Status: spec (2026-11-04). Background and decisions: [brainstorm.md](brainstorm.md).

## Functional Requirements

### FR-1 Data ingestion
- FR-1.1 Download the `My Trades` Activity Flex Query (id in
  `IBKR_FLEX_TRADES_QUERY_ID`, period *Last Month*) through the Flex Web
  Service once a month.
- FR-1.2 Archive every downloaded statement verbatim to
  `data/portfolio/trade_ledger/raw/monthly/My_Trades_YYYY-MM.xml` before
  parsing it.
- FR-1.3 Import any directory of Flex XML files on demand (one-time backfill
  of `data/portfolio/trade_ledger/raw/backfill/My_Trades_YYYY.xml`, 2020-10-20
  → 2026-10-30), and allow the same files to be imported again.
- FR-1.4 Import is idempotent: executions are unique on
  `(account_id, trade_id)`, corporate actions and transfers on
  `(account_id, transaction_id)`. Overlapping periods and re-imports never
  duplicate rows.
- FR-1.5 Parse every `FlexStatement` in a file (the files contain two
  accounts from 2023 on).
- FR-1.6 Keep only `assetCategory=STK` executions; `CASH` rows (FX
  conversions) are skipped and counted in the run summary.

### FR-2 Ledger engine
- FR-2.1 Replay all stored events per position in chronological order
  (`dateTime`, then `trade_id` / `transaction_id` as tie-breaker).
- FR-2.2 Both accounts are combined into one ledger per position. Internal
  transfers between the user's own accounts do not change quantity or cost.
- FR-2.3 Maintain two weighted average costs per position:
  - **WAC excl. commission** = Σ(buy qty × trade price) / shares held
  - **WAC incl. commission** = Σ(buy qty × trade price + |buy commission|) / shares held
  Sells reduce shares at the current WAC (the average does not change on a sell).
- FR-2.4 Restart a position when its quantity reaches exactly 0: the next buy
  opens a new cycle with fresh WAC values.
- FR-2.5 Realized P&L per sell, against the incl.-commission WAC:
  `proceeds − |sell commission| − qty × WAC_incl`. Also store the
  excl.-commission variant (`proceeds − qty × WAC_excl`).
- FR-2.6 Corporate actions:
  - forward/reverse splits and ticker/ISIN changes (legs linked by
    `actionID`): carry the full cost of the removed leg to the added leg;
    quantity changes, total cost unchanged, so WAC rescales automatically;
  - cash paid on an action (e.g. cash in lieu of fractional shares, cash
    mergers): treat as a sell of the affected quantity;
  - action types the engine does not handle: keep the position unchanged and
    raise a warning in the report.
- FR-2.7 Positions are identified across symbol / ISIN / conid changes by
  following corporate-action links; the display symbol is the latest one.
- FR-2.8 Short positions (selling below zero) are not expected; if one
  occurs, flag it as a warning and do not compute WAC for it.
- FR-2.9 All arithmetic uses `Decimal`; no float rounding in stored values.
- FR-2.10 Each position keeps its trading currency (almost always USD). No
  conversion to the base currency.
- FR-2.11 Commission adjustments (Cash Transactions of type "Commission
  Adjustments", linked to a trade by `tradeID`): a refund on a buy whose
  shares are still held lowers the incl.-commission cost; otherwise it is
  realized P&L at the refund date. Both reduce the period's commissions.

### FR-3 Monthly report (email)
- FR-3.1 HTML body, section 1: **executions of the report month**, one row per
  execution: trade date, account, symbol, side, quantity, price, gross amount,
  commission, net amount, currency. Partial fills stay separate rows.
- FR-3.2 HTML body, section 2: **per-position summary** for every position
  with activity in the month or still held at month end: shares bought / sold
  in the month, shares held at month end, WAC excl. and incl. commission,
  IBKR cost-basis price (when available), realized P&L month and YTD,
  commissions month and YTD.
- FR-3.3 HTML body, section 3: totals (commissions, realized P&L month / YTD
  per currency) and warnings (reconciliation mismatches, unhandled corporate
  actions, short positions).
- FR-3.2a **Year to date** table: one row per ticker traded, realized or
  paying dividends since 1 January — including positions closed earlier in
  the year — with shares bought / sold / held, executions, commissions,
  realized P&L, net dividends, total; totals row per currency. The month
  positions table (FR-3.2) therefore shows month figures only.
- FR-3.2b Account IDs are shown in full (header and executions table).
- FR-3.2c **Net deposits** per currency (deposits minus withdrawals, all accounts
  combined) for the month, YTD and since the first deposit — a tile in the
  header, a line in the plain text, every deposit in the XLSX `Deposits`
  sheet. Source: Cash Transactions of type `Deposits/Withdrawals` (no Flex
  change); rows whose description starts with "ADJUSTMENT" are booking
  corrections and are ignored.
- FR-3.3a Last section: **P&L per ticker** for every ticker since the start of
  history (2020): shares held, average incl. commission, IBKR month-end
  `markPrice`, market value, unrealized (= shares × mark − remaining cost incl.
  commission), realized (all cycles, after commissions), net dividends
  (dividends + payments in lieu + withholding tax), total = realized + dividends
  + unrealized, lifetime commissions; one totals row per currency. Selling costs
  are not deducted. A held ticker without a mark price shows "n/a" and its
  total excludes unrealized. Market value / unrealized totals are blank for a
  currency with no holdings.
- FR-3.4 XLSX attachment `trade_ledger_YYYY-MM.xlsx` with sheets:
  `Summary`, `Executions` (full history), `Positions` (state at month end),
  `Cycles` (closed position cycles with realized P&L), `Corporate Actions`,
  `Reconciliation`, `YTD <year>`, `Deposits`, `P&L per Ticker`, `Warnings`.
- FR-3.5 Sent by email only, to `recipient_id` from config (default 2,
  same as `pnl_alert`).
- FR-3.6 A month without executions still sends the report (holdings summary).

### FR-4 Reconciliation
- FR-4.1 Compare month-end quantities from the ledger with IBKR's month-end
  open positions (Open Positions section of `My Trades` once added, see
  Tasks). Any mismatch is a warning in the report and in the run summary.
- FR-4.1a IBKR dates the snapshot on the last business day of the period; the
  ledger is compared as of that same date. Months without a snapshot are
  reported as "no snapshot" (not as a mismatch).
- FR-4.2 Show IBKR's `costBasisPrice` next to our WAC for each held position.

### FR-5 Operation
- FR-5.1 Scheduled monthly on the Pi: cron `0 6 2 * *` (UTC) — 2nd of the
  month, when the previous month is final.
- FR-5.2 CLI (`python -m src.portfolio.trade_ledger`): `import PATH...` (files or
  directories), `run [--month YYYY-MM] [--no-download] [--dry-run]`, `reconcile`
  (all stored snapshots).
- FR-5.3 A failed download falls back to the archived file for that month if
  present; otherwise the run fails and sends a short failure email.
- FR-5.4 The scheduler stores a JSON run summary (rows imported, skipped,
  warnings, email sent).

## Non-Functional Requirements
- Full-history rebuild (~400 executions, ~15 corporate actions) finishes in
  seconds on the Pi; the engine always recomputes from stored events and keeps
  no derived state.
- The ledger engine is pure (no I/O) and fully unit-testable.
- Raw XML files contain account ids and trades: kept under `data/` only
  (gitignored), never logged in full; account ids masked in logs.

## Python Dependencies
- `requests` (Flex Web Service, via the existing downloader)
- `openpyxl` >= 3.1.5 (XLSX; already in `requirements.txt`)
- `SQLAlchemy`, `alembic` (already used by `src.data.db`)
- standard library: `xml.etree.ElementTree`, `decimal`

## External Dependencies (cross-module)
- `src.portfolio.pnl_alert.flex_downloader` — Flex Web Service
  SendRequest / GetStatement client (to be generalized, see Design).
- `src.data.db` — SQLAlchemy models, session, alembic migration `014_trade_ledger`; `TradeLedgerRepo` + `TradeLedgerService`.
- `src.notification.service.client.NotificationServiceClient` — email with
  file-path attachment.
- `src.scheduler.scheduler_service` — `portfolio.*` dispatch branch.
- `src.notification.logger` — logging.
- `config.donotshare.donotshare` — `IBKR_FLEX_TOKEN`, `IBKR_FLEX_TRADES_QUERY_ID`.

## External Services
- IBKR Flex Web Service (token expires; renew in Client Portal).
- SMTP via the notification service.
- PostgreSQL (prod DB on the Pi).

## Security Requirements
- Token and query id only from `config/donotshare/.env`; never in code,
  logs, or emails.
- Account ids masked (`U4***31`) in logs and the run summary; shown in full
  only in the email / XLSX sent to the account owner.
