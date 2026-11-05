# Tasks

## Implementation Status

### ✅ COMPLETED
- [x] Brainstorm + decisions — [brainstorm.md](brainstorm.md)
- [x] `My Trades` Activity Flex Query (id 1695936) — [flex-query-setup.md](flex-query-setup.md)
- [x] `IBKR_FLEX_TRADES_QUERY_ID` in `donotshare.py`, `.env.example`, `.env` (dev + Pi)
- [x] Backfill XML 2020-10-20 → 2026-10-30, Trades = Select All (dev + Pi)
- [x] Spec: Requirements.md, Design.md, Tasks.md

### 🔄 USER ACTIONS
- [x] Add **Open Positions** section (Options: Summary only, no Lot; Select All) to
      `My Trades` and re-download the 7 backfill years (dev + Pi) — gives
      year-end reconciliation checkpoints for the whole history (see Design).
- [x] Check Flex Web Service token expiry date: valid 2026-08-12 → **2027-07-14 13:09 EDT**.
      Renew before then (Client Portal → Performance & Reports → Flex Queries → Flex Web Service
      Configuration gear); update the token on dev and in the Pi `.env` (pnl_alert + trade_ledger use it).

### 🚀 PLANNED (implementation order)
1. [x] `flex_downloader`: extract public `download_flex_statement(token, query_id) -> bytes`
       (pnl_alert behaviour unchanged; its tests stay green)
2. [x] `models.py` dataclasses (input events; engine outputs added in step 5)
3. [x] `xml_parser.py` + tests (synthetic fixture, 15 tests); parses all 7 backfill files: 389 STK executions, 0 malformed
4. [x] DB model `model_trade_ledger.py` + migration `014_trade_ledger.py` + repo/service + `repository.py` adapter + tests;
       verified on a scratch Postgres DB: upgrade/downgrade/upgrade, repo tests pass, full backfill import
       389 executions / 14 CA legs / 6 transfers / 63 open positions, re-import inserts 0, exact round-trip
5. [x] `ledger_engine.py` + 18 tests (buy/sell WAC, commission variants, cycle reset,
       reverse split with linked legs, chained splits, cash in lieu, cash merger, spin-off,
       internal/external transfer, short warning, ordering, commission refunds, month/YTD);
       commission adjustments added end-to-end (parser, migration 014 table, repo, engine);
       validated on real backfill — see Design "Validation of the real engine"
6. [x] `reconciler.py` + 9 tests (snapshot = last business day in month, ledger rebuilt at that date,
       accounts combined, split-conid mapping, no_snapshot, reconcile_all); real backfill: 7/7 checkpoints ok
7. [x] `report_builder.py` + `xlsx_writer.py` (HTML, plain text, XLSX); P&L-per-ticker table
       (dividends ingested as cash transactions, IBKR `markPrice` stored on open positions,
       lifetime totals in the engine), YTD table (incl. positions closed earlier in the year),
       full account IDs + 11 report tests; layout agreed 2026-11-04 (4-decimal averages)
8. [x] `notifier.py`, `runner.py`, `cli.py`, `__main__.py`, `config.py` + YAML; 26 tests (runner, config,
       notifier, CLI); scratch-DB end-to-end incl. real Flex download (see Design "Runner")
9. [x] Scheduler branch `portfolio.trade_ledger` (returns `success` + `error`, so a failed report is a FAILED run;
       generic `error` support added to the scheduler's failed-result message) + `seed_schedule.py`; 4 tests.
       Also fixed: `NotificationServiceClient.send_notification(html=...)` — pnl_alert / management HTML
       went into metadata and never reached the email body
10. [x] README.md
11. [x] Validation on real history (dev, 2026-11-04, final code): shares exact at all 7 IBKR
        checkpoints; WAC incl. commission = IBKR cost basis for every position never partially
        sold; the differing ones (MSFT, PINS, CRSP, KRMN) equal our own FIFO replay exactly,
        VT differs only by IBKR's per-account transfer cost; scratch-DB `import` → `reconcile`
        7/7 → `run --month 2026-10 --dry-run` with a real Flex download ok
12. [x] pyright + mypy clean (`scripts/typecheck.py`, whole repo, 2026-11-04)
13. [x] Deploy on Pi: migration 014, import backfill, seed schedule, one manual `run --month 2026-10` (live 2026-11-04;
        deposit rows confirmed in the Pi DB 2026-11-05)
14. [x] **Deposits** (requested and built 2026-11-04): net deposits for the month, YTD and since start, per currency,
        as tiles / a small table. No Flex change needed — Cash Transactions already carry type
        `Deposits/Withdrawals` (backfill: 54 deposits, all CHF, no withdrawals). Add the type to
        `CASH_TYPES` (account-level rows: no conid), store, sum in the report. Exclude rows whose
        description starts with "ADJUSTMENT" (IBKR files a 0.01 USD commission adjustment under this type).
        Internal cash transfers between own accounts, if any appear, net to zero when accounts are combined.
        Done: parser keeps account-level `Deposits/Withdrawals` rows (conid / symbol ""), same table
        (no migration change); engine `DepositTotals` per currency; tile + plain-text line + `Deposits`
        XLSX sheet; 6 tests. Real backfill: Oct 2026 2,200 CHF, YTD 22,100, total 167,300 since
        2020-10-20 (54 deposits) = independent XML sum.
        Deploy: after `git pull` re-run `import data/portfolio/trade_ledger/raw/backfill` once — the
        already-imported files now yield the deposit rows (idempotent, only the new rows are inserted).

## Technical Debt
- None yet.

## Known Issues / Open Points
- Cash in lieu on reverse splits: none in history — all 7 reverse splits have
  `proceeds=0` and IBKR keeps fractional shares (e.g. BINI 0.04 → 0.0002),
  so `Decimal` quantities are mandatory. Cash-in-lieu handling stays in the
  engine for future cases.
- Spin-offs (none in history so far): cost allocation between parent and
  child is not in the Flex data; would need manual input. Engine warns.
- Open Positions reflect the *end* of each download range, so reconciliation
  checkpoints are the range ends (year-ends + 2026-10-30), not every month.

## Testing Requirements
- [x] Unit tests per component (pure engine fully covered)
- [x] Parser test on a fixture that mimics the real multi-statement layout
- [x] Idempotent re-import test (same file twice → same row counts): `test_repository.py`, `test_runner.py`, `test_repo_trade_ledger.py`
- [x] End-to-end dry run on real backfill data (manual, step 11)

## Documentation Updates
- [x] README.md with Quick Start
- [x] Update `pnl_alert` Design.md: flex_downloader generalized
- [x] `docs/SUBMODULES.md` entry (section 8.2)
