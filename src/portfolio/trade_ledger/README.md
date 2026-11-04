# Trade Ledger

## Overview
Monthly email of every IBKR share execution (one row per execution, with
commission) and the **true weighted average cost per ticker**, across both
accounts. IBKR's own average is per lot (FIFO), which hides what a position
really cost; this module replays the full trade history since 2020 to compute
it. The source is the IBKR Flex Query "My Trades". The report is an HTML
email plus an XLSX attachment, sent on the 2nd of every month for the
previous month.

## Features
- Weighted average cost **excluding and including commission**; it does not
  change on a sell and restarts when a position goes to 0 shares (cycles).
- Realized P&L per execution, month, YTD and lifetime, after commissions.
- Corporate actions (reverse splits with new conids, ticker changes, cash in
  lieu, cash mergers), internal transfers between own accounts (no-ops),
  commission refunds, dividends net of withholding tax.
- **Year to date** table (incl. positions closed earlier in the year) and
  **P&L per ticker since 2020**: realized + net dividends + unrealized at
  IBKR's month-end closing price.
- Reconciliation: share counts are checked against IBKR's month-end open
  positions every month; mismatches are shown at the top of the email.
- Idempotent storage in Postgres (`trade_ledger_*` tables); raw Flex XML is
  archived verbatim.

## Quick Start
```bash
# One-time: load the history (files or directories, safe to repeat)
python -m src.portfolio.trade_ledger import data/portfolio/trade_ledger/raw/backfill

# Check the ledger against every stored IBKR snapshot (exit 1 on mismatch)
python -m src.portfolio.trade_ledger reconcile

# Build the report for a month without emailing it
python -m src.portfolio.trade_ledger run --month 2026-10 --dry-run

# Full run as the scheduler does it (previous month, download + email)
python -m src.portfolio.trade_ledger run

# Create / update the monthly schedule row (cron from the YAML)
python -m src.portfolio.trade_ledger.seed_schedule
```

```python
from datetime import date

from src.portfolio.trade_ledger.config import load_config
from src.portfolio.trade_ledger.runner import run_month

summary = run_month(load_config(), date(2026, 10, 1), dry_run=True)
print(summary.ok, summary.xlsx_path)
```

`run` options: `--month YYYY-MM` (default: previous month), `--no-download`
(report from what is already imported), `--dry-run` (write XLSX / HTML, send
nothing), `--config PATH`.

## Integration
This module integrates with:
- `src.portfolio.pnl_alert.flex_downloader` — Flex Web Service download (`download_flex_statement`)
- `src.data.db` — `TradeLedgerService` / `TradeLedgerRepo`, migration `014_trade_ledger`
- `src.data.db.services.notification_service` — email queue (HTML body + base64 XLSX)
- `src.scheduler` — `portfolio.trade_ledger` target (job type ALERT), cron `0 6 2 * *` UTC

## Configuration
- `config/trade_ledger.yaml`: `recipient_id` (2), `channels` (email),
  `cron`, `download`, `raw_dir`, `reports_dir`.
- `config/donotshare/.env`: `IBKR_FLEX_TOKEN` (shared with pnl_alert) and
  `IBKR_FLEX_TRADES_QUERY_ID` (the "My Trades" query, period *Last Month*).
  Query setup: [docs/flex-query-setup.md](docs/flex-query-setup.md).
- Files: monthly downloads in `data/portfolio/trade_ledger/raw/monthly/My_Trades_YYYY-MM.xml`,
  reports in `data/portfolio/trade_ledger/reports/trade_ledger_YYYY-MM.{xlsx,html}`.
  Both live under `data/` (gitignored; the only place scheduler.service may write).

A failed download falls back to the archived file of that month; without one
the run fails, a short "FAILED" email is sent and the scheduler run is
marked FAILED.

## Related Documentation
- [Requirements](docs/Requirements.md) - Technical requirements
- [Design](docs/Design.md) - Architecture, engine rules and validation results
- [Tasks](docs/Tasks.md) - Implementation roadmap
- [Brainstorm](docs/brainstorm.md) - Decisions taken with the owner
- [Flex query setup](docs/flex-query-setup.md) - IBKR Flex Query configuration
