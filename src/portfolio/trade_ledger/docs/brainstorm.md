# Trade Ledger — Brainstorm

Status: brainstorm (2026-11-04). Nothing implemented yet.
Step 1 is setting up the Flex Query — see [flex-query-setup.md](flex-query-setup.md).

## Goal

A monthly email listing every share purchase and sale (one line per execution,
including commission), plus a per-ticker **weighted average cost (WAC)** that
reflects all buys — not IBKR's per-lot figure.

## Why IBKR's "average price" is misleading

IBKR computes cost basis via **tax-lot matching** (FIFO by default). Each sell
closes specific lots (the oldest first under FIFO), and the displayed average
price is the average of the *remaining open lots* — not the weighted average of
all purchases. After a few partial sells it diverges from the intuitive number.

WAC must therefore be computed by us, by replaying the full trade history per
ticker.

## Data source: Activity Flex Query

- One Activity Flex Query with sections **Trades** (level of detail:
  *Executions* only), **Corporate Actions**, **Transfers**, optionally
  **Cash Transactions** (dividends). Field choices: see
  [flex-query-setup.md](flex-query-setup.md).
- Format: XML.
- **Limit: max 365 days per run.** History is obtained by:
  1. **One-time backfill** — run the query manually with a custom date range,
     one calendar year at a time, back to account opening; save each XML.
  2. **Monthly** — scheduler pulls period *Last Month* via the Flex Web Service.
- The Flex Web Service may accept `fd` / `td` (YYYYMMDD) date overrides, which
  would allow a scripted backfill — unverified, to be tested before relying on it.
- TWS API `reqExecutions` (ib_async) only returns the last few days — not usable
  for history.

## Reuse of existing code

`src/portfolio/pnl_alert/flex_downloader.py` already implements the 2-step
SendRequest / GetStatement Flex Web Service flow using `IBKR_FLEX_TOKEN`. It is
hardwired to the "Open Positions" query; generalize it to accept any
`query_id`. The same token serves all queries. The new query id goes in a new
env var (proposed: `IBKR_FLEX_TRADES_QUERY_ID`).

## Proposed architecture

```
Flex "Trades" XML (monthly + one-time yearly backfill)
   │  raw XML archived verbatim: data/portfolio/trade_ledger/raw/monthly/My_Trades_YYYY-MM.xml
   ▼
parser → upsert into a trades table keyed by TradeID
   │  (idempotent: overlapping periods / re-downloads never duplicate)
   ▼
ledger engine: replays full history per ticker
   - WAC per ticker (incl. / excl. commission)
   - realized P&L under WAC (vs IBKR's FifoPnlRealized for comparison)
   - applies splits / ticker changes from Corporate Actions
   ▼
monthly email (1st of month, via scheduler)
```

Storage: raw XML is the source of truth (archived as-is); a Postgres table
keyed by `TradeID` on top for idempotent reruns and ad-hoc queries. A single
deduplicated `trades.json` is a viable DB-free alternative at personal-account
volumes.

## Email content (draft)

1. **Executions this month** — one line per execution: date, ticker, side,
   quantity, price, gross amount, commission, net amount, currency.
2. **Per-ticker summary** — shares bought/sold this month, current holding,
   WAC (with and without commission), IBKR cost basis alongside, realized P&L
   for the month and YTD.
3. **Attachment** — full ledger as CSV/XLSX (to be decided).

## Edge cases

- **Splits / reverse splits** — must rescale quantity and WAC; otherwise the
  averages are silently wrong.
- **Mergers / ticker changes** — key history by `Conid` / ISIN, not symbol.
- **Position fully closed, then re-bought** — does WAC restart? (usually yes)
- **Currency** — WAC in trade currency, base currency, or both? Matters for
  CH/EU tax.
- **Commission in a different currency** than the trade — needs FX conversion.
- **Shares transferred in from another broker** — no trade price in IBKR; cost
  basis comes from the Transfers section or manual input.

## Backfill data check (2026-11-04)

Files: `data/portfolio/trade_ledger/raw/backfill/My_Trades_2020.xml` …
`My_Trades_2026.xml` (2020-10-20 → 2026-10-30). Date format `yyyyMMdd;HHmmss`
as configured.

| Year | Trades (STK / CASH) | Corporate actions | Transfers | Accounts |
|---|---|---|---|---|
| 2020 | 7 / 6 | — | — | 1 |
| 2021 | 18 / 13 | 2 (RS) | — | 1 |
| 2022 | 10 / 10 | — | — | 1 |
| 2023 | 16 / 7 | 2 (RS) | 1 internal | 2 |
| 2024 | 26 / 8 | 2 (RS) | 1 internal | 2 |
| 2025 | 161 / 13 | 8 (RS) | 1 internal | 2 |
| 2026 | 148 / 7 | — | — | 2 |

Findings:

- **Trades section has only 19 fields** (the minimum list, not "Select All").
  Missing: `transactionID`, `taxes`, `cost`, `ibOrderID`, `transactionType`,
  `notes` (codes: cancellation / correction / assignment / partial),
  `multiplier`. The core ledger works without them, but `notes` and `taxes`
  matter for correctness and `cost` gives IBKR's basis for comparison.
  Corporate Actions / Transfers / Cash Transactions do have all fields.
- **Two accounts** from 2023 on, with internal transfers of shares between them
  (e.g. VT 24 shares OUT of one, IN to the other in 2025). An internal
  transfer must move the position *and its cost basis*, not count as a sale.
- **Reverse splits** occur in 2021, 2023, 2024, 2025 (7 actions, each as
  two legs: old line removed, new line added — e.g. `MULN` 1-for-100).
- **CASH rows** in Trades are FX conversions, not shares — exclude from the
  share ledger (or report separately).
- STK currencies: almost all USD, one CHF trade; commission currency always
  equals trade currency in the sample.
- No duplicate `tradeID` within a file.

**Re-download with Trades = "Select All" (2026-11-04, local + Pi):** 86 Trade
fields, nothing missing. Across all years (both accounts): 389 STK rows, all
`levelOfDetail=EXECUTION`, all `transactionType=ExchTrade`, multiplier 1
(ETF 87 / COMMON 277 / ADR 22 / REIT 3). Notes codes: `P` (partial execution)
62, `IA` (executed against IBKR/affiliate) 6 — no cancellations or
corrections. `taxes` is 0 everywhere. (The 386 count in the table above
covered only the first account per file.)

## Decisions (2026-11-04)

1. **Commission in WAC:** show both — average price excl. commission and
   incl. commission (break-even).
2. **WAC reset:** restart when a position goes to 0 shares; the closed cycle's
   realized P&L is finalized.
3. **Scope:** stocks/ETFs only (`assetCategory=STK`); CASH (FX conversion) rows
   excluded. Data has no options.
4. **External transfers:** none in history (only internal) — no manual
   cost-basis input needed for now.
5. **Email:** HTML body (this month's executions + per-ticker summary) plus the
   full ledger history attached as XLSX.
6. **Accounts:** combined — one ledger per ticker across both accounts;
   internal transfers between own accounts are no-ops for WAC; account shown as
   a column on each execution.

## Next steps

1. ~~Create the Flex Query~~ — done (`My Trades`, id 1695936).
2. ~~Backfill~~ — done, 2020–2026, re-downloaded with Trades = "Select All"
3. ~~Answer the open questions~~ — see Decisions.
4. Write the spec, then implement `src/portfolio/trade_ledger/` (README,
   docs/Requirements.md, Design.md, Tasks.md, tests/, scheduler job).
