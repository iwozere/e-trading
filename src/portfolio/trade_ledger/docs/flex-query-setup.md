# IBKR Flex Query Setup — Trade Ledger

How to create the Activity Flex Query used by the trade ledger.
Context and design: [brainstorm.md](brainstorm.md).

## Do I tick the same fields in every section?

**No.** Each section has its own field list (Trades have `Trade Price` and
`IB Commission`; Corporate Actions have `Type` and `Action Description`;
Transfers have `Direction` and `Transfer Price`, and so on). Only a handful of
identifier fields are shared (Account ID, Currency, FX Rate To Base, Asset
Class, Symbol, Conid, ISIN, Date/Time, Transaction ID). Tick those in every
section — they are what lets the ledger link rows across sections.

**Recommendation: use "Select All" in every section.** The raw XML is archived
verbatim, so extra fields cost nothing, and if we later need a field we did not
foresee we will not have to re-download years of history. The per-section lists
below are the **minimum** — use them as a checklist to confirm the essential
fields are present (field labels may differ slightly in the Client Portal UI).

## 1. Create the query

Client Portal → **Performance & Reports → Flex Queries → Activity Flex Query →
Create (+)**.

- **Query name:** `My Trades` (Query ID 1695936)
- Add the sections below.

## 2. Sections

### Trades (required)

**Options / level of detail:** tick **Executions** only.
Untick *Orders*, *Closed Lots*, *Wash Sales*, *Symbol Summary*,
*Asset Class Summary* — lot rows are exactly the noise we want to avoid.

Minimum fields:

| Field | Why |
|---|---|
| Account ID | multi-account safety |
| Currency, FX Rate To Base | trade currency + base-currency conversion |
| Asset Class | filter STK vs OPT/CASH |
| Symbol, Description, Conid, ISIN, Security ID, Listing Exchange | identity; Conid/ISIN survive ticker changes |
| Multiplier | correct for options if ever included |
| Trade ID, Transaction ID, Exec ID (if offered) | **dedup key** (Trade ID) |
| Trade Date, Date/Time, Settle Date Target | ordering + reporting month |
| Transaction Type, Notes/Codes | detect cancellations, corrections, assignments |
| Buy/Sell, Quantity, Trade Price | the trade itself |
| Trade Money, Proceeds | gross amount |
| IB Commission, IB Commission Currency | commission line per execution |
| Taxes | stamp duty / FTT where applicable |
| Net Cash | amount actually paid / received |
| Cost Basis, FIFO P/L Realized | IBKR's figures, shown alongside our WAC |
| Open/Close Indicator | sanity check |
| Order ID | group partial fills of one order |

### Corporate Actions (required)

Needed for splits, reverse splits, mergers, ticker changes, spin-offs —
without them WAC is silently wrong.

| Field | Why |
|---|---|
| Account ID, Currency, FX Rate To Base, Asset Class | shared identifiers |
| Symbol, Description, Conid, ISIN, Security ID | identity (old and new line both appear) |
| Report Date, Date/Time | ordering |
| Type | action code (forward split, reverse split, ticker change, merger, spin-off, …) |
| Action Description | human-readable ratio / details |
| Quantity | share delta caused by the action |
| Proceeds, Value | cash-and-stock mergers |
| Cost Basis, FIFO P/L Realized | IBKR's figures for cross-checking |
| Transaction ID, Action ID | dedup + linking the legs of one action |

### Transfers (required if shares ever moved between brokers)

| Field | Why |
|---|---|
| Account ID, Currency, FX Rate To Base, Asset Class | shared identifiers |
| Symbol, Description, Conid, ISIN, Security ID | identity |
| Date, Date/Time | ordering |
| Type, Direction | ACATS / internal, in / out |
| Company, Account (counterparty) | where it came from |
| Quantity | shares moved |
| Transfer Price, Position Amount, Position Amount In Base | cost basis of transferred-in shares (if IBKR has it) |
| Transaction ID | dedup |

### Cash Transactions (required — commission adjustments)

Needed for **Commission Adjustments**: IBKR books commission refunds weeks
after the trade as a separate cash transaction (e.g. a $1 refund on a 2023
BUG buy), linked to the trade by `tradeID`. Without them the incl.-commission
average is slightly too high. The ledger reads only this type; dividends,
interest etc. in the same section are ignored. Under *Options*, Select All.

| Field | Why |
|---|---|
| Account ID, Currency, FX Rate To Base, Asset Class | shared identifiers |
| Symbol, Description, Conid, ISIN | identity |
| Date/Time, Settle Date, Report Date | ordering |
| Type, Amount | dividend / tax and amount |
| Transaction ID, Action ID | dedup + link dividend to its withholding tax |

### Not needed

Open Positions is already pulled by the existing `pnl_alert` query
(`IBKR_FLEX_QUERY_ID`) — the ledger can reconcile share counts against it, so
no need to add it here.

## 3. Delivery configuration

| Setting | Value |
|---|---|
| Models | Optional / none |
| Format | **XML** |
| Period | **Last Month** (used by the monthly scheduled pull) |
| Date Format | `yyyyMMdd` |
| Time Format | `HHmmss` |
| Date/Time Separator | `;` (semicolon) |
| Profit and Loss | Default |
| Include Canceled Trades? | **No** |
| Include Currency Rates? | Yes (cheap; helps commission-currency conversion) |
| Include Audit Trail Fields? | No |
| Display Account Alias in Place of Account ID? | No |
| Breakout by Day? | No |

Save, then note the **Query ID** shown in the query list.

## 4. One-time history backfill

A single run covers **at most 365 days**.

1. In the Flex Queries list, click **Run** next to `My Trades`.
2. Choose a **custom date range** for one calendar year (e.g. 2021-01-01 →
   2021-12-31). If the Run dialog does not offer a period override, save a copy
   of the query named `My Trades - Backfill` with Period = *Custom Date Range*
   and edit its dates for each run.
3. Format XML → download. Save as `My_Trades_YYYY.xml` into `data/portfolio/trade_ledger/raw/backfill/`.
4. Repeat for every year back to account opening, and for the current year to date.

Overlapping ranges are harmless — rows are deduplicated by Trade ID /
Transaction ID.

## 5. Flex Web Service (for the monthly automated pull)

- **Settings → Flex Web Service**: already enabled for `pnl_alert`; the
  existing `IBKR_FLEX_TOKEN` works for every query on the account.
- Check the token's **expiry date** there — it expires and must be renewed, or
  both `pnl_alert` and the trade ledger stop downloading.
- Add the new query id to `config/donotshare/.env` (dev + Pi; template: `config/donotshare/.env.example`):

  ```
  IBKR_FLEX_TRADES_QUERY_ID=<query id>
  ```

## Checklist

- [x] `My Trades` Activity Flex Query created (Trades = Executions only, Corporate Actions, Transfers, Cash Transactions)
- [ ] Format XML, Period Last Month, Include Canceled Trades = No
- [x] Query ID noted → `IBKR_FLEX_TRADES_QUERY_ID` in `config/donotshare/.env` (dev + Pi) — done
- [ ] Flex Web Service token expiry checked
- [x] Yearly XML backfill downloaded back to account opening (2020-10-20 → 2026-10-30, local + Pi)
