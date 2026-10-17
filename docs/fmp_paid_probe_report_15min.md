# FMP paid probe report (Phase 2)

Hand-written findings go above the marker.

<!-- AUTO-GENERATED BELOW by `fmp probe-paid`; everything after this line is overwritten -->

Generated 2026-10-17 19:42 UTC. Calls used: **138 / 450**; this run's responses: 3.8 MB wire, 26.4 MB decoded.

## Settings `fmp plan` and `fmp build` now use (intraday `15min`)

Written to `<data_root>/probe-paid/probe_results.json` and applied over `config/fmp/fmp.yaml` (set `fmp.use_probe_results: false` to ignore):

```json
{
 "build": {
  "intraday_label": "start",
  "intraday_tz": "America/New_York"
 },
 "calendar_window_days": 30,
 "economic_indicators": [
  "CPI",
  "inflationRate",
  "federalFunds",
  "unemploymentRate",
  "initialClaims",
  "retailSales",
  "consumerSentiment",
  "industrialProductionTotalIndex"
 ],
 "intraday": {
  "bars_per_day": 26,
  "decoded_bytes_per_row": 144.6,
  "earliest": "2003-10-01",
  "max_rows": 782,
  "wire_bytes_per_row": 20.6
 },
 "page_size": 100
}
```

Also measured; applied instead when `intraday.interval` is set to one of these:

```json
{
 "1hour": {
  "build": {
   "intraday_label": "start",
   "intraday_tz": "America/New_York"
  },
  "intraday": {
   "bars_per_day": 7,
   "decoded_bytes_per_row": 145.5,
   "earliest": "2003-10-01",
   "max_rows": 428,
   "wire_bytes_per_row": 22.6
  }
 }
}
```

## Answers to the paid-probe items

| Item | Finding |
|---|---|
| §3.1 / A1 intraday intervals | 1min, 5min, 15min, 30min, 1hour, 4hour |
| §3.2 / A1 intraday depth (`15min`) | earliest month with bars: 2003-10-01 |
| §3.2 / A1 intraday depth (`1hour`) | earliest month with bars: 2003-10-01 |
| §3.3 row cap / maximum window (`15min`) | 782 rows, keeps the newest end |
| §3.3 row cap / maximum window (`1hour`) | 428 rows, keeps the newest end |
| Intraday bytes per row (`15min`) | 20.6 B/row wire, 144.6 B/row decoded |
| Intraday bytes per row (`1hour`) | 22.6 B/row wire, 145.5 B/row decoded |
| §3.4 / A2 bandwidth basis | measured by `fmp calibrate-usage` (dashboard vs local counters) |
| §3.5 / A4 delisted symbols in prices | TWTR: 22 bars, SIVB: 21 bars, BBBY: 21 bars, ATVI: 21 bars, LEH: ok_empty |
| §3.5 / A5 constituent endpoints | historical-sp500-constituent: ok (1531 rows), historical-nasdaq-constituent: ok (452 rows), historical-dowjones-constituent: ok (86 rows), sp500-constituent: ok (503 rows), nasdaq-constituent: ok (101 rows), dowjones-constituent: ok (30 rows) |
| §5.3 / A5 index prices | ^GSPC: plan_restricted, ^IXIC: plan_restricted, ^DJI: plan_restricted, ^VIX: plan_restricted, ^RUT: plan_restricted |
| §3.6 intraday split adjustment (`15min`) | split_adjusted |
| §3.6 intraday split adjustment (`1hour`) | split_adjusted |
| §3.7 / A9 / A11 label, time zone, extended hours, DST (`15min`) | `date` is America/New_York time labelling the bar start; 26 bars per regular day; extended hours no |
| §3.7 / A9 / A11 label, time zone, extended hours, DST (`1hour`) | `date` is America/New_York time labelling the bar start; 7 bars per regular day; extended hours no |
| A6 calendar `from`/`to` maximum window | splits-calendar: 90 days, dividends-calendar: 30 days, earnings-calendar: 30 days, ipos-calendar: 90 days, economic-calendar: 90 days |
| A7 class-share symbol | BRK-B: ok, BRK.B: plan_restricted |
| A7 exchange codes (stock-list, available-exchanges) | AMEX, AMS, ASX, ATH, BER, BME, BRU, BSE, BUE, CBOE, CNQ, CPH, DFM, DOH, DUB, DUS, DXE, FSX, HAM, HEL, HKSE, ICE, IOB, IST, JKT, JNB, JPX, KLS, KOE, KSC, LIS, LSE, MCX, MEX, MIL, MUN, NASDAQ, NEO, NSE, NYSE |
| A10 ticker reuse | CMLGF — delisted and actively trading: earlier listing only; LGMK — delisted and actively trading: earlier listing only; CCO — delisted and actively trading: earlier listing only; IDAU — freed and retaken via symbol-change: latest listing only; QAU — freed and retaken via symbol-change: latest listing only |
| §5 page sizes | delisted-companies: cap 100, mergers-acquisitions-latest: cap 1000 / full pages up to: delisted-companies: 100, cik-list: 10000, symbol-change: 1000, mergers-acquisitions-latest: 1000 |
| §5.4a market-cap date range | complete for a 10-year window |
| economic-indicators names | GDP: 0, realGDP: 0, CPI: 2, inflationRate: 61, federalFunds: 2, unemploymentRate: 2, initialClaims: 13, retailSales: 2, consumerSentiment: 2, industrialProductionTotalIndex: 2 |
| Coverage (US/UK/Canada, unknown symbol) | AAPL: ok, AMGN: ok, QQQ: ok, BRK-B: ok, BRK.B: plan_restricted, ^GSPC: plan_restricted, ^IXIC: plan_restricted, ^DJI: plan_restricted, ^VIX: plan_restricted, ^RUT: plan_restricted, VOD.L: ok, SHOP.TO: ok, ZZZZQX: ok_empty |
| Rate-limit headers | none |

## Row-cap requests (AAPL `15min`)

| Window | Rows | Expected | First | Last |
|---|---|---|---|---|
| 2025-01-01 … 2025-12-31 | 782 | 6,500 | 2025-11-17 | 2025-12-31 |
| 2023-01-01 … 2025-12-31 | 782 | 19,552 | 2025-11-17 | 2025-12-31 |
| 2016-01-01 … 2025-12-31 | 782 | 65,364 | 2025-11-17 | 2025-12-31 |

## `15min` conventions

Days examined: 14. First bar per day (as interpreted): {"09:30": 14}. Early-close days: {"2024-11-29": 14, "2024-12-24": 14}.

## Row-cap requests (AAPL `1hour`)

| Window | Rows | Expected | First | Last |
|---|---|---|---|---|
| 2025-01-01 … 2025-12-31 | 428 | 1,750 | 2025-10-03 | 2025-12-31 |
| 2023-01-01 … 2025-12-31 | 428 | 5,264 | 2025-10-03 | 2025-12-31 |
| 2016-01-01 … 2025-12-31 | 428 | 17,598 | 2025-10-03 | 2025-12-31 |

## `1hour` conventions

Days examined: 14. First bar per day (as interpreted): {"09:30": 14}. Early-close days: {"2024-11-29": 4, "2024-12-24": 4}.

## Reference lists

| Endpoint | Status | Rows | Fields |
|---|---|---|---|
| `stock-list` | ok | 51,920 | companyName, symbol |
| `actively-trading-list` | ok | 35,601 | name, symbol |
| `etf-list` | ok | 11,579 | name, symbol |
| `available-exchanges` | ok | 63 | countryCode, countryName, delay, exchange, name, symbolSuffix |
| `financial-statement-symbol-list` | ok | 32,901 | companyName, reportingCurrency, symbol, tradingCurrency |
| `symbol-change` | ok | 100 | companyName, date, newSymbol, oldSymbol |
| `historical-sp500-constituent` | ok | 1,531 | addedSecurity, addedTicker, changeType, date, dateAdded, reason, removedSecurity, removedTicker, symbol |
| `historical-nasdaq-constituent` | ok | 452 | addedSecurity, addedTicker, changeType, date, dateAdded, reason, removedSecurity, removedTicker, symbol |
| `historical-dowjones-constituent` | ok | 86 | addedSecurity, addedTicker, changeType, date, dateAdded, reason, removedSecurity, removedTicker, symbol |
| `sp500-constituent` | ok | 503 | cik, dateFirstAdded, founded, headQuarter, name, sector, subSector, symbol |
| `nasdaq-constituent` | ok | 101 | cik, dateFirstAdded, founded, headQuarter, name, sector, subSector, symbol |
| `dowjones-constituent` | ok | 30 | cik, dateFirstAdded, founded, headQuarter, name, sector, subSector, symbol |

## Calls

| Group | Label | HTTP | Class | Rows |
|---|---|---|---|---|
| coverage | `coverage:AAPL` | 200 | ok | 5 |
| coverage | `coverage:AMGN` | 200 | ok | 5 |
| coverage | `coverage:QQQ` | 200 | ok | 5 |
| coverage | `coverage:BRK-B` | 200 | ok | 5 |
| coverage | `coverage:BRK.B` | 402 | plan_restricted | 0 |
| coverage | `coverage:^GSPC` | 402 | plan_restricted | 0 |
| coverage | `coverage:^IXIC` | 402 | plan_restricted | 0 |
| coverage | `coverage:^DJI` | 402 | plan_restricted | 0 |
| coverage | `coverage:^VIX` | 402 | plan_restricted | 0 |
| coverage | `coverage:^RUT` | 402 | plan_restricted | 0 |
| coverage | `coverage:VOD.L` | 200 | ok | 5 |
| coverage | `coverage:SHOP.TO` | 200 | ok | 5 |
| coverage | `coverage:ZZZZQX` | 200 | ok_empty | 0 |
| coverage | `delisted:TWTR` | 200 | ok | 22 |
| coverage | `delisted:SIVB` | 200 | ok | 21 |
| coverage | `delisted:BBBY` | 200 | ok | 21 |
| coverage | `delisted:ATVI` | 200 | ok | 21 |
| coverage | `delisted:LEH` | 200 | ok_empty | 0 |
| intervals | `interval:1min` | 200 | ok | 390 |
| intervals | `interval:5min` | 200 | ok | 78 |
| intervals | `interval:15min` | 200 | ok | 26 |
| intervals | `interval:30min` | 200 | ok | 13 |
| intervals | `interval:1hour` | 200 | ok | 7 |
| intervals | `interval:4hour` | 200 | ok | 2 |
| conventions | `conventions:15min:2024-03-08` | 200 | ok | 52 |
| conventions | `conventions:15min:2024-11-01` | 200 | ok | 52 |
| conventions | `conventions:15min:2025-03-07` | 200 | ok | 52 |
| conventions | `conventions:15min:2025-10-31` | 200 | ok | 52 |
| conventions | `conventions:15min:2024-11-27` | 200 | ok | 66 |
| conventions | `conventions:15min:2024-12-23` | 200 | ok | 66 |
| row-cap | `cap:15min:2025-01-01:2025-12-31` | 200 | ok | 782 |
| row-cap | `cap:15min:2023-01-01:2025-12-31` | 200 | ok | 782 |
| row-cap | `cap:15min:2016-01-01:2025-12-31` | 200 | ok | 782 |
| depth | `depth:15min:1998-01-05` | 200 | ok_empty | 0 |
| depth | `depth:15min:2003-01-05` | 200 | ok_empty | 0 |
| depth | `depth:15min:2008-01-05` | 200 | ok | 260 |
| depth | `depth:15min:2005-07-01` | 200 | ok | 234 |
| depth | `depth:15min:2004-04-01` | 200 | ok | 234 |
| depth | `depth:15min:2003-08-01` | 200 | ok_empty | 0 |
| depth | `depth:15min:2003-12-01` | 200 | ok | 260 |
| depth | `depth:15min:2003-10-01` | 200 | ok | 260 |
| split | `split:NVDA:15min` | 200 | ok | 104 |
| conventions | `conventions:1hour:2024-03-08` | 200 | ok | 14 |
| conventions | `conventions:1hour:2024-11-01` | 200 | ok | 14 |
| conventions | `conventions:1hour:2025-03-07` | 200 | ok | 14 |
| conventions | `conventions:1hour:2025-10-31` | 200 | ok | 14 |
| conventions | `conventions:1hour:2024-11-27` | 200 | ok | 18 |
| conventions | `conventions:1hour:2024-12-23` | 200 | ok | 18 |
| row-cap | `cap:1hour:2025-01-01:2025-12-31` | 200 | ok | 428 |
| row-cap | `cap:1hour:2023-01-01:2025-12-31` | 200 | ok | 428 |
| row-cap | `cap:1hour:2016-01-01:2025-12-31` | 200 | ok | 428 |
| depth | `depth:1hour:1998-01-05` | 200 | ok_empty | 0 |
| depth | `depth:1hour:2003-01-05` | 200 | ok_empty | 0 |
| depth | `depth:1hour:2008-01-05` | 200 | ok | 70 |
| depth | `depth:1hour:2005-07-01` | 200 | ok | 63 |
| depth | `depth:1hour:2004-04-01` | 200 | ok | 63 |
| depth | `depth:1hour:2003-08-01` | 200 | ok_empty | 0 |
| depth | `depth:1hour:2003-12-01` | 200 | ok | 70 |
| depth | `depth:1hour:2003-10-01` | 200 | ok | 70 |
| split | `split:NVDA:1hour` | 200 | ok | 28 |
| depth | `depth:daily:AAPL` | 200 | ok | 266 |
| lists | `list:stock-list` | 200 | ok | 51920 |
| lists | `list:actively-trading-list` | 200 | ok | 35601 |
| lists | `list:etf-list` | 200 | ok | 11579 |
| lists | `list:available-exchanges` | 200 | ok | 63 |
| lists | `list:financial-statement-symbol-list` | 200 | ok | 32901 |
| lists | `list:symbol-change` | 200 | ok | 100 |
| lists | `list:historical-sp500-constituent` | 200 | ok | 1531 |
| lists | `list:historical-nasdaq-constituent` | 200 | ok | 452 |
| lists | `list:historical-dowjones-constituent` | 200 | ok | 86 |
| lists | `list:sp500-constituent` | 200 | ok | 503 |
| lists | `list:nasdaq-constituent` | 200 | ok | 101 |
| lists | `list:dowjones-constituent` | 200 | ok | 30 |
| paging | `page:delisted-companies:100` | 200 | ok | 100 |
| paging | `page:delisted-companies:1000` | 200 | ok | 100 |
| paging | `page:delisted-companies:p1` | 200 | ok | 100 |
| paging | `page:cik-list:100` | 200 | ok | 100 |
| paging | `page:cik-list:1000` | 200 | ok | 1000 |
| paging | `page:cik-list:10000` | 200 | ok | 10000 |
| paging | `page:cik-list:p1` | 200 | ok | 100 |
| paging | `page:symbol-change:100` | 200 | ok | 100 |
| paging | `page:symbol-change:1000` | 200 | ok | 1000 |
| paging | `page:symbol-change:10000` | 200 | ok | 5517 |
| paging | `page:symbol-change:p1` | 200 | ok | 100 |
| paging | `page:mergers-acquisitions-latest:100` | 200 | ok | 100 |
| paging | `page:mergers-acquisitions-latest:1000` | 200 | ok | 1000 |
| paging | `page:mergers-acquisitions-latest:10000` | 200 | ok | 1000 |
| paging | `page:mergers-acquisitions-latest:p1` | 200 | ok | 100 |
| windows | `window:splits-calendar:30` | 200 | ok | 76 |
| windows | `window:splits-calendar:90` | 200 | ok | 286 |
| windows | `window:splits-calendar:180` | 200 | ok | 356 |
| windows | `window:dividends-calendar:30` | 200 | ok | 2406 |
| windows | `window:dividends-calendar:90` | 200 | ok | 4000 |
| windows | `window:earnings-calendar:30` | 200 | ok | 2179 |
| windows | `window:earnings-calendar:90` | 200 | ok | 4000 |
| windows | `window:ipos-calendar:30` | 200 | ok | 118 |
| windows | `window:ipos-calendar:90` | 200 | ok | 362 |
| windows | `window:ipos-calendar:180` | 200 | ok | 443 |
| windows | `window:economic-calendar:30` | 200 | ok | 2308 |
| windows | `window:economic-calendar:90` | 200 | ok | 7147 |
| windows | `window:economic-calendar:180` | 200 | ok | 7131 |
| windows | `window:market-cap:1y` | 200 | ok | 250 |
| windows | `window:market-cap:10y` | 200 | ok | 2514 |
| indicators | `indicator:GDP` | 200 | ok_empty | 0 |
| indicators | `indicator:realGDP` | 200 | ok_empty | 0 |
| indicators | `indicator:CPI` | 200 | ok | 2 |
| indicators | `indicator:inflationRate` | 200 | ok | 61 |
| indicators | `indicator:federalFunds` | 200 | ok | 2 |
| indicators | `indicator:unemploymentRate` | 200 | ok | 2 |
| indicators | `indicator:initialClaims` | 200 | ok | 13 |
| indicators | `indicator:retailSales` | 200 | ok | 2 |
| indicators | `indicator:consumerSentiment` | 200 | ok | 2 |
| indicators | `indicator:industrialProductionTotalIndex` | 200 | ok | 2 |
| reuse | `reuse:delisted:0` | 200 | ok | 100 |
| reuse | `reuse:delisted:1` | 200 | ok | 100 |
| reuse | `reuse:delisted:2` | 200 | ok | 100 |
| reuse | `reuse:CMLGF:prices` | 200 | ok | 5000 |
| reuse | `reuse:LGMK:prices` | 200 | ok | 3305 |
| reuse | `reuse:CCO:prices` | 200 | ok | 5000 |
| reuse | `reuse:IDAU:prices` | 200 | ok | 53 |
| reuse | `reuse:QAU:prices` | 200 | ok | 53 |
