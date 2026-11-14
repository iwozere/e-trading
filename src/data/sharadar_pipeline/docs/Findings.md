# Sharadar — findings before purchase

Researched 2026-11-14 from Sharadar's public pages, its public API schema, and its free test key.
Nothing here comes from paid data. Items marked **[VERIFY]** could not be confirmed before purchase.

## 1. Why Sharadar

The FMP vintage 2026-10 has almost no stocks that were delisted before 2021, so every backtest before 2021 is
survivor-biased and only about five years are trustworthy. Sharadar is the cheapest source found that has both
delisted daily prices and point-in-time fundamentals in one product.

## 2. The plan considered: Bundle, Full History, one month

| Item | Value |
|---|---|
| Product | Sharadar Bundle, Full History, Personal Use, bought direct at sharadar.com |
| Price | $69 per month, or $499 per year |
| Shorter history | 10 years $49/mo, 5 years $29/mo (not useful: the gap is 1998–2020) |
| Billing | Card is billed automatically; cancelling stops future billing; no refund of money already paid |
| Free trial | None. A free sample exists (section 6) |

The Bundle contains every table. The cheaper single plans split them: Prices ($39/mo full history) has no
fundamentals, Fundamentals ($39/mo) has no daily prices.

## 3. Licence — read this before buying

Source: https://sharadar.com/terms (no effective date on the page; copyright 2026) and the FAQ.

| Topic | What the terms say |
|---|---|
| Who may buy | A natural person, for personal use only. No company, fund or other organisation. Account is personal and non-transferable |
| Allowed use | Own research, backtesting, and automated trading of your own account (stated in the FAQ) |
| Not allowed | Professional or commercial use, trading or investing for others, consulting, publication |
| **Retention** | **Within 30 days of termination you must delete all copies of the data from every system you own or operate: downloads, bulk files, caches, extracts, and any dataset that could reproduce Sharadar tables (Section 10)** |
| What you may keep | Research outputs, backtest results, models, summary statistics and trade logs, as long as they do not contain and cannot reproduce the data |
| Proof | Sharadar may ask for proof of personal use, and for an affidavit confirming deletion |
| Sharing | No publishing, redistributing or sharing of the data, of access, or of derivative data that allows reverse engineering (Section 4) |
| Evaluations | Conclusions from evaluating the data may not be published without Sharadar's written approval (Section 8) |
| Attribution | Public display of data needs "Data from Sharadar.com"; research outputs that do not display data need none |
| Liability | Capped at the amount paid |

### What this means for a one-month purchase

- **One month does not buy a permanent copy.** It buys about 60 days of use: 30 paid days plus 30 days to delete.
  This is the opposite of the FMP purchase, where the vintage is kept.
- Everything that should outlive the subscription must be a derived result (backtest tables, statistics, fitted
  models), not a converted copy of the tables.
- The data must be deleted from the dev machine, the Pi and every backup. It therefore must live in its own data
  root and never be merged into the FMP vintage.
- This repository is public. Never commit Sharadar data, the API key, or (after subscribing) quality-evaluation
  results about Sharadar.
- **[VERIFY]** The terms do not say in so many words that cancelling a monthly plan is "termination". Reading it
  any other way would be unsafe. If a kept copy matters, ask Sharadar in writing before buying, or take the annual
  plan ($499) and keep the subscription running while the data is in use.

This document is a summary, not legal advice.

## 4. Depth and coverage

Counts are from the product pages and from the free `tickers` table pulled on 2026-11-14.

| Table (legacy code) | Content | History from | Coverage |
|---|---|---|---|
| `stocks` (SEP) | Daily OHLCV, active and delisted | 1997-12-31 | 21,037 tickers, 14,726 delisted |
| `fundamentals` (SF1) | Statements and ratios, 6 dimensions | 1998-01 | 17,884 companies, 12,375 delisted |
| `daily` | Daily market cap, EV, P/E, P/B, P/S | 1998-12 | Companies in `fundamentals` |
| `actions` | Splits, dividends, spinoffs, delistings, mergers, ticker changes | 1998-01 | All tickers in the price and fundamentals tables |
| `tickers` | Security master | 1990-06 | All tables |
| `sp500` | Historical S&P 500 membership changes | 1998-01 | — |
| `events` | 8-K material event codes | 2004-01 | — |
| `metrics` | Price-based metrics | 1996-09 | — |
| `funds` (SFP) | Daily fund and ETF prices | 1997-12 | 10,076 tickers |
| `insiders` (SF2) | Forms 3, 4, 5 | 2008-01 | about 16,000 tickers, 200,000 insiders |
| `holdings` (SF3, SF3A, SF3B) | 13F holdings | 2013-06 | 31,000 tickers, 13,000 investors |

Updates run daily (17:30 and 23:30 US Eastern for prices, fundamentals and actions).

### Securities in `stocks`

| Category | Tickers |
|---|---|
| Domestic common stock (single or primary class) | 15,747 |
| Domestic common stock, secondary class | 1,342 |
| ADR common stock (all classes) | 2,297 |
| Canadian common stock (all classes) | 408 |
| Preferred stock (domestic, ADR, Canadian) | 1,243 |

Exchanges: NASDAQ 13,083, NYSE 6,597, NYSE American 1,323, others 34. **OTC stocks are not covered** (6 tickers),
and there is no warrant category. A stock that moved to OTC ends at its exchange delisting.

### Delisted coverage against the FMP vintage

Sharadar delisted common stocks (primary or single class), matched to vintage 2026-10 by CIK. "Covered" means the
vintage has a stock with the same CIK whose prices end within 45 days of Sharadar's last price date.

| Last price year | Sharadar delisted common stocks | Covered by vintage | Missing from vintage |
|---|---|---|---|
| 1998–2011 | 7,402 | 0.1% | 7,396 |
| 2012–2020 | 2,784 | 10.3% | 2,498 |
| 2021–2026 | 2,569 | 69.2% | 791 |

Sharadar lists 250–600 delisted common stocks in every year since 1998 (916 in 1999, 925 in 2000, 412 in 2008,
249 in 2020). Of 5,527 active common stocks in Sharadar, 92.1% have a CIK match in the vintage, which gives a large
overlap for cross-checking the two vendors. The CIK comes from the `secfilings` URL and is present for every row
of the `fundamentals` ticker list.

## 5. Data quality

### What Sharadar claims

- About 99% free of survivorship bias; the remaining gap is malformed old electronic filings.
- "Reference grade": values can be reconciled to the source filings.
- About 99% coverage of acquisition and merger consideration (cash and stock per share) in `actions`.

### Prices (`stocks`)

| Column | Basis |
|---|---|
| `open`, `high`, `low`, `close`, `volume` | Split-adjusted |
| `closeunadj` | Unadjusted (as traded) |
| `closeadj` | Adjusted for splits, cash dividends and spinoffs |

Only the close has all three versions. Raw values are derived:
`open_raw = open × closeunadj / close` (same for high and low) and `volume_raw = volume × close / closeunadj`.

### Fundamentals (`fundamentals`)

| Dimension | Meaning | Use |
|---|---|---|
| ARQ, ART, ARY | As reported; `date` is the SEC filing date; never rewritten by restatements | Backtests |
| MRQ, MRT, MRY | Most recent reported; `date` is the report period; includes restatements | Not for backtests |

- Primary key: `ticker, dimension, date, reportperiod`. About 105 indicator columns.
- Quarterly dimensions exist for US domestic filers only. ADRs and Canadian filers have annual data.
- Only the primary common share class is covered.
- A late filer that reports several periods on one date gets only the most recent period in the AR dimensions.
- The filing date is the 10-K or 10-Q date. Earnings usually reached the market earlier through an 8-K press
  release, so the filing date is a safe (late) availability date, not the announcement date.
- There is no earnings-announcement calendar and no analyst estimates. Earnings-surprise work still needs the FMP
  earnings events.

### Identity and corporate actions

- `ticker` is the current ticker; history is rewired on a ticker change. A delisted company whose ticker was
  reused gets a number appended. `permaticker` is the stable id per share class.
- `actions` has 75 action types. The ones needed: `split` (new shares per old share, effective date), `dividend`
  (cash per share, **split-adjusted**, ex-date), `spinoff` and `spinoffdividend`, `delisted`, `regulatorydelisting`,
  `voluntarydelisting`, `bankruptcyliquidation`, `acquisitionby`, `mergerto`, `acquisitioncash`,
  `acquisitioncommon`, `tickerchangefrom` / `tickerchangeto`, `listed`, `initiated`.
- Delisting rows carry the last trade date and the reason. Acquisition rows carry the cash and stock received per
  share, which allows a real delisting return instead of "last close". FMP has neither. Spinoffs, which FMP lacks
  entirely, are present.

### Known limits

- No OTC history, so the post-delisting OTC life of a stock is absent.
- No intraday data.
- Insiders start 2008 and 13F starts 2013; the project already has its own Form 4 backfill.
- `daily` formulas are not documented on the table page **[VERIFY]** against the `descriptions` table.
- Third-party reviews of Sharadar's accuracy were not found; the claims above are the vendor's own.

## 6. Volume and access

| Fact | Value |
|---|---|
| API base | `https://api.sharadar.com/v1.0` (not Nasdaq Data Link; the `nasdaqdatalink` library does not work with this key) |
| Auth | `api_key` query parameter or `x-api-key` header |
| Query | `GET /data/{table}` with `ticker`, `from`, `to`, `fields`, `limit` (default 10,000), `skip`, `sort`, `format=csv|json` |
| Default window | `from` defaults to one year ago; always pass `from` explicitly |
| Bulk | `GET /data/{table}?years=full` returns HTTP 302 to a pre-signed zip of one CSV |
| Bulk status | add `status=True` to get file name, size and last-modified time without downloading (needs a paid key) |
| Schema | `GET /schema/{table}?format=postgres|sqlite|mysql`, public, no key |
| Rate limit | Response headers on the test key: `X-RateLimit-Limit: 5000`, `X-RateLimit-Weighted-Limit: 25000`. Window length and paid limits **[VERIFY]** |

Bulk file sizes from the documentation examples:

| Table | Zip size |
|---|---|
| `stocks` | about 951 MB (example dated 2026-07-16; history tier of the example not stated) |
| `actions`, full history | about 9.4 MB |
| `tickers` | about 3.7 MB |
| All others | **[VERIFY]** with `status=True` on purchase day |

A full-history download of all tables is a few GB zipped at most. The whole pull is a handful of requests and
should take minutes, not days; there is no bandwidth cap in the terms.

### Free sample (`api_key=test-api-key`)

| Works | Result |
|---|---|
| `tickers`, all tables | Full list, all rows, paged with `skip` |
| `descriptions` | Full field dictionary (430 rows), including all action types and event codes |
| `stocks`, `fundamentals`, `actions` for sample tickers (AAPL, MSFT, about the DJIA) | About the last five years |
| Other tickers (for example LEH) | HTTP 403 "Exceeds free tier" |

This is enough to build and test the downloader, the converter and the vendor cross-check before paying.

Paging gotcha: the default row order is not stable across pages. Paging the unfiltered `tickers` table lost and
repeated rows (9,775 duplicates, AAPL missing). The pipeline reads it per `table` value with `sort=permaticker.asc`.
Bulk files are not affected.

### Dry run on the free sample (2026-11-14)

The pipeline was run end to end on 20 large caps (about five years each): download, build, cross-check, purge.

| Check | Result |
|---|---|
| Securities in the store | 21,037 (14,726 delisted); 8,838 mapped to an FMP security by CIK |
| Our total-return adjustment vs Sharadar `closeadj` | 20 of 20 within 1% on every day; worst day 0.29% |
| Raw close vs FMP raw close | 20 of 20 pairs agree on at least 99% of days; median ratio 1.000 |
| Raw volume vs FMP | Median ratio 0.9996 |
| Splits vs FMP | 4 of 4 match on date and ratio (split date = ex-date, as in FMP) |

This validates the raw-price derivation and the dividend re-basing on liquid names only. Delisted and thinly
traded names can only be checked with a paid key.

## 7. Other providers and what they let you keep

Checked 2026-11-14. "Keep" means what the provider's own published terms say about data after the subscription ends.

| Provider | Price | Keep the data? | Delisted prices | Point-in-time fundamentals for delisted |
|---|---|---|---|---|
| Sharadar Bundle | $69/mo | No: delete within 30 days. Models, backtests, statistics may be kept | 14,726 tickers from 1998, with delisting reason and consideration | Yes, from 1998 |
| EODHD EOD All-World | $19.99/mo | **Yes** for private, non-commercial use; no deletion requirement; no redistribution | 16,691 delisted common stocks on NYSE/NASDAQ/AMEX in its list; depth unverified | No |
| FirstRate Data | One-time, about $300–500 (not shown in USD by the vendor) | **Yes**: perpetual licence; training models on it is explicitly allowed; no redistribution of raw data | 7,000+ from 2000; the vendor says the universe is incomplete | No |
| Norgate Platinum | $630/yr | No: access and use end with the subscription | From 1990 | No |
| MarketParquet | $79 one-time | "Perpetual access"; licence text not verified | About 8,500 from 2000, split-adjusted only | No |
| FinzData | One-time, price per product | Not stated | Claimed, from 2000 | Claimed "as filed" |
| SEC EDGAR | Free | Yes, public data | — | Yes, as filed, about 2009 onward (XBRL); not re-checked today |

Notes:

- **EODHD.** Its help page states that data downloaded while subscribed may be kept for private use and that the
  terms have no deletion requirement. The project already has an EODHD key (free plan). The free key serves the
  delisted symbol list but only one year of prices, so history depth for old delistings (Lehman, Kite) could not be
  checked. A ticker-level comparison with Sharadar's delisted common stocks found the ticker in EODHD's list for 76%
  (1998–2005), 84% (2006–2011), 86% (2012–2017) and 90% (2018–2020). This is an upper bound: a ticker in the list
  does not prove a full price history, and tickers get reused. A third-party review warns that EODHD's pre-2018
  delisted data needs validation. EODHD has no delisting reasons or deal consideration.
- **FirstRate.** The clearest "buy once, keep, train on it" licence found. Weaker coverage of delistings and no
  mergers or spinoffs in its corporate actions.
- **MarketParquet, FinzData.** Neither names its operator or its data source; treat as unverified.
- Nobody in this price range sells kept, point-in-time fundamentals for delisted companies. EDGAR is the only kept
  source, and it starts around 2009.

### What this means

- The Sharadar deletion rule covers the data, not what was learned from it: fitted models, backtest results and
  statistics stay. What cannot stay is a copy to re-train on later.
- A kept price history for delisted stocks is possible from EODHD for $19.99, if its depth holds up. That needs one
  paid month to verify, and Sharadar's free ticker list is the yardstick to measure it against.
- Sharadar remains the only source here for point-in-time fundamentals back to 1998 and for delisting returns, so
  it stays the right choice for the research window.

## 8. Sources

- https://sharadar.com/terms
- https://sharadar.com/llms.txt (plans, prices, API reference)
- https://sharadar.com/bundle
- https://sharadar.com/docs/faqs
- https://sharadar.com/docs/stocks, `/docs/fundamentals`, `/docs/tickers`, `/docs/actions`, `/docs/daily`, `/docs/insiders`
- https://api.sharadar.com/v1.0/schema
- Free API pulls of `tickers` and `descriptions`, 2026-11-14
- https://eodhd.com/help/storage, https://eodhd.com/pricing, EODHD delisted symbol list pulled with the project's key
- https://firstratedata.com/about/FAQ
- https://norgatedata.com/subscribe/subscribe.php
- https://marketparquet.com/llms.txt, https://finzdata.com/survivorship-free-stock-data
- https://frontierledger.ai/strategy-backtesting-evaluation/survivorship-bias-free-equity-universes-data-vendors-compared
