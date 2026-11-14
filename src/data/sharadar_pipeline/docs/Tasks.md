# Tasks

## Implementation Status

### ✅ COMPLETED
- [x] Provider research: licence, depth, coverage, quality, volume, access ([Findings](Findings.md), 2026-11-14)
- [x] Delisted-coverage check of Sharadar against FMP vintage 2026-10 using the free `tickers` table
- [x] Purchase-day [Runbook](Runbook.md)
- [x] Owner accepted the 30-day deletion rule (2026-11-14)
- [x] Pipeline built: client, download, manifest, convert, build, mapping, store, crosscheck, purge, cli
- [x] 66 unit tests on synthetic fixtures, 97% coverage; pyright, mypy and flake8 clean
- [x] End-to-end dry run on the free sample (20 large caps): download, build, cross-check, purge
- [x] Other providers checked for what they let you keep ([Findings](Findings.md) §7)

- [x] `eodhd-check` command: depth of EODHD's delisted history against Sharadar's free ticker list (2026-11-14);
      unit-tested, mechanics run once on the free EODHD key (which serves one year only, so no result yet)

### 🔄 WAITING FOR THE OWNER
- [ ] Buy one EODHD EOD month ($19.99), then run `eodhd-check` ([Runbook](Runbook.md))
- [ ] Confirm the research queue below and its order

### 🚀 PLANNED — if the EODHD check passes
- [ ] EODHD delisted-price downloader and a kept store (prices, splits, dividends), cross-checked against the FMP vintage

### 🚀 PLANNED — before purchase
- [ ] Research scripts for the queue below, written against the store interface and run once on the FMP vintage
- [ ] Fundamentals as-of join helper (latest AR row per security and date) for the factor screens

### 🚀 PLANNED — on purchase day (needs a paid key)
- [ ] Confirm the bulk zip layout (one CSV per zip, same columns as the query CSV) and file sizes
- [ ] Full build timing and memory on `stocks`, `daily` and `holdings`
- [ ] Cross-check on delisted and thinly traded names

### 🚀 PLANNED — research queue for the access window

Rules are frozen before the data arrives. Period for all: 1998–2020 on Sharadar, survivorship-free.

| # | Question | Why it needs Sharadar |
|---|---|---|
| 1 | How large is the survivor bias in the 2012–2021 results already recorded (quality-value, momentum, falling knife)? | Calibrates every earlier "upper bound" number |
| 2 | Cash quality-value top 40, frozen rule of 2026-11-11 | Failed the 2024–26 holdout; 23 clean years decide whether it is dead or was unlucky |
| 3 | Dividend cut of 10% or more as an exit rule | Same sign on holdout but weak; needs more events |
| 4 | P24 falling knife gate | Failed its holdout gate; delisted names are exactly the knives that never recovered |
| 5 | P21 momentum tracks | Momentum crashes (2001, 2009) are outside the vintage |
| 6 | Delisting return by reason (acquired, bankrupt, regulatory) | Replaces the "last close" assumption in every backtest |
| 7 | Insider cluster buys, 2008–2020 | Longer and delisted-inclusive history than the Form 4 backfill |

Earnings-surprise work stays on FMP: Sharadar has no announcement dates or estimates.

## Known Issues
- Only USD cash and plain stock consideration are interpreted for the delisting return; foreign-currency cash,
  preferreds, elections and estimates set `other_consideration` and give no return.
- The free sample covers liquid large caps only; nothing delisted has been through the converter with real data.
- The bulk zip layout is assumed (one CSV per zip, query columns); the build stops with a clear error otherwise.
- Licence wording does not explicitly equate cancelling with termination (treated as termination).
- Paid rate limits and most bulk file sizes are unknown until a paid key exists.
- No OTC prices: the post-delisting life of a stock is absent.

## Testing Requirements
- [x] Unit tests for raw-price derivation, dividend re-basing, delisting return, id mapping
- [x] P24 helpers (`load_securities`, `is_common_stock`, `load_daily_batch`) run on `SharadarStore`
- [x] Purge test on a scratch data root
- [ ] Reader parity on real delisted names (paid key)

## Documentation Updates
- [ ] Replace **[VERIFY]** items in Findings with measured values on purchase day (in the data root copy, not git)
