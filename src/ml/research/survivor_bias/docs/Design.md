# Design

## Purpose

Measure how much the results recorded on the FMP vintage were helped by its missing delisted stocks. The vintage
holds 0–10% of the stocks that delisted before 2021, so every pre-2021 number on it is survivor-biased. The EODHD
store (`src.data.eodhd_pipeline`) holds 16,670 delisted US common stocks with prices back to 1999, which makes the
bias measurable: run the same frozen rule on the full EODHD universe and on the part of it the vintage also has.

## Rules (frozen 2026-11-22, before any result was seen)

### Universes

All three are subsets of the EODHD store, so prices and adjustment are identical and only membership differs.

| Universe | Members | Stands for |
|---|---|---|
| `full` | every EODHD common stock (`is_common_stock`, plain symbol, no ADR) | the survivorship-free answer |
| `vintage_like` | EODHD stocks mapped to an FMP vintage security (`fmp_security_id` set) | what a vintage backtest sees |
| `survivors` | EODHD stocks still trading on the store date | the textbook survivor-only universe |

Bias = `vintage_like` − `full` (and `survivors` − `full`).

### Delisting rule

A stock is **delisted** when the symbol list marks it inactive and its last bar is more than 10 days before the store
date. Its last close is the exit price. It is **distressed** when any of these holds:

- the symbol is five letters ending in `Q` (bankruptcy suffix);
- the last raw close is below $1;
- the total return over its last 63 sessions is −50% or worse.

Two versions are run for every result:

| Version | Delisted, distressed | Delisted, other |
|---|---|---|
| `last_close` | last close | last close |
| `haircut` | last close × 0.70 (−30%) | last close |

−30% is roughly the average delisting return of performance-related delistings in the literature (Shumway 1997 and
later work on CRSP). Mergers, which make up most of the "other" delistings, usually end near the deal price, so the last
close is a fair exit.

### Monthly panel

- Month-end values are the last bar of each calendar month: total-return close (splits and dividends, `adjust()`
  mode `total`), raw close, 63-session median raw dollar volume (zero-volume days left out).
- A stock can be bought at the end of month m when it traded in the last 7 calendar days of the month, its raw close is
  at least $5, its median dollar volume is at least $5M, and it has at least 63 bars.
- Its return for month m+1 is the next month-end close over this one. If the stock delists inside month m+1, the
  return runs to its last close and the delisting rule applies; the money sits in cash for the rest of the month. If
  its last bar is the last bar of month m and it delists, the month m+1 return is the delisting haircut alone (0% or
  −30%).
- A missing next month while the stock keeps trading later (a gap) gives no return for that month.

### Strategies

| Strategy | Rule | Costs |
|---|---|---|
| `ew_universe` | equal weight, every eligible stock, monthly | none (benchmark) |
| `momentum_decile` | 12-1 return (close 21 bars ago over close 252 bars ago), top 10% of eligible, equal weight, monthly — the 2026-11-11 screen | P24 cost tiers per side on turnover |
| `momentum_p21` | P21 core: 500 most liquid eligible stocks (S&P 500 stand-in), signal = 12-1 return / annualised weekly volatility of the same window (≥ 40 weekly returns, vol ≥ 0.05), hold names ranked ≤ 60, fill to 20 by rank, equal weight, monthly | 10 bps per side on turnover |
| `p24_full`, `p24_drop_only` | P24 daily detector with the frozen live parameters (`LIVE_OVERRIDES`), exits as in P24, delisting rule on trades that end with the delisting | P24 cost tiers |

P21's sector cap, regime overlay, inverse-volatility sizing and earnings blackout need data the store does not have
(sectors, VIX, earnings dates) and are left out; the momentum sort itself is what the survivor question is about.

Quality-value needs fundamentals, which the vintage has only for its own stocks and EODHD does not have at all, so it
cannot be rerun on the full universe. Its benchmark (`ew_universe`) is measured, which shows how much the vintage's
universe return was inflated.

### Periods

Signals are grouped by formation (or signal) date: 2000–2007, 2008–2011, 2012–2021 (the window of the recorded screens),
2022–2026 (the recorded train and holdout years), and 2000–2020 as the long survivorship-free test.

### Sensitivity

Bad prints in tiny stocks can dominate an equal-weight average. Every monthly result is also shown with stock-month
returns clipped at +300%; if the two differ materially the clipped number is not the answer, it is a warning.

### Amendment 1 (2026-11-22, after the first run)

The first run (`results/survivor_bias/20261122-full`, kept) showed the equal-weight `full` universe at +76%/yr in
2000–2007 unclipped and +2% clipped: a few hundred stock-months with returns of 100× to 8,000× dominated it. They come
from stocks whose EODHD raw history is partly already split-adjusted (e.g. YELL: a 1:7,500 cumulative reverse-split
factor applied to prices that already carried part of it), the 12% of stocks that failed the adjustment cross-check.
The `+300%` clip cannot catch the matching false crashes. Rule added, applied to every universe alike:

- A stock-month's return counts only when our total-return return and the return from EODHD's own adjusted close agree
  within 10% (as growth factors); otherwise the stock is skipped that month. A missing vendor value passes.
- The 12-1 momentum return is kept only when the same check passes on the momentum window.

The P24 trades are left as frozen (P24 has its own suspect-gap filter); their median is reported next to the mean.
The clip sensitivity stays.

### Amendment 2 (2026-11-22, after the Amendment 1 run)

Amendment 1 cleaned `vintage_like` (equal-weight 2000–2020 from +25% to +10%/yr) but not `full` (+27%): EODHD's adjusted
close shares the bad raw prints of many stocks the vintage lacks (one-day spikes, a peso-quoted listing falling 99.999%,
"raw" closes of $9,000–37,000 retro-multiplied by later reverse splits). Rule added, applied to every universe alike:

- A day is a bad print when the raw close moves 3× up or to a third and comes back within 1.5× of the previous close
  within 5 bars, or moves 20× either way on a day without a split.
- A stock-month is skipped when the formation month or the holding month contains a bad print (column `suspect`).
- The momentum return is dropped when its 253-bar window contains a bad print.
- P24 is reported twice: as frozen, and `screened` (trades with a suspect month from 12 months before the signal to the
  exit removed).

Bad data is more common among the stocks the vintage lacks (spike or adjustment flag on 25% of them against 9.5%), and
data problems cluster in failing stocks, so the screened numbers are a lower bound on the survivor bias.

## Architecture

```
EodhdStore ──► panel.build_panel ──► monthly panel (parquet cache) ──► portfolios.* ──► stats table
          └──► knife.run_knife (P24 run_security per stock, process pool) ──► trades ──► delisting rule ──► stats table
                                     delisting.classify_end (per stock, shared by both)
jobs/run_survivor_bias.py writes results/survivor_bias/<run_id>/
```

- `config.py` — every number above.
- `delisting.py` — `classify_end`, `terminal_factor`, `apply_to_trades`.
- `panel.py` — month-end panel from the store, one bucket of stocks at a time (54M daily rows never sit in memory).
- `portfolios.py` — the monthly strategies and the period statistics.
- `knife.py` — P24 on the EODHD store (P24's own runner opens the FMP vintage in its workers).

## Design Decisions

- One data source for all universes: comparing EODHD to the real vintage would mix the survivor effect with vendor
  differences (the vintage's spin-off price errors, for instance).
- The stats are computed on monthly portfolio returns (calendar time), not on averages of event returns, except for P24,
  whose recorded result was a trade average; P24 also gets its own slot-limited portfolio (P24 `simulate_portfolio`).
- Results go to `results/survivor_bias/`, aggregates only.
