# Pipeline Specification: P24 Falling Knife Reversal (ADX / PPO / CCI)

**Version:** 1.0
**Date:** 2026-11-05
**Audience:** an implementing agent working in this repo (`C:\dev\cursor\e-trading`)
**Source method:** `docs/11-chapter.html`, a Russian summary of chapter 11, "Explosive Profits Catching Falling Knives", of
*Explosive Stock Trading Strategies* (pp. 251–285). Read that file for the original wording and examples. This spec turns the
chapter's visual rules into exact rules a program can test.
**Mode:** research backtest first (Phase 1). A live scan with paper alerts comes after a go/no-go gate (Phase 2). The pipeline
never places orders.

---

## 0. READ FIRST

1. **The chapter is discretionary and its evidence is anecdotal.** The author says more than 80% of trades win, but gives
   nine hand-picked examples, mostly from the 2008–2009 bottom. Nothing in the chapter is a statistical test. The first job
   of this pipeline is to **test that claim**, not to assume it. Phase 2 (live scan) is built only if Phase 1 passes the gate
   in §11.
2. **Survivorship bias works against this strategy more than any other.** Knives that went to zero (bankruptcies,
   delistings) are exactly the losers the method has to avoid. A universe without them inflates the win rate. The FMP
   vintage has thin coverage of names delisted before 2021 (measured for P22: about 9% of pre-2021 exits). The backtest
   must **measure and report delisted coverage per year** and must state the residual bias in every report (§7.4).
3. **The visual words need exact definitions.** "Mountain bulge", "valley bulge", "narrow channel", "narrowest point",
   "clear divergence" are defined in §4 with parameters. Defaults are a starting point. They are calibrated **only on the
   training period** (§7.3). Do not tune them to make the book's examples fire.
4. **Repo lessons that apply here:** P17 (penny stocks) and P19 alerts lost money in the signal ledger once costs were
   counted. Small-cap and sub-$5 knives have wide spreads. The cost model (§7.2) is tiered by price and is mandatory.

---

## 1. Strategy summary (from the chapter)

A "falling knife" is a stock that has:
1. dropped at least **40%** from its last significant top;
2. moved down strongly in one direction;
3. a strong, persistent downward momentum trend;
4. price and trend indicators moving together (price makes new lows and so do the indicators).

A reversal is likely when three things happen at about the same time:

| Indicator | Role | Shape at the turn |
|---|---|---|
| **ADX** (direction strength) | the down move weakens | rises above 30, then flattens and starts to fall: the **"mountain bulge"** |
| **PPO** (trend) | the downtrend turns | negative, goes above its signal line (histogram rises toward 0), then flattens and turns up: the **"valley bulge"** |
| **CCI** (momentum) | momentum shifts | **positive divergence**: price makes a lower low, CCI makes a shallower trough than at the last intermediate support |

Viewed as stacked panels (PPO above ADX), the PPO valley bends down and the ADX mountain bends up. The gap between them is the
**channel**. The narrower the channel, the more likely the reversal. The narrowest point marks the signal bottom ("point 2").
Entry is on the next trading day.

**Entry checklist (all required):** drop ≥ 40%; ADX mountain above 30; PPO valley with the histogram moving toward zero;
narrow ADX/PPO channel whose narrowest point is at the signal low; positive CCI divergence measured against the **last
intermediate support between the top and the signal low** (point 3); entry next day.

**Exits:** CCI ≥ +200 → exit next day. If CCI never reaches +200 → exit on bearish CCI divergence (price makes a higher
high, CCI makes a lower peak).

**Post-signal shakeout:** if price falls below the signal low after entry, stay in only if the CCI trough at the shakeout is
shallower than the CCI trough at point 3 (the minimum support before the signal). If CCI breaks below that trough, exit.
A widening ADX/PPO channel after the signal also points to a shakeout, not a continuation.

**Other rules:** "If in doubt, opt out". A neutral (flat) CCI at a double bottom is **not** a divergence. A signal from one
indicator alone is not a trade. The method does not apply to stocks moving sideways in a range. Weekly charts give fewer
false signals and are recommended for beginners. Indices need the same 40% drop.

---

## 2. Scope and phases

| Phase | Deliverable | Gate to the next phase |
|---|---|---|
| 0 | Indicator + geometry library with unit tests (§4, §12) | All tests green, pyright + mypy clean |
| 1 | Historical backtest on the FMP vintage, daily and weekly, with ablations, control baseline, cost tiers, train/holdout split, report (§6–§8) | Go/no-go criteria in §11, **reviewed by the owner** |
| 2 | Daily + weekly live scan, charts, paper positions, notifications, signal-ledger harvester, scheduler registration (§9–§10) | Owner approval |
| 3 (optional) | Index mode (SPY/QQQ/DIA, §5.4), multi-timeframe confirmation mode, intraday-on-the-way-down trades (§5.5) | — |

Out of scope: live orders, shorting, options, intraday bars, fundamentals and sentiment (the chapter calls message-board
panic a contrarian hint, but not a trade input).

---

## 3. Data

### 3.1 Sources (reuse existing code, no new data access paths)

| Use | Source | Code |
|---|---|---|
| Backtest daily OHLCV, including delisted names | FMP vintage `2026-10` (active vintage from `current.txt`) | `src/data/fmp_pipeline/datamanager_adapter/fmp_store.py`: `FmpStore.securities()`, `FmpStore.bulk_daily(ids, start, end)`, `FmpStore.events_many(ids)`; adjust with `src/data/fmp_pipeline/adjust/adjust.py::adjust(..., mode="split")`. See how P22 uses it: `src/ml/pipeline/p22_biotech_ma/ingest/fmp_vintage.py` |
| Live scan daily OHLCV | DataManager (Yahoo path, cached) | `DataManager.get_ohlcv_batch(symbols, "1d", start, end)` in `src/data/data_manager.py` |
| Live universe | Nasdaq Trader symbol files | `src/ml/pipeline/shared/universe_downloader.py` (FTP→HTTPS fallback, about 6,900 tickers) |
| Indices (Phase 3) | yfinance through DataManager (`^GSPC`, `^IXIC`, `^DJI`; FMP marks `^` indices plan-restricted) or the ETFs SPY/QQQ/DIA from FMP | DataManager |
| Benchmark | SPY, VT | FMP vintage / DataManager |

### 3.2 Price basis

- **Signals and the 40% drop:** split-adjusted prices (`mode="split"`). Total-return adjustment makes old prices
  artificially lower and distorts the drop percentage on high-yield names. Indicators are computed on the same series.
- **Trade returns:** split-adjusted prices plus cash dividends paid while holding (or `mode="total"`; pick one and
  document it in the run manifest).
- **Weekly bars:** resampled from daily bars, `W-FRI`, using only completed weeks (O = first open, H = max, L = min,
  C = last close, V = sum). A week is complete after its last trading session closes; holiday weeks with fewer sessions
  are still valid.
- Use `pandas_market_calendars` (`XNYS`) for sessions. Do not infer trading days from weekdays.

### 3.3 Known data hazards (must be handled, not ignored)

1. **Unadjusted corporate events in FMP.** Spin-offs are missing from FMP (for example SITC 2024-09-23, AIV Dec 2020), and
   some `1d_raw` stretches are already split-adjusted. Both create fake cliffs that look like a 40% knife. Rules:
   - drop bars flagged suspect (`flags` bitmask; `exclude_suspect` semantics);
   - flag any candidate whose drop contains **one bar with a close-to-close fall ≥ 30% on volume below 1.5× its 20-bar
     median**. Report these separately in the backtest (`suspect_gap=True`) and exclude them from the main statistics.
     Real crashes come with heavy volume; data cliffs usually don't.
   - read `<data_root>/overrides/price_basis.csv` through `FmpStore` (automatic).
2. **Delisting while in a position.** Exit at the last available close and tag the trade `exit_reason="delisted"`. Also
   report a stress case where delisted trades that were not acquisitions count as −100%. The vintage has `delisted_date`;
   if the reason is unknown, treat it as non-acquisition in the stress case.
3. **Reused tickers and renames.** Key everything on `security_id`, never on the symbol. Use `FmpStore.resolve(symbol,
   as_of=...)` when mapping symbols.
4. **Stale prints / zero volume.** Skip bars with volume = 0 when computing liquidity. Require at least 90% non-zero-volume
   bars in the indicator warm-up window.

### 3.4 Universe

- **Backtest:** all US common stocks in the FMP vintage, active and delisted. Exclude ETFs, ETNs, funds, preferreds,
  warrants, rights, units, and (by default) ADRs. Reuse the common-listing filter logic from P22
  (`is_common_listing`) if it fits; otherwise write `universe.py` with the same idea.
- **Point-in-time filters, evaluated on bar t using only data up to t:**
  - at least `min_history_bars` of history (default 300 daily bars, so ADX/PPO/CCI are warmed up and the 252-bar peak
    lookback is filled);
  - 20-bar median dollar volume ≥ `min_dollar_volume` (default $1M daily). Weekly uses the daily figure on the week's
    last session;
  - close ≥ `min_price` (default **$1.00**). The book trades sub-$1 names (PWER $0.35). The backtest also runs a separate
    `min_price=0.10` variant and reports the sub-$1 bucket on its own. Live default stays $1.00 until the data says
    otherwise.
- **Live scan:** the `universe_downloader` list, same filters.

---

## 4. Indicator and geometry definitions (exact)

All computations happen on bar t after it closes and use only bars ≤ t. Timeframe-specific defaults are shown as
`daily / weekly`. Every number below is a config value in `config.py` (no literals in code).

### 4.1 Indicators

| Name | Definition | Default |
|---|---|---|
| `ADX`, `+DI`, `−DI` | Wilder ADX, TA-Lib `ADX`, `PLUS_DI`, `MINUS_DI` (TA-Lib 0.8.3 is in `.venv`) | period 14 / 14 |
| `PPO` | `100 * (EMA_fast − EMA_slow) / EMA_slow`, EMA = `ewm(span=n, adjust=False)` on close | 12, 26 |
| `PPO_signal` | EMA of PPO | 9 |
| `PPO_hist` | `PPO − PPO_signal` | — |
| `CCI` | `(TP − SMA(TP, n)) / (0.015 * MeanAbsDev(TP, n))`, TP = (H+L+C)/3; TA-Lib `CCI` | 20 / 20 |

**Do not use TA-Lib `PPO`.** Its default `matype=0` is an SMA, not an EMA. Implement PPO yourself and unit-test it against a
hand-computed series. The book's argument for PPO over MACD is that PPO is in percent, so values are comparable across price
levels and over a 50%+ drop. Keep it in percent.

### 4.2 Reference peak (point 1) and drop

- Swing points use a ZigZag on **closes** with reversal threshold `zz_pct` (default **25%**). A swing high is confirmed
  only after price has closed `zz_pct` below it, so it is causal.
- **Reference peak P1** at bar t = the most recent confirmed swing high before t. If there is none in the last
  `peak_lookback` bars (252 / 104), use the highest close in that window.
  - Why: the chapter measures the drop from the "last significant top". For repeated knives on the way down (VMW trade 2),
    it is measured from the **intermediate** top, i.e. the top of the last counter-trend rally. A 25% threshold keeps
    small bounces from resetting the peak (the Dow example kept 11,800 as the reference through a volatile bounce).
- **Knife low L2** = lowest close in `(P1, t]`; `bar_L2` its bar.
- **Drop** = `(close[P1] − close[L2]) / close[P1]`. Requires `drop ≥ min_drop` (default **0.40**).
- **Freshness:** `t − bar_L2 ≤ max_bars_since_low` (default 5 / 3). If the low is older, the stock is consolidating, and the
  chapter says the method does not apply there.

### 4.3 Direction check

At the bar where ADX peaked in `(P1, t]`: `−DI > +DI`. This ensures the ADX strength came from the down move. The chapter
takes the direction as given; we check it.

### 4.4 ADX mountain bulge

Let `adx_peak` = max ADX over `(P1, t]` and `bar_X` its bar ("point X").

All of:
1. `adx_peak ≥ adx_min_peak` (default **30**);
2. ADX started rising before the drop accelerated: `adx_peak − min(ADX over [P1, bar_X]) ≥ adx_min_rise` (default 8);
3. the mountain has formed: `1 ≤ t − bar_X ≤ adx_max_bars_after_peak` (default 10 / 6), and
   `ADX[t] ≤ adx_peak − adx_min_decline` (default 0.5 points) **or** the ADX slope over the last
   `slope_bars` (3 / 2) bars is ≤ `adx_flat_slope` (default 0.25 points per bar);
4. `ADX[t] ≥ adx_min_peak − adx_floor_margin` (default margin 8). Once ADX has collapsed well below 30 the setup has
   passed.

### 4.5 PPO valley bulge

Let `ppo_trough` = min PPO over `(P1, t]` and `bar_Y` its bar ("point Y").

All of:
1. `PPO[t] < 0`;
2. histogram turned toward zero: `PPO_hist[t] > min(PPO_hist over (P1, t])` and `PPO_hist[t] > PPO_hist[t−1]`; and
   `PPO[t] > PPO_signal[t]` **or** `PPO_hist[t] ≥ hist_near_zero_frac * min(PPO_hist over (P1, t])` (default 0.5, i.e. the
   histogram has recovered at least half way to zero);
3. valley formed: `1 ≤ t − bar_Y ≤ ppo_max_bars_after_trough` (default 10 / 6), and `PPO[t] ≥ PPO[t−1]`;
4. **depth (lower third):** let `ppo_floor` = min PPO over the last `ppo_range_bars` bars (504 / 260), including t. Require
   `ppo_trough ≤ ppo_lower_frac * ppo_floor` (default `ppo_lower_frac = 2/3`). The chapter states this explicitly for indices
   ("valley in the lower third of the range below zero") and uses it for VMW. Here it is the default for stocks too
   (switched off in one ablation).

### 4.6 Channel width (the "narrow channel")

The chapter judges the gap between two panels with different scales by eye. Normalize both to [0, 1] over the same
context window (`ppo_range_bars`):

```
a_t = ADX_t / max(ADX over context window)              # 1 = mountain at its highest
p_t = PPO_t / abs(ppo_floor)                            # −1 = valley at its deepest (ppo_floor < 0)
W_t = (1 − a_t) + (1 + p_t)                             # 0 = narrowest possible; larger = wider
```

If `ppo_floor ≥ 0` there is no valley: no setup.

Conditions:
1. **Narrow:** `min(W over (P1, t]) ≤ w_narrow` (default **0.60**, calibrated in §7.3);
2. **Convergence in time:** `|bar_X − bar_Y| ≤ max_xy_offset` (default 5 / 3), so the mountain and the valley are the same
   event;
3. **Narrowest point reached (causal):** `bar_Wmin` = bar of `min(W over (P1, t])`, with `1 ≤ t − bar_Wmin ≤ 5 / 3`, and
   `W_t ≥ W_{t−1} − w_eps` (default 0.02). In words: the width has stopped shrinking. This is the chapter's own real-time
   test ("watch the channel width; when it stops decreasing noticeably, the narrow point has been reached");
4. **Point 2 is at the narrow point:** `|bar_L2 − bar_Wmin| ≤ max_low_to_wmin` (default 5 / 3).

Keep `W_t` in the output; the shakeout rule (§5.3) and the chart (§9.3) use it.

### 4.7 Intermediate support (point 3) and CCI positive divergence

- **Swing lows** inside `(P1, bar_L2)` use a smaller ZigZag on closes with threshold `support_bounce_pct` (default 8% /
  10%). A swing low counts as support only if price then bounced at least that much.
- **Point 3 (S3)** = the most recent such swing low before `bar_L2`, with `close[S3] > close[L2]` (price made a lower low
  into point 2). It must lie after P1. If there is none, there is no divergence to measure: **no signal.**
  - Why "most recent support", not the top of the bounce: the chapter calls comparing with the bounce top the most common
    mistake, because then no divergence shows up.
  - If a previous P24 signal already happened in this knife (ATHR daily, point 2 vs point 3), point 3 must lie **after**
    that earlier signal's low. The earlier signal "used up" its ADX/PPO turn.
- **CCI troughs:** `cci_3 = min(CCI over [S3 − j, S3 + j])` and `cci_2 = min(CCI over [bar_L2 − j, min(bar_L2 + j, t)])`,
  `j = cci_trough_window` (default 2 / 1).
- **Positive divergence**, all of:
  1. `cci_3 ≤ cci_stretch` (default **−100**): the rubber band was stretched at the support;
  2. `cci_2 − cci_3 ≥ cci_div_min` (default **25 points**). This is the "a flat CCI is not a divergence" rule (ATHR daily
     first attempt: the troughs were level, so no trade);
  3. `close[L2] < close[S3] * (1 − price_div_min)` (default 0.02): the lower low in price is real, not a double bottom.

### 4.8 Signal

A **P24 signal** fires on bar t (`signal_bar`) when §3.4 filters, §4.2 (drop + freshness), §4.3, §4.4, §4.5, §4.6 and §4.7
all hold.

- **One signal per knife:** after a signal, the same security can signal again only after either a new reference peak
  (§4.2), or a new knife low below the signal low **and** a channel that widened by `w_rearm` (default 0.30) and then
  narrowed again. VMW trade 2 and ATHR daily trade 2 are examples of the second case.
- **Partial setups ("forming"):** keep the per-condition booleans. Phase 2 lists stocks where the drop + ADX + PPO
  conditions hold but CCI or the narrow point is still missing, as a watchlist.
- **Score (for ranking only, not a filter):** `score = (w_narrow − W_min) + (cci_2 − cci_3) / 100 + (drop − min_drop)`.
  Used when there are more signals than free slots (§6.3).

### 4.9 Exits

Evaluated on each closed bar while a position is open; execution on the next bar's open.

| Priority | Rule | Default |
|---|---|---|
| 1 | **Delisted**: last available close | — |
| 2 | **Shakeout test failed** (§5.3) | on |
| 3 | **CCI overbought**: `CCI[t] ≥ cci_exit` | +200 |
| 4 | **Bearish CCI divergence**: price closes at a new swing high (ZigZag `support_bounce_pct` on highs since entry) above the previous swing high since entry, while the CCI peak at the new high is lower than at the previous one by ≥ `cci_bear_div_min` (default 25). The chapter: signals must be obvious; small wobbles in the CCI peak don't count. | on |
| 5 | **Time stop**: `bars_held ≥ max_hold_bars` (none in the chapter; added for capital lockup) | 126 / 52 |
| 6 | **Hard stop** (none in the chapter; A/B tested, §7.5): close < entry × (1 − `hard_stop_pct`) | off; A/B at 30% |

---

## 5. Trade logic details

### 5.1 Entry

- Default fill: **next bar's open** after `signal_bar`, plus the cost of §7.2.
- Sensitivity variant: next bar's midpoint `(H+L)/2`. The chapter says "often at the middle of the day's range"; it cannot
  be executed exactly, so it is a report variant only.
- If the next bar is missing (halt, delisting), no trade; log it.

### 5.2 Holding requirements are stricter than entry

The chapter (VMW hypothetical): once in a trade, a renewed fall needs a **clear, strong** positive divergence to stay in;
neutral means exit. §5.3 puts this into rules.

### 5.3 Post-signal shakeout rule

Triggered when, while holding, `close[t] < close[L2]` (below the signal low):
1. Find the current CCI trough `cci_s = min(CCI over the bars since the close fell below close[L2])`.
2. Reference: `cci_3`, the CCI trough at **point 3** (the minimum support before the signal), not `cci_2`. The chapter's
   Nasdaq example: a bearish divergence against point 2 does not count, because point 2 was already the turn.
3. **Stay** if `cci_s > cci_3 + cci_shakeout_margin` (default 10). Otherwise **exit** on the next open
   (`exit_reason="shakeout_failed"`).
4. Report-only flag: channel widening `W_t − W_min ≥ w_widen` (default 0.25) at the shakeout. Log it with the trade. If the
   backtest shows it separates shakeouts from continuations, a later version can make it a rule.

### 5.4 Index mode (Phase 3)

Same rules on SPY, QQQ, DIA (or `^GSPC`, `^IXIC`, `^DJI` from yfinance) with these extra requirements from the chapter:
- the reference top is a clear peak held for at least `index_top_min_bars` (40 daily / 8 weekly), formed on low
  volatility: 20-bar ATR% at the top ≤ its 252-bar median. Peaks in periods of extreme volatility do not count;
- `adx_peak ≥ 30` and the PPO lower-third rule are **mandatory** (no ablation);
- drop ≥ 40% with no exceptions. The chapter: on smaller pullbacks, ADX/PPO/CCI signals are false and cost money.
Expect very few signals (2008–09, maybe 2000–02, 2020 does not reach 40% on SPY). This mode is a curiosity, not a sleeve.

### 5.5 On-the-way-down trades (Phase 3)

The chapter trades counter-trend bounces during a long fall, but only at bars that meet **all** knife criteria, including
40% from the intermediate top. §4.2 (intermediate peak) and §4.8 (re-arm) already allow this. Phase 3 only adds reporting:
knife sequence number per security, and statistics for the 1st vs 2nd+ signal in the same decline.

### 5.6 Multi-timeframe confirmation (Phase 3, optional mode)

The chapter: if the daily CCI divergence is unclear, look at the weekly chart. If there is no converging weekly mountain and
valley with a narrow channel, skip, or wait for the weekly setup. Mode `mtf_confirm`: a daily signal counts only if the
weekly channel at the last completed week has `W ≤ w_narrow_weekly_confirm` (default 0.9) and the weekly ADX ≥ 25. Tested as
its own variant against plain daily and plain weekly.

---

## 6. Backtest engine (Phase 1)

### 6.1 Approach

Event-driven, per security, in plain pandas/numpy. Signals and exits need path state (swing points, re-arm, shakeout), so
a fully vectorized vectorbt `from_signals` is a poor fit. Pre-compute indicators vectorized, then walk bars in a small
state machine. Use numba only if profiling shows it is needed. Securities are independent until the portfolio layer
(§6.3), so run them in a process pool (Windows `spawn`: keep worker functions at module level).

### 6.2 Trade-level output (primary)

Each signal produces one trade row, independent of capital limits. This answers "does the signal work?" before any
portfolio effects:

`security_id, symbol, timeframe, signal_bar_date, entry_date, entry_price, exit_date, exit_price, exit_reason, bars_held,
gross_return, net_return, max_adverse_excursion, max_favorable_excursion, drop_pct, peak_date, peak_close, l2_date,
l2_close, s3_date, s3_close, adx_peak, ppo_trough, ppo_floor, w_min, cci_2, cci_3, score, price_bucket,
dollar_volume_20, suspect_gap, shakeout_seen, shakeout_widening, knife_seq, delisted_in_trade, train_or_holdout`

### 6.3 Portfolio layer (secondary)

Signals cluster heavily in crashes (Oct 2008, Mar 2009, Mar 2020). The portfolio simulation shows what that does:
- equal weight, `max_positions` (default 20), `position_pct = 1 / max_positions`, cash earns 0;
- when there are more signals than free slots on a day, rank by `score` (deterministic tie-break: `security_id`);
- report max concurrent signals per month and the share of all trades from the top 3 months.

### 6.4 Control baselines (mandatory)

Without a control, a high win rate after a 40% drop could just be ordinary mean reversion, not the indicators.
1. **Drop-only control:** enter on the first bar where drop ≥ 40% and freshness hold (§4.2). Same exits.
2. **Matched random control:** for each P24 signal, pick a random bar from the same security's knife window
   (`(P1, t + 20]`) where drop ≥ 40%, and enter there. Same exits. Repeat 200 times with fixed seeds and report the
   distribution.
3. **Market:** SPY buy-and-hold over each trade's holding period (trade-level excess return), and over the whole period
   for the portfolio.

---

## 7. Validation and statistics

### 7.1 Metrics (trade level, net of costs, by timeframe)

Count, win rate, mean, median, trimmed mean (5%), profit factor, average win / average loss, mean bars held, MAE
distribution, return per bar held, share of trades at −50% or worse, exit-reason breakdown, excess vs SPY. Also block
bootstrap 95% CIs (blocks by calendar month, since signals cluster) for the mean and the win rate.
Portfolio level: CAGR, max drawdown, Sharpe on daily returns, exposure %, turnover.

### 7.2 Cost model (per side, tiered by entry price)

| Price | Cost per side (spread + slippage + fee) |
|---|---|
| ≥ $10 | 0.15% |
| $5–10 | 0.30% |
| $1–5 | 0.75% |
| < $1 | 2.00% |

Plus a liquidity cap: position size ≤ 5% of the 20-bar median daily dollar volume (portfolio layer only). Reports also show
0.5×, 2× and 3× costs. Results that turn negative at 2× are flagged fragile.

### 7.3 Train / holdout split and calibration

- **Train:** 2004-01-01 – 2016-12-31. **Holdout:** 2017-01-01 – vintage end. The holdout is evaluated **once**, after
  parameters are frozen, and written to a separate report section. No tuning after looking at it.
- **Calibrated parameters (train only):** `w_narrow`, `cci_div_min`, `zz_pct`, `support_bounce_pct`,
  `max_bars_since_low`. Everything else stays at the book's value or its default. Use a coarse grid (no Optuna), at most
  about 100 combinations per timeframe. Pick a **plateau**, not the single best cell: the chosen cell's neighbours must
  score within 25% of it. Objective: median net trade return, with at least 100 train trades. Save the grid as CSV and a
  heatmap for (`w_narrow`, `cci_div_min`).
- Report the number of grid cells tried with the result (multiple-testing context).

### 7.4 Survivorship report (mandatory, top of every report)

Per year: number of securities in the vintage, number delisted that year, number of delisted that have daily price files,
and an estimate of coverage. Also: share of P24 trades in names that later delisted, and their returns. If delisted coverage
before 2021 is too thin (owner decides; suggest < 50%), the headline result uses 2021+ only and the earlier years are
shown as "optimistic upper bound".

### 7.5 Ablations and variants (same costs, same period)

1. Drop only (= control 1)
2. + ADX mountain
3. + PPO valley
4. + narrow channel (§4.6)
5. Full system (+ CCI divergence)
6. Full system without the PPO lower-third rule
7. Full system with hard stop 30%
8. Full system without the shakeout rule (hold through)
9. Daily vs weekly vs `mtf_confirm` (Phase 3)
10. `min_price` 1.00 vs 0.10; price buckets reported separately

This shows which conditions earn their place. If the full system is not clearly better than control 1, the indicators add
nothing and that is the main finding.

### 7.6 Book-example regression check (sanity, not a target)

Check whether each example is in the vintage and whether the detector fires near the book's date. Dates are approximate,
reconstructed from the chapter. Report hit/miss with the reason (missing data, which condition failed). **Do not change
parameters to hit these.** A miss is information, not a bug.

| Example | Timeframe | Approx. signal / entry | Entry price | Book's exit |
|---|---|---|---|---|
| AONE (A123 Systems, delisted 2012) | daily | early Sep 2010 | $6.75 | $9.50, 2010-09-20, CCI > 200 |
| PWER (Power-One, acquired 2013) | daily | early 2009 | ≈ $0.40 | ≈ $1.15, bearish divergence |
| MPWR | weekly | late 2008 | ≈ $9.00 | $23–24 about 6 months later |
| ATHR (Atheros, acquired 2011) | daily | early Dec 2008 | ≈ $12.00 | just below $16, about 1 month |
| ATHR | weekly | Feb 2009 (between points 3 and 4) | ≈ $12 | ≈ $18, early Apr 2009, CCI > 200 |
| VMW #1 | daily | Oct 2008 | $33 | ≈ $40 within 15 days, CCI > 200 |
| VMW #2 | daily | Nov 2008 | ≈ $20 | $34 about 1 month later, CCI > 200 |
| DIA / ^DJI (index mode) | daily | Mar 2009 | ≈ 6,700 | ≈ 8,200, bearish divergence |
| QQQ / ^IXIC (index mode) | weekly | about Dec 2008, shakeout in Mar 2009 | ≈ 1,500 | ≈ 2,100, bearish divergence |

---

## 8. Phase 1 outputs

```
results/p24_falling_knife/backtest/<run_id>/
    manifest.json            # git hash, config snapshot + hash, vintage id, date range, package versions, seeds
    trades_daily.csv         # §6.2, all variants (column `variant`)
    trades_weekly.csv
    controls.csv             # control 1 + random-control distribution summaries
    portfolio_daily.csv      # equity, positions, cash per day for the full-system variant
    calibration_grid.csv
    survivorship.csv         # §7.4
    examples.csv             # §7.6
    report.md                # human-readable summary: survivorship warning first, then train results, holdout, ablations, costs
    charts/                  # equity curves, return histograms, calibration heatmap, 20 sample trade charts (§9.3)
```

`run_id` = `YYYYMMDD-HHMMSS`. `results/` is the repo-wide root; follow the existing `results/<pipeline>/...` layout.

---

## 9. Live scan (Phase 2, only after the §11 gate)

### 9.1 Jobs

| Job | Schedule (UTC) | What it does |
|---|---|---|
| `P24 Falling Knife Daily Scan` | `30 22 * * 1-5` (after the US close incl. DST; check against existing P05/P21 crons and move if they collide) | daily signals + forming watchlist + paper exits |
| `P24 Falling Knife Weekly Scan` | `0 12 * * 6` | weekly signals on the completed week |

Register both in a new `src/data/pipeline/specs/p24_specs.py` (`PluginSpec`, category `p24`, like
`signal_ledger_specs.py`), then add the module to the registry the same way the other spec modules are loaded. Deploy is
`python -m src.data.pipeline.register_jobs` on the Pi. Register them **disabled** until the owner enables them.

> **As built (2026-11-06):** daily cron `45 22 * * 1-5` (22:30 already hosts three jobs); outputs carry a timeframe
> suffix (`signals_daily.csv`, `signals_weekly.csv`, …) because the weekly folder date is also a daily folder; paper
> state is replayed from the signal each run; Telegram gets a text message plus one photo message per chart.

### 9.2 Daily folder

```
results/p24_falling_knife/YYYY-MM-DD/
    signals.csv              # timeframe, ticker, security/symbol, signal_bar_date, close, drop_pct, adx_peak, ppo_trough, w_min, cci_2, cci_3, score, suggested_entry (= next open), price_bucket
    forming.csv              # partial setups (§4.8): which conditions hold, which are missing
    paper_positions.csv      # open paper trades and their current exit-rule state (CCI, shakeout status)
    paper_exits.csv          # exits triggered today, with reason
    charts/<ticker>_<tf>.png # one per signal and per exit
    run_summary.json         # counts, universe size, data failures, duration
```

The folder date is the signal date. The signal ledger already assumes "entry at the next session's open".

### 9.3 Chart (one PNG per signal)

Four stacked panels, same x-axis, last 180 daily / 104 weekly bars: price (line chart; the chapter recommends a line chart
over candles to see divergences), PPO with signal line and histogram, ADX (with +DI/−DI faint), CCI with ±100/±200 lines.
Mark P1, S3, L2 and the entry bar E on price; X on ADX, Y on PPO; the CCI troughs at S3 and L2 joined by a line. Plot `W_t`
as a thin line in the ADX panel or as a fifth mini panel. Use matplotlib with the `Agg` backend. The owner reviews these by
eye, the chapter's "if in doubt, opt out" step.

### 9.4 Paper positions

State in `results/p24_falling_knife/_state/paper_positions.json` (idempotent: a re-run on the same date must not duplicate
entries or exits). One position per signal, fixed $2,000 notional (same convention as the signal ledger books), no
portfolio cap in paper mode; the cap is a reporting column.

### 9.5 Notifications

Send one message per run, only if there are new signals or exits: ticker, timeframe, drop, score, suggested entry, chart
attached. Reuse the notification path existing pipelines use. Read how P05/P21 run scripts send Telegram/email before
writing anything (`src/notification/`). Email HTML goes in `content.html`. Telegram messages longer than the limit are
chunked by the existing sender.

### 9.6 Signal ledger

Add a P24 harvester to `src/ml/pipeline/signal_ledger/harvesters.py` that reads `signals.csv` (side `long`, sub-streams
`daily` / `weekly`), following the existing harvester pattern (`clean_ticker`, folder date = `signal_date`). Add a test in
`signal_ledger/tests/`. This gives live forward-return tracking against VT/SPY with no extra work.

---

## 10. Module layout

```
src/ml/pipeline/p24_falling_knife/
    __init__.py              # empty
    README.md                # overview, quick start, how to run backtest/scan
    config.py                # all parameters (§4–§7), per timeframe; dataclasses
    data.py                  # FmpStore loading + adjust, DataManager loading, weekly resample, suspect-gap flag
    universe.py              # point-in-time filters (§3.4)
    indicators.py            # ADX/DI (TA-Lib), PPO (own EMA), CCI
    geometry.py              # causal ZigZag, reference peak, swing lows/highs, channel width W
    detector.py              # §4.2–§4.8 → per-bar condition table + signals (state machine incl. re-arm)
    exits.py                 # §4.9 + §5.3
    backtest.py              # §6: trade loop, controls, portfolio layer, process pool
    costs.py                 # §7.2
    validation.py            # bootstrap CIs, calibration grid, ablations, survivorship report
    report.py                # report.md + charts for Phase 1
    chart.py                 # §9.3
    scan.py                  # Phase 2 live scan
    paper.py                 # Phase 2 paper positions state
    jobs/
        __init__.py
        run_p24_backtest.py  # CLI: --timeframe daily|weekly|both --variant ... --calibrate --holdout
        run_p24_scan.py      # CLI: --timeframe daily|weekly [--date YYYY-MM-DD] [--dry-run]
    docs/
        11-chapter.html      # source (exists)
        pipeline-specification.md  # this file
        Requirements.md
        Design.md
        Tasks.md
    tests/
        test_indicators.py
        test_geometry.py
        test_detector.py
        test_exits.py
        test_backtest.py
        test_no_lookahead.py
        test_scan.py
```

### 10.1 Conventions (binding; from `.claude/CLAUDE.md` and project memory)

- Logger: `from src.notification.logger import setup_logger`; `_logger = setup_logger(__name__)`. Lazy `%s` formatting.
  **No `print()`.**
- Type hints and docstrings (PEP 257, Args/Returns) on every function. `Optional[...]` for nullable values.
- `datetime.now(timezone.utc)`, never naive `now()`.
- Absolute imports `from src....`. Empty `__init__.py`.
- Run scripts bootstrap the path **unconditionally**:
  `PROJECT_ROOT = Path(__file__).resolve().parents[5]; sys.path.insert(0, str(PROJECT_ROOT))` (never the guarded or
  `append` form; it shadows sibling `config.py` modules under the prod PYTHONPATH). Verify the `parents[n]` count.
- Line length 180. No bare `except:`; `_logger.exception(...)` in handlers.
- Always use the project venv: `.venv/Scripts/python.exe -m pytest src/ml/pipeline/p24_falling_knife/tests`,
  and keep `scripts/typecheck.py` (pyright + mypy) at 0 errors. CI runs only the typecheck job.
- Don't edit files with the bare `python` in Bash (cp1252 global interpreter).
- Fill in `README.md`, `docs/Requirements.md` (list cross-module dependencies: `src.data.fmp_pipeline`, `src.data`,
  `src.notification`, `src.ml.pipeline.shared`, `src.ml.pipeline.signal_ledger`), `docs/Design.md`, `docs/Tasks.md` from the
  templates in `.claude/CLAUDE.md` §13.
- Commit messages in the imperative mood.

---

## 11. Phase 1 → Phase 2 go/no-go gate

Present the report to the owner with a recommendation. The owner decides. Suggested criteria, all on the **holdout**, net
of 1× costs, for at least one timeframe:

1. ≥ 60 holdout trades (excluding `suspect_gap`);
2. median net trade return > 0 and mean net trade return > 0, with the bootstrap 95% CI of the mean above 0;
3. mean net return beats control 1 (drop-only) by a margin whose paired bootstrap CI excludes 0;
4. still positive at 2× costs;
5. the result does not depend on one month: removing the best calendar month keeps criterion 2;
6. win rate reported next to the book's ">80%" claim, whatever it is.

If the gate fails: write the findings into `docs/Tasks.md` and the memory notes, and stop. Do not loosen the rules to pass.

---

## 12. Acceptance tests (pytest)

- [ ] **PPO:** matches a hand-computed EMA series to 1e-9; `PPO_hist = PPO − signal`; TA-Lib `PPO` is not used.
- [ ] **ADX/CCI:** match TA-Lib on a fixed fixture; NaN warm-up handled; no forward fill across the warm-up.
- [ ] **ZigZag is causal:** a swing high is reported only after the `zz_pct` reversal has closed; truncating the series at
      any bar never changes swing points confirmed before that bar.
- [ ] **No look-ahead (truncation test):** for 50 random cut points on 20 real securities, the per-bar condition table and
      signals computed on data up to t equal the full-history run for all bars ≤ t.
- [ ] **Synthetic knife:** a constructed series (rise, −50% fall with an intermediate bounce, slowing fall, lower low with
      a shallower CCI trough, rebound) produces exactly one signal within the expected bar range. Variants with the drop at
      35%, a flat CCI (equal troughs), no intermediate support, or a W that never narrows produce no signal.
- [ ] **Point 3 choice:** with a bounce top and a support low in the fixture, S3 is the support low, never the bounce top.
- [ ] **Re-arm:** a second knife leg makes a second signal only after the re-arm conditions; a repeat bar in the same leg
      does not.
- [ ] **Exits:** CCI ≥ 200 exits next open; bearish divergence fixture exits at the right bar; shakeout fixture stays in
      when `cci_s > cci_3 + margin` and exits otherwise; delisting exits at the last close.
- [ ] **Execution:** every fill is at the next bar's open ± tiered cost; costs appear in the trade rows.
- [ ] **Weekly resample:** an incomplete current week is never used; a holiday week resamples correctly.
- [ ] **Suspect gap:** a 40% one-bar fall on low volume is flagged `suspect_gap=True`.
- [ ] **Determinism:** same config + seed → identical `trades_*.csv` (hash compared).
- [ ] **Scan idempotency (Phase 2):** running the scan twice for the same date gives the same files and no duplicate paper
      entries.
- [ ] **Harvester (Phase 2):** the signal ledger reads a sample `signals.csv` into the expected `SignalRecord`s.

---

## 13. Suggested build order

1. Module skeleton + docs templates + `config.py`.
2. `indicators.py` + tests.
3. `geometry.py` (ZigZag, peak, supports, W) + causality tests.
4. `detector.py` + synthetic-knife and truncation tests.
5. `exits.py` + tests.
6. `data.py` / `universe.py` on the FMP vintage; run the survivorship report (§7.4) **first** and show it to the owner
   before spending time on the backtest. It may change the backtest window.
7. `backtest.py` + `costs.py` with defaults, daily and weekly, controls. Sanity-check on the book examples (§7.6).
8. Calibration on train, ablations, then the holdout run once; `report.md`.
9. Owner gate (§11).
10. Phase 2: `chart.py`, `scan.py`, `paper.py`, notifications, harvester, `p24_specs.py` (registered disabled).

---

## 14. Owner decisions (2026-11-05)

1. **Coverage threshold: 50%.** A year whose delisted-name coverage (§7.4) is below 50% counts as an "optimistic upper
   bound" only, and the headline result excludes it. If that leaves only 2021+, the train/holdout split in §7.3 is redone
   inside the remaining window (e.g. train 2021–2023, holdout 2024+), and the report says so.
2. **Live `min_price`: $1.00.** The backtest still reports the sub-$1 bucket (`min_price=0.10` variant). Lower the live
   floor only if that bucket stays positive at the 2% per-side cost tier and the owner agrees.
3. **Live signals: paper + chart only.** Notification with the chart, paper position booked automatically, no Telegram
   accept/skip step.
4. **Tracking: signal ledger only** (P24 harvester, §9.6). No separate sleeve/NAV.

### 14.1 Measured vintage coverage (2026-11-05, vintage `2026-10`, 20,324 securities)

Delisted securities with a daily price file, by delisting year: 2000–2015 ≤ 6 per year; 2016–2020 59–203 per year;
2021–2026 437–1,218 per year. Real US delistings run to several hundred per year, so coverage before 2021 is far below 50%.
Consequences, binding for the implementation:
- **Headline window: 2021-01-01 to the vintage end.** Train 2021-01-01 – 2023-12-31 (includes the 2022 bear market),
  holdout 2024-01-01 – vintage end. Indicator warm-up data is loaded from 2019-06.
- 2004–2020 runs on survivors only and is reported as an **optimistic upper bound**. It is never used for calibration.
- **Superseded 2026-11-06 (owner): headline window 2022+.** 2021's estimated delisted coverage is 44.5%, under the
  50% threshold, so train is 2022-01-01 – 2023-12-31, holdout 2024-01-01 – vintage end, and 2004–2021 is the
  upper bound. The coverage reference years are 2022–2025.
- **Book examples:** `AONE`, `PWER` and `ATHR` in the vintage are **different, later companies that reuse the ticker**
  (first prices 2020-10, 2023-11, 2025-04). The originals are not in the vintage. Only MPWR, VMW, DIA and QQQ can be
  checked in §7.6. Match examples on `security_id` + date range, never on the symbol alone.

### 14.2 Implementation corrections (2026-11-05)

- **Price floor and cost tier use the as-traded (raw) price**, not the split-adjusted one. The split-adjusted history
  of a stock that later reverse-split looks far pricier than it traded (a 1:10 reverse split shows a $0.50 print as $5).
- **FMP `JUMP_NO_SPLIT` alone does not make a bar suspect** (§3.3). It fires on a close ratio near 2, 3, 3/2, …, which
  real −33%/−50% crash days also match (ADEA 2008, ACAD 2009, AGTC 2016 at 20–35× normal volume). A jump bar counts as
  suspect only if its volume is below `jump_volume_ratio` (3×) the prior 20-bar median.
- **Second leg of a knife** (§4.8 re-arm): the ADX/PPO/W extremes are measured from the widest channel point after the
  re-arm, not from the re-arm bar itself (that picked up the first leg's decaying ADX).
- **ADX peak, PPO trough and W minimum are local** (§4.4–§4.6): searched over `(max(P1, t − extreme_window), t]`,
  `extreme_window` = 40 daily / 20 weekly, not over the whole `(P1, t]`. One reference top can carry several
  mountains: the book's own Dow table has three (June 2008, Oct 2008, Mar 2009) under the 11,800 top. Over `(P1, t]` the
  October 2008 mountain masked the March 2009 one and the book's Dow trade never fired; with the local window DIA signals
  on 2009-03-11 at $69.38 and exits at $84.06 (book: ≈ 6,700 → 8,200 on the Dow).
- **Narrowest-point windows match the ADX/PPO windows** (§4.6 items 3–4): `max_bars_after_wmin` and `max_low_to_wmin`
  are 10 daily / 6 weekly (were 5 / 3), the same as `adx_max_bars_after_peak` / `ppo_max_bars_after_trough`. With 5,
  a valley/mountain allowed to be 10 bars old could never pass the channel test; VMW Oct 2008 (book trade 2) had its
  narrowest point at the first crash low on Oct 10 and the lower low with a shallower CCI on Oct 21.
- **Weekly freshness `max_bars_since_low` = 8** (was 3). Weekly ADX lags: on QQQ the mountain topped 5 weeks after the
  Nov 2008 low, so a 3-week limit dropped the bar before the mountain formed. The chapter (§8) says weekly channels are
  long and the entry can be "anywhere between points 3 and 4". QQQ weekly then signals 2009-01-02 at $30.82 (book: ≈
  1,500 on the Nasdaq). The parameter stays in the calibration grid.
- **Book-example check, defaults after these fixes:** daily VMW 2008-10-22 $20.29 (book ≈ $20), MPWR 2008-12-09 $9.60
  (book weekly ≈ $9), DIA 2009-03-11 $69.38; weekly QQQ 2009-01-02, DIA 2008-12-19. MPWR weekly fails narrowly (ADX
  peak 31.1, CCI divergence 21 < 25). None of the defaults were tuned beyond the fixes listed above.
- **More suspect-data rules (2026-11-06):** exchange test listings (`ZXZZT`-style symbols, names with "test
  (common) stock" / "SuperMontage test") are not common stock. A one-bar rise of ≥ 4× (`splice_ratio`) on volume under
  3× the median is a suspected reorg splice. Trades are excluded if the event is in the drop (`suspect_gap`) or in the
  holding period (`suspect_in_trade`); the random control drops such draws too.
- **Calibration (§7.3) as built:** grid `zz_pct` {0.20, 0.25, 0.30} × `support_bounce_pct` {0.06, 0.10} × `w_narrow`
  {0.4, 0.6, 0.8} × `cci_div_min` {15, 25, 40} × `max_bars_since_low` {5, 8} daily / {8, 12} weekly = 108 cells per
  timeframe, each with the shakeout rule on and off (216 runs; the off runs reuse the same signals). Plateau = every
  one-step neighbour's median within 25% of the cell's; the shakeout setting in `calibration_frozen.json` is the
  owner's choice (book rule: on).
