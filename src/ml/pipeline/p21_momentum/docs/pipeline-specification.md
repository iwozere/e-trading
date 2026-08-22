# Pipeline Specification: DIY Momentum (Paper) with Monthly QDVA Comparison

**Version:** 1.2
**Date:** 2026-08-22
**Purpose:** Executable specification for a programmatic agent
**Mode:** Paper trading — no live orders are placed
**Changes from 1.1:** Rewired §2/§3/§10/§11/§13/§14 onto this repo's existing downloader layer and
`results/<pipeline>/YYYY-MM-DD/` convention instead of an ad-hoc `state/` folder and raw `yfinance` calls.
Adopted adjusted-close-only bookkeeping (§10) since the project's Yahoo downloader does not expose raw
unadjusted prices or corporate-action events. IBKR bid/ask reconciliation (old §10) deferred to Phase 3+
(§17) — the existing `IBKRDownloader` only supports historical bars, not live bid/ask snapshots.

---

## 0. READ FIRST: What This Pipeline Cannot Do

The framing "run it on paper, and if the results look good, trade it live" contains a statistical error that will invalidate the entire project unless it is addressed at the outset.

**Expected tracking error between a 20-position momentum portfolio and MSCI USA Momentum: 6–9% annualized.**

Consequences:

| Horizon | Std. dev. of return difference | What can be concluded |
|---|---|---|
| 3 months | ~4% | Nothing |
| 12 months | ~7.5% | Nothing |
| 36 months | ~4.3% | Almost nothing |
| ~50 years | ~1.0% | A 2% edge is distinguishable from zero at p<0.05 |

If after 12 months the DIY portfolio beats QDVA by 5%, the probability that this is noise rather than edge is roughly 75%. If it trails by 5%, the same holds in reverse. **Return comparison is not, and cannot be, a decision criterion.**

### What the pipeline is actually for

1. **Operational validation.** Does the code run, is the data clean, do the filters hit their intended targets, does anything break on corporate actions.
2. **Cost model validation.** Realized turnover and costs versus forecast (~175% and ~0.15%).
3. **Operator discipline validation.** Will you actually run the rebalance every month. This is the single most valuable output of the project.
4. **Attribution.** Separating the contribution of the regime overlay from the contribution of stock selection (§9). Signal emerges faster here than in headline returns.
5. **Psychological calibration.** Watching a −20% drawdown in the sleeve before it costs real money.

§12.6 defines the T+12 decision criteria, built on items 1–5 rather than on returns. **The agent must reproduce this warning in every monthly report.**

---

## 1. Architecture

Three scheduled jobs. All idempotent, all safe to re-run.

| Job | Schedule | Purpose | Criticality |
|---|---|---|---|
| `monthly_rebalance` | Last NYSE trading day of month, 16:30 ET | Signal computation, target portfolio | High |
| `monthly_execute` | First trading day of month, 09:45 ET | Fill simulation, reporting | High |
| `daily_mark` | Every trading day, 16:30 ET | NAV marking, catastrophic stops, price-anomaly check | Medium |

Separating rebalance from execute is mandatory: it structurally eliminates look-ahead bias. Signal is computed at day T close; execution occurs at T+1 open. Both timestamps are recorded.

```
monthly_rebalance:
  fetch → validate → signal → filter → rank → select → size → regime → write TARGET

monthly_execute:
  read TARGET → fetch opens → simulate fills → apply costs → write POSITIONS + LEDGER
  → update all 4 tracks → generate REPORT
```

### Trading calendar
Use `pandas_market_calendars` (`XNYS`). Do not derive trading days heuristically from weekends — holidays will break the logic.

---

## 2. Data Sources and Contracts

All fetching goes through this repo's existing `src/data/downloader/` layer, not raw `yfinance`/`requests`
calls. This is a deliberate constraint (§2.1) — the pipeline must not grow a second, parallel data-access
path for data the project already knows how to fetch.

| Data | Source | Reused code | Frequency |
|---|---|---|---|
| S&P 500 constituents + GICS sector | Wikipedia `List_of_S&P_500_companies` | `src.util.tickers_list.get_sp500_constituents_with_sector()` — **new function**, added next to the existing `get_sp500_tickers_wikipedia()` in the same file; same page, same request pattern, additionally captures the `GICS Sector` column that the ticker-only helper drops | Monthly |
| Equity prices (OHLCV, adjusted) | Yahoo Finance | `DataDownloaderFactory.create_downloader("yahoo")` → `YahooDataDownloader.get_ohlcv_batch(symbols, interval="1d", start_date, end_date)` | Monthly / daily |
| Momentum benchmark | Yahoo Finance | Same `YahooDataDownloader.get_ohlcv` — ticker `MTUM` | Daily |
| Market reference | Yahoo Finance | Same — ticker `SPY` | Daily |
| Regime: index | Yahoo Finance | Same — ticker `^GSPC` | Daily |
| Regime: volatility | Yahoo Finance | Same — ticker `^VIX` | Daily |
| Earnings calendar | yfinance `.get_earnings_dates()` | **No existing wrapper** — no downloader class exposes this. Add a small local helper in `p21_momentum/data/earnings.py` calling `yf.Ticker(symbol).get_earnings_dates()` directly, matching the lazy-logging convention. Flagged as the one place this pipeline talks to yfinance outside the shared downloader | Monthly |
| Fundamentals (quality filter) | Yahoo Finance | `YahooDataDownloader.get_fundamentals_batch(symbols)` → `Fundamentals.free_cash_flow`, `Fundamentals.net_income` (TTM as exposed by yfinance `.info`) | Quarterly (cached) |
| Real execution prices (optional) | — | **Deferred to Phase 3+** — see §17. The existing `IBKRDownloader` only wraps `reqHistoricalData` (historical bars); a live bid/ask snapshot at execution time is new code against `src/trading/broker/ibkr_broker.py`'s connection and is out of scope for Phase 0–2 | — |

### 2.1 Why route through the existing downloader layer
`YahooDataDownloader` already solves problems this spec would otherwise re-solve badly: TLS-fingerprint
session rotation against Yahoo's anti-bot layer, batch-download with automatic per-symbol fallback when a
batch call fails, and suppression of yfinance's noisy false-positive "possibly delisted" errors on holiday
gaps. Re-implementing `yf.download()` calls locally would silently drop all of that. The one real gap —
raw unadjusted prices / corporate-action events — is addressed by simplifying the bookkeeping model instead
of forking the downloader (§10).

### Why MTUM rather than QDVA
QDVA trades on Xetra in **EUR**, IUMO on LSE in **USD**; both have gaps in yfinance history and carry FX contamination. MTUM tracks the **same MSCI USA Momentum index**, quotes in USD, and has clean daily data back to 2013.

MTUM is used as an *index return proxy*. Adjust for the TER difference: MTUM 0.15% versus QDVA 0.20%, i.e. subtract an additional **5 bps annually** (`0.0005 / 252` per day) from benchmark returns. The difference is immaterial, but it is made explicit so the comparison carries no hidden advantage.

### Ticker normalization
yfinance uses hyphens instead of dots: `BRK.B` → `BRK-B`, `BF.B` → `BF-B`. `get_sp500_constituents_with_sector()` applies `ticker.replace('.', '-')` across the constituent list, matching the existing `get_sp500_tickers_wikipedia()` behavior. Store both forms in `_state/`.

### Fetching
`YahooDataDownloader.get_ohlcv_batch()` already batches and falls back to individual downloads per symbol
on batch failure — do not add a second retry loop on top of it. It does not raise on an empty result, so
the pipeline must still check each returned frame for non-emptiness explicitly before use (§13).

**Adjusted close only — provisional, marked for revisit (§10.1).** `get_ohlcv_batch()` calls yfinance with
`auto_adjust=True` and returns a single `close` column (open/high/low move with it). There is no separate
raw `Close` and no `actions` feed. §10 specifies how the signal, sizing, and execution logic all work from
this one adjusted series. This is a deliberate scope simplification for v1.2, not a permanent architectural
stance — see the callout at the top of §10.1 for when to revisit adding raw `Close`/`Adj Close` back in.

---

## 3. Results Schema

All output on disk, JSON/CSV/Markdown, under `results/p21_momentum/` — the same root every other pipeline
in this repo uses (`results/p06_emps2/`, `results/p18_institutional_flow/`, `results/p20_kestrel/`, …).

The layout has two parts, and the split is deliberate, not incidental:

- **Dated run folders** (`YYYY-MM-DD/`) — one per job invocation, named by the date the job ran. These are
  point-in-time snapshots: once written, a given run's folder is never edited by a later run. This is what
  makes the full history browsable by just listing the directory — exactly what was asked for.
- **`_state/`** — the handful of files that are *not* point-in-time snapshots: either genuinely continuous
  logs that are the history (`ledger.jsonl`, `nav_daily.csv` — every row ever appended, never truncated) or
  mutable pointers a job needs to read on its next run (`current_positions.json`, TTL caches). Putting these
  in dated folders would either duplicate the same growing file 250+ times or force every job to scan
  history to find "what do I hold right now" — both worse than one well-known path. The leading underscore
  matches the existing `_ohlcv_cache/` / `_runs/` convention already used under `results/p06_emps2/` and
  `results/p09_arbitrage/`.

```
results/p21_momentum/
  YYYY-MM-DD/                      # one folder per job run, named by run date
    universe.json                  # constituents + sectors at signal date      (monthly_rebalance)
    signals.json                   # full signal table, all ~500 names          (monthly_rebalance)
    targets.json                   # target portfolio, pre-execution            (monthly_rebalance)
    positions.json                 # post-execution snapshot                    (monthly_execute)
    report.md                      # monthly report, §12                        (monthly_execute)
    daily_mark.json                # NAV/high-water/anomaly snapshot            (daily_mark)
  _state/
    current_positions.json         # live holdings — overwritten every run that changes them
    ledger.jsonl                   # append-only, one line per simulated trade, ever
    nav_daily.csv                  # append-only, daily NAV for all 5 tracks, ever
    regime_history.json            # append-only, one entry per month
    cache/fundamentals.json        # TTL 90 days
    cache/sectors.json             # TTL 30 days
```

A given month's trades are not duplicated into that month's dated folder — they are the rows of
`_state/ledger.jsonl` with a matching `ts`. Filter by date rather than maintaining two copies that could
drift apart.

`config/pipeline/p21_exclusions.json` (§5, F5) is operator-maintained input, not pipeline output, so it does
not live under `results/` at all — see §5, which follows the `config/pipeline/*.json` convention already
used by `activists.json` in P20.

### `_state/current_positions.json`

```json
{
  "as_of": "2026-09-01",
  "track": "A",
  "nav_total": 250000.00,
  "cash": 201430.50,
  "sleeve_market_value": 48569.50,
  "regime_scalar": 1.0,
  "positions": [
    {
      "ticker": "NVDA",
      "shares": 14.2371,
      "avg_cost": 175.40,
      "entry_date": "2026-06-01",
      "entry_rank": 3,
      "current_rank": 7,
      "sector": "Information Technology",
      "target_weight_pct": 0.95,
      "high_water_price": 198.20
    }
  ]
}
```

### `_state/ledger.jsonl` — one line per trade, ever

```json
{"ts":"2026-09-01T09:45:00-04:00","track":"A","ticker":"NVDA","side":"BUY","shares":14.2371,"ref_open":175.35,"fill_price":175.40,"slippage_bps":3.0,"commission_usd":0.35,"gross_usd":2496.00,"net_usd":2496.35,"reason":"ENTRY_RANK_3"}
```

The `reason` field is mandatory. Permitted values:
`ENTRY_RANK_n` · `EXIT_RANK_DROP` · `EXIT_SECTOR_CAP` · `EXIT_FILTER_FAIL` · `EXIT_DELISTED` · `EXIT_CATASTROPHIC_STOP` · `REBAL_TRIM` · `REBAL_ADD` · `REGIME_SCALE_DOWN` · `REGIME_SCALE_UP`

`CORPORATE_ACTION_SPLIT` from the original design is dropped: under adjusted-close-only bookkeeping (§10)
a split never produces a discontinuity in the price series in the first place, so there is nothing to log.

### Idempotency
Each job checks for `results/p21_momentum/<run_date>/<its primary output file>` as its first step. If present, no-op with log `SKIP: already processed`, unless `--force` is passed. Re-running must not duplicate `_state/ledger.jsonl` entries.

---

## 4. Signal Computation

Signal date `T` = last trading day of the month. All indices are in trading days back from `T`.

```python
LOOKBACK_START = 252   # ~12 months
SKIP_RECENT    = 21    # ~1 month, EXCLUDED
MIN_HISTORY    = 260   # minimum bars for eligibility

def compute_signal(adj_close: pd.Series) -> dict | None:
    if len(adj_close) < MIN_HISTORY:
        return None                      # IPO / insufficient history

    window = adj_close.iloc[-LOOKBACK_START : -SKIP_RECENT]   # 231 bars

    raw_return = window.iloc[-1] / window.iloc[0] - 1.0

    weekly = window.resample('W-FRI').last().pct_change().dropna()
    if len(weekly) < 40:
        return None
    vol = weekly.std(ddof=1) * math.sqrt(52)

    if vol < 0.05:                       # guard against division by ~0
        return None

    return {
        "raw_return": raw_return,
        "vol": vol,
        "signal": raw_return / vol,      # ← RANK ON THIS FIELD
    }
```

**Critical:** ranking uses `signal` (risk-adjusted), not `raw_return`. This error fails silently and costs roughly the entire factor premium.

**Critical:** the window ends at `-SKIP_RECENT`, not at `-1`. Including the most recent month imports short-term reversal into the portfolio and consistently degrades results.

---

## 5. Filters (applied BEFORE ranking)

Order matters: cheap filters first, expensive (network) filters only on survivors.

| # | Filter | Rule | Action on missing data |
|---|---|---|---|
| F1 | History | < 260 bars | Exclude |
| F2 | Liquidity | Median (Close × Volume) over 60 days < $50M | Exclude |
| F3 | Gap filter | Sum of top-3 daily log returns / total log return > 0.40 | Exclude |
| F4 | Quality | TTM FCF < 0 **AND** TTM net income < 0 | **Pass** + flag |
| F5 | M&A | Listed in `config/pipeline/p21_exclusions.json` | Exclude |
| F6 | Earnings | Report within 5 calendar days after execution date | Exclude from NEW entries only; holdings unaffected |

### F3 — implementation

```python
log_rets = np.log(window / window.shift(1)).dropna()
total = log_rets.sum()
if total <= 0:
    passes_f3 = True          # negative momentum will be filtered by ranking
else:
    top3 = log_rets.nlargest(3).sum()
    passes_f3 = (top3 / total) <= 0.40
```

The purpose is to exclude names whose "trend" consists of a single gap (acquisition announcement, one-off surprise). That does not persist.

### F4 — data quality warning
yfinance fundamentals are unreliable and incomplete, particularly for financials. F4 is therefore specified **loosely**: it excludes only names that are simultaneously loss-making on FCF and on net income. On missing data, **pass rather than exclude**, and record a `f4_data_missing` counter in the report. If the counter consistently exceeds 15% of candidates, the filter is inert and should either be removed or moved to a paid data source.

Fundamentals cache: TTL 90 days, keyed `ticker → {fcf_ttm, net_income_ttm, fetched_at}`. Backed by
`YahooDataDownloader.get_fundamentals_batch()` (§2), written to `results/p21_momentum/_state/cache/fundamentals.json`.

### F5 — manual list
`config/pipeline/p21_exclusions.json` (project-root config, checked into git — same convention as P20's
`config/pipeline/activists.json`) is the only point of manual intervention in the system:

```json
{"exclusions": [
  {"ticker": "XYZ", "reason": "announced acquisition at fixed price", "added": "2026-07-15", "expires": "2026-12-31"}
]}
```

The agent reads this file but never writes to it. Expired entries are ignored.

---

## 6. Selection: Ranking, Hysteresis, Sector Cap

```
ENTRY_RANK      = 20
HOLD_RANK       = 60
MAX_PER_SECTOR  = 4
TARGET_COUNT    = 20
```

Operation order is **strict**:

```python
# 1. Rank filter survivors by descending signal
ranked = sorted(survivors, key=lambda x: -x['signal'])
for i, s in enumerate(ranked): s['rank'] = i + 1

# 2. Retain: current positions with rank <= HOLD_RANK stay
keep = [p for p in current_positions
        if rank_of(p.ticker) is not None and rank_of(p.ticker) <= HOLD_RANK]

# 3. Forced exits: dropped from index / failed F1,F2,F3,F5 / delisted
keep = [p for p in keep if p.ticker not in forced_exits]

# 4. Apply sector cap to retained names:
#    if a sector holds > MAX_PER_SECTOR, drop the worst-ranked
keep = enforce_sector_cap(keep, MAX_PER_SECTOR)

# 5. Fill to TARGET_COUNT from ranked[:ENTRY_RANK], ascending rank,
#    skipping already-held names and sector-cap violations
# 6. If top-20 is exhausted and positions < TARGET_COUNT —
#    widen the pool to ranked[:40], do NOT relax the sector cap
# 7. If still < TARGET_COUNT — accept the smaller portfolio,
#    remainder to cash, log WARN_UNDERFILLED
```

**Step 6 fires more often than expected.** In a concentrated market (entire top-20 in two sectors), a 4-name sector cap leaves 8 positions out of 20. Widening the pool to 40 solves this; relaxing the sector cap does not — that cap is precisely what prevents the factor from degenerating into a sector bet.

**Hysteresis is the primary turnover lever.** `ENTRY=20 / HOLD=60` yields ~175% annualized. Without the buffer (`HOLD=20`), turnover exceeds 350% for essentially the same return. Do not change without a backtest.

---

## 7. Position Sizing

```python
NAV_TOTAL          = 250_000
SLEEVE_TARGET_PCT  = 0.20     # 20% of portfolio
MAX_POSITION_PCT   = 0.01     # 1% of NAV — hard cap
```

Inverse-volatility weights with iterative capping:

```python
def size_positions(selected, nav_total, sleeve_pct, max_pos_pct, regime_scalar):
    sleeve_usd = nav_total * sleeve_pct * regime_scalar
    cap_usd    = nav_total * max_pos_pct          # ← cap is off TOTAL NAV,
                                                  #   not off sleeve size
    inv_vol = {t: 1.0 / s['vol'] for t, s in selected.items()}
    capped, free = {}, dict(inv_vol)
    remaining = sleeve_usd

    for _ in range(10):
        total_w = sum(free.values())
        if total_w == 0: break
        alloc = {t: remaining * w / total_w for t, w in free.items()}
        over  = {t: v for t, v in alloc.items() if v > cap_usd}
        if not over:
            capped.update(alloc); break
        for t in over:
            capped[t] = cap_usd
            free.pop(t)
            remaining -= cap_usd

    return capped   # sums to sleeve_usd (± rounding)
```

Notes:
- **The cap is computed off full NAV ($250k), not off sleeve size.** 1% = $2,500. This reflects the operator's original constraint.
- When `regime_scalar < 1`, the entire sleeve scales down proportionally; positions are retained, only scale changes. Released funds go to cash.
- Fractional shares are required. Round `shares` to 4 decimals (IBKR fractional granularity).

---

## 8. Regime Overlay

Computed at signal date `T`, applied at execution `T+1`.

```python
spx = fetch('^GSPC')
vix = fetch('^VIX')

bear = (spx.iloc[-1] / spx.iloc[-252] - 1 < 0) or (spx.iloc[-1] < spx.rolling(200).mean().iloc[-1])

vix_20d = vix.iloc[-20:].mean()      # 20-day average, NOT spot
high_vol = vix_20d > 28

if not bear:                scalar = 1.00
elif bear and not high_vol: scalar = 0.60
else:                       scalar = 0.25
```

**The 20-day VIX smoothing is mandatory.** Spot VIX triggers on single-day spikes and produces expensive exposure chatter.

Regime hysteresis: downward changes to `scalar` apply immediately; upward changes require **two consecutive** months in the new state. The asymmetry is deliberate — protection engages fast, disengages slowly.

Write to `regime/history.json` each month:
```json
{"date":"2026-08-31","spx_12m_return":0.142,"spx_vs_200dma":1.058,
 "vix_20d_avg":16.4,"bear":false,"high_vol":false,
 "scalar_raw":1.00,"scalar_applied":1.00,"months_at_state":14}
```

---

## 9. Four Tracks — Attribution

All four are computed simultaneously from the same data. This is the pipeline's key analytical output.

| Track | Content | Overlay |
|---|---|---|
| **A** | DIY 20 stocks | Yes |
| **B** | DIY 20 stocks | No (always 100%) |
| **C** | MTUM (QDVA proxy) | No |
| **D** | MTUM (QDVA proxy) | Yes |

Decomposition:

```
B − C  = stock selection effect      (20 names vs. 126)
A − B  = overlay effect on stocks
D − C  = overlay effect on the ETF
A − D  = total DIY benefit over "QDVA + overlay"
```

**`A − D` is the only number that answers the real question.** QDVA plus an overlay is already available at 15 minutes per month. DIY is justified only by what it adds beyond that.

Additionally maintain **Track E: SPY** as an anchor. If the entire momentum complex trails plain beta for three consecutive years, the question is about the factor, not the implementation.

All tracks in USD, total return, starting from an identical notional NAV of 250,000 on the same day. Costs apply to A and B (commissions + slippage); TER applies to C and D (0.20%/252 daily on position value).

---

## 10. Execution Model (Paper)

### 10.1 Adjusted-close-only bookkeeping

> **Status: provisional, marked for revisit — not a closed decision.** This is a scope simplification made
> to ship v1.2 against the existing downloader layer, not a claim that raw `Close`/`Adj Close` has no value
> here. If Phase 0/1 (§14, §15) shows the adjusted-only model materially distorts share counts, cash
> accounting, or the corporate-action audit trail (§17 item 9), the fix is to extend `YahooDataDownloader`
> with a raw-price + `actions` fetch (§2.1) and re-derive §10/§11 from both series — not to keep patching
> around the gap. Revisit at that point rather than treating this section as final.

This section departs from the literal "share count on a real brokerage statement" model, by design, because
`YahooDataDownloader.get_ohlcv_batch()` (§2) returns a single split-and-dividend-adjusted `close` series
(and a matching adjusted `open`) with no raw price and no `actions` feed alongside it. Rather than bypass
the shared downloader to reconstruct that data locally, the whole position/ledger model is expressed in the
adjusted series throughout:

- **Signal, sizing, fills, avg_cost, high_water_price — all computed off the adjusted series.** There is
  only one price per (ticker, date); there is no reconciliation to do between an "adjusted" and a "raw"
  number.
- **No manual split adjustment.** A split never produces a discontinuity in an already-adjusted series, so
  there is nothing to detect or correct — `enforce_sector_cap`, `high_water_price` tracking, and the
  catastrophic-stop check all just keep working across a split with no special case.
- **No separate dividend crediting.** Total return from dividends is already embedded in the adjusted price
  path, so the "credit dividends to cash on ex-date" step from the original design is redundant and dropped
  — crediting it separately would double-count the dividend.
- **`shares` is a derived, notional quantity**, not a literal broker-reconcilable share count: it is
  `dollar_allocation / adjusted_price_at_fill`, and it moves when the adjustment factor for a name is
  revised retroactively (e.g. a newly-declared dividend). This is fine for a paper P&L simulation that never
  places a real order. It is **not** fine for placing a real order — see §17 for the one-time reconciliation
  a live migration would need.

```python
SLIPPAGE_BPS      = 3.0        # S&P 500 large caps
COMMISSION_MIN    = 0.35       # IBKR Tiered, per-order minimum
COMMISSION_PER_SH = 0.0035
COMMISSION_MAX_PCT= 0.01       # 1% of trade value
MIN_TRADE_USD     = 150        # chatter threshold

def simulate_fill(ticker, side, shares, open_price):
    """open_price is the adjusted 'open' for the fill date, from get_ohlcv_batch()."""
    sign = 1 if side == 'BUY' else -1
    fill = open_price * (1 + sign * SLIPPAGE_BPS / 10_000)
    gross = fill * shares
    comm = min(max(COMMISSION_MIN, COMMISSION_PER_SH * shares),
               COMMISSION_MAX_PCT * gross)
    return fill, comm
```

- Fill at the **adjusted open** of the first trading day of the month. Not at the prior close.
- Slippage always works against you: buys higher, sells lower.
- Sells execute before buys. If cash is insufficient, buys are scaled down proportionally and `WARN_INSUFFICIENT_CASH` is logged.
- **Chatter threshold:** skip the trade if |target − current| position value < $150. Saves commissions with no risk impact.

### 10.2 IBKR reconciliation — deferred

The original design's "request real bid/ask at execution time" is **out of scope for Phase 0–2** (confirmed
with the user). `src/data/downloader/ibkr_downloader.py` only wraps `reqHistoricalData` (historical bars,
delayed data); a live bid/ask snapshot at the moment of a simulated fill would be new code against
`src/trading/broker/ibkr_broker.py`'s connection, not a reuse of existing plumbing. Revisit at Phase 3
(§15) once the paper simulation itself has 12 months of history — see §17.

---

## 11. Daily Job

```
1. Fetch closes for all holdings + MTUM + SPY + ^GSPC + ^VIX via YahooDataDownloader.get_ohlcv_batch
2. Append today's row to _state/nav_daily.csv for all 5 tracks;
   write results/p21_momentum/YYYY-MM-DD/daily_mark.json (today's snapshot, for browsing)
3. Update high_water_price per position (adjusted close)
4. Catastrophic stop check: if adjusted close < avg_cost * 0.65 →
   flag EXIT_CATASTROPHIC_STOP, execute at next open
5. Anomaly check: if |close_t / close_{t-1} - 1| > 0.35 → flag MANUAL_REVIEW, do NOT auto-trade
```

Steps 6–7 of the original design (manual split adjustment, dividend crediting) are dropped — see §10.1 for
why. One consequence worth flagging in step 5: because the series is pre-adjusted, a jump that size is
never a real split masquerading as an anomaly (adjustment already smooths those away) — every trigger here
is a genuine price event or a data error, so there is no false-positive class to filter out that the
original design's "and no split event in actions" clause existed to handle.

The −35% catastrophic stop is the **only** stop in the system. Conventional stops degrade momentum performance: they exit on pullbacks inside a trend. −35% catches genuine accidents only.

---

## 12. Monthly Report

File `results/p21_momentum/YYYY-MM-DD/report.md` (run date = execution date). Mandatory sections:

### 12.1 Header
Signal date, execution date, regime (`scalar`, state, months in state), NAV for all five tracks.

### 12.2 Statistical power disclaimer
Reproduced **every month, in full**:

> N months elapsed. With tracking error of ~7.5% annualized, the observed A−D difference of X.X% is statistically indistinguishable from zero (t = …). This report is **not** evidence that the strategy does or does not work. Decision criteria are in §12.6.

Compute the t-statistic explicitly: `t = mean_monthly_diff / (std_monthly_diff / sqrt(N))`. Display it. It will almost certainly sit between −1 and 1 for the first several years, and seeing that is the point.

### 12.3 Trades this month
Table: ticker, side, shares, price, commission, `reason`, rank before/after.

### 12.4 Current portfolio
20 rows: ticker, sector, weight %, rank, return since entry, days held. Plus sector distribution with cap verification.

### 12.5 Attribution
Cumulative return table for A/B/C/D/E — month, YTD, since inception. Plus the four differences from §9. Plus realized metrics: annualized turnover, costs in bps, max drawdown per track.

### 12.6 Decision criteria panel

Evaluated at T+12. **No criterion involves returns.**

| # | Criterion | Threshold | Failure means |
|---|---|---|---|
| D1 | Missed rebalances | 0 of 12 | No discipline → buy QDVA |
| D2 | Realized turnover | 130–220% p.a. | Model diverged from reality; re-check hysteresis |
| D3 | Realized costs | < 0.30% p.a. | Economics worse than forecast |
| D4 | `MANUAL_REVIEW` events | ≤ 4 per year | Pipeline demands too much hand-holding |
| D5 | `f4_data_missing` | < 15% of candidates | Quality filter inert; remove or change source |
| D6 | `WARN_UNDERFILLED` | ≤ 2 per year | Sector cap too tight for current market |
| D7 | Track A max drawdown | Survived without intervention | Psychological test |
| D8 | Mean A−D difference | **informational, not a criterion** | — |

**Decision rule:** go live if D1–D7 pass. Do not go live if D1 or D7 fails, regardless of returns. On D2–D6 failure, fix the specific component and extend the paper period by 6 months.

---

## 13. Data Quality Gates

Every job begins with validation. **An `ABORT` halts the job, leaves state unchanged, sends an alert, and leaves the portfolio untouched until intervention.**

| Check | Threshold | Action |
|---|---|---|
| Constituent list loaded | ≥ 450 names | ABORT |
| Tickers with complete price data | ≥ 95% of universe | ABORT |
| `^GSPC` and `^VIX` available | Required | HOLD: retain prior `scalar`, flag |
| Daily price change | > 50% | Exclude ticker, flag (§10.1: adjusted series, so this is never a split) |
| Signal date is a trading day | Required | ABORT |
| Sum of target weights | == sleeve_usd ± $1 | ABORT |
| No weight exceeds cap | ≤ $2,500 + $1 | ABORT |
| Positions in target portfolio | 8–20 | < 8 → ABORT; 8–19 → WARN |
| Cash after execution | ≥ 0 | ABORT |

Logging: structured JSON, levels `INFO`/`WARN`/`ABORT`, with rotation. Every ABORT carries full context sufficient for reproduction.

---

## 14. Phase 0: Backtest Protocol

**Duration:** 1–2 weeks. **Runs before any scheduler is enabled.**

### 14.1 What this backtest is and is not

It is a **mechanical validation harness**. Its job is to answer questions about plumbing, not about profitability:

- Does turnover land in the 150–200% range the hysteresis was designed to produce?
- How often does the sector cap starve the portfolio below 20 positions?
- How often is the regime overlay active, and does it engage during the episodes it was designed for?
- How many names does each filter remove, and are any of them inert?
- Does the code survive 15 years of splits, delistings, index changes, and missing data?

It is **not** a return estimate. Any CAGR or Sharpe the backtest produces is contaminated by the biases in §14.2 and should not be quoted, remembered, or used to set expectations. The most disciplined approach is to configure the reporting so that headline return figures are suppressed by default and only risk/mechanics metrics are printed.

### 14.2 Bias inventory

Each of these must be either eliminated or explicitly acknowledged with a magnitude estimate. Silent acceptance is what makes most retail backtests worthless.

| Bias | Mechanism | Magnitude | Treatment |
|---|---|---|---|
| **Survivorship** | Today's S&P 500 membership applied to history; failed firms absent | **+1 to 3% p.a.** | Eliminate with point-in-time data (§14.3) or acknowledge and suppress return output |
| **Delisting** | yfinance simply ends the series; the terminal loss is never realized | **+0.5 to 1.5% p.a.** | Apply a −30% delisting return for performance-related delistings (CRSP convention) |
| **Look-ahead: index membership** | A name added to the index in 2019 appears eligible in 2015 | Large, overlaps survivorship | Point-in-time membership only |
| **Look-ahead: fundamentals** | yfinance returns as-of-today financials with no reporting lag | Moderate | Lag quarterly data 45 days, annual data 90 days |
| **Restatement** | Fundamentals reflect restated, not originally reported, figures | Small for F4 | Acknowledge; F4 is deliberately loose |
| **Data snooping** | Testing many parameter sets on one history | Potentially total | Governed by §14.5 |
| **Price adjustment drift** | yfinance retroactively re-adjusts historical prices | Small | Snapshot the price panel once to Parquet; never re-download mid-study |

The last point is easy to overlook and quietly destroys reproducibility. **Download the full price panel once via `YahooDataDownloader.get_ohlcv_batch()` (§2), write it to `backtest/data/prices.parquet` with a fetch timestamp, and run every subsequent experiment against that frozen file.** Otherwise two runs of the same code a week apart produce different results and you will not know why.

Note `backtest/` is deliberately outside `results/p21_momentum/` (§3): it is a one-time frozen research
snapshot for Phase 0 parameter/robustness work, not run history from the live paper pipeline. The two
should not be confused — nothing under `backtest/` is regenerated by `monthly_rebalance` / `monthly_execute`
/ `daily_mark`.

### 14.3 Universe construction — three options

| Option | Cost | Survivorship handled | Recommendation |
|---|---|---|---|
| **A. Current S&P 500 applied backward** | Free | No | Use only if return output is suppressed |
| **B. Point-in-time constituents** | ~$50–70/month (Norgate Data, Sharadar via Nasdaq Data Link) | Yes | **Recommended if returns matter to you at all** |
| **C. Reconstructed liquid universe** | Free but laborious | Partially | Not worth the effort at this scale |

If Option A is chosen, the backtest report must carry a banner on page one:

> UNIVERSE: current S&P 500 constituents applied retroactively. Return figures in this report are upward-biased by an estimated 1–3% annually and are not usable as forecasts. Mechanical metrics (turnover, position count, filter attrition, regime frequency) are unaffected by this bias and are the intended output.

Mechanical metrics genuinely are largely unaffected — turnover and sector concentration do not depend much on which names existed. That is why Option A remains useful despite the bias.

### 14.4 Period and data specification

```yaml
backtest:
  start: 2005-01-01          # captures 2008-09 and the 2009 momentum crash
  end:   2026-06-30
  frequency: monthly rebalance, daily marking
  price_panel: backtest/data/prices.parquet   # frozen snapshot
  calendar: XNYS
  initial_nav: 250_000
  warmup: 2005-01-01 to 2006-02-01  # 260 bars needed before first signal
```

Rationale for the 2005 start: it includes the 2008–09 bear market and, critically, the **March–May 2009 momentum crash**, which is the single most informative episode for evaluating the regime overlay. A backtest starting in 2010 omits the exact scenario the overlay exists to handle and is therefore close to worthless for this purpose.

Handling of missing data:
- Forward-fill prices for a maximum of 3 trading days; beyond that, treat the name as untradeable that month
- Never back-fill
- A name that disappears mid-month: liquidate at the last available price with a −30% haircut if the disappearance is performance-related, at last price otherwise; log `EXIT_DELISTED`

### 14.5 Parameter robustness — the discipline section

This is where backtests are most often ruined. The rules below are more restrictive than they may appear necessary, deliberately.

**Rule 1: Do not optimize the core signal.** `LOOKBACK_START=252`, `SKIP_RECENT=21` come from three decades of published literature across multiple markets and asset classes. They are not free parameters. Test them to confirm the implementation behaves sensibly, never to select a better value.

**Rule 2: Examine the surface shape, not the maximum.** Run the grid below, then plot each parameter's marginal effect. A parameter sitting on a broad plateau is robust. A parameter sitting on a narrow peak is overfitted — and the correct response is to move to the plateau's center, not to the peak.

```yaml
robustness_grid:
  lookback_start:  [189, 252, 315]      # 9, 12, 15 months
  skip_recent:     [10, 21, 42]
  entry_rank:      [10, 20, 30]
  hold_rank:       [40, 60, 100]
  max_per_sector:  [3, 4, 6]
  vix_threshold:   [24, 28, 32]
```

**Rule 3: Count your tests.** The grid is 3^6 = 729 combinations. Over ~20 years of monthly data (~245 observations), this is heavy data mining. Apply a deflated Sharpe adjustment (Bailey & López de Prado): with 729 trials, the expected maximum Sharpe under a true null of zero is roughly 0.9–1.1 **purely from chance**. Any single configuration reporting Sharpe below that band carries no evidence whatsoever.

Practical decision rule: **if the best configuration's Sharpe is not clearly separated from the median of the top quartile, treat the whole surface as flat and use the literature defaults.** This will usually be the outcome, and it is the correct outcome.

**Rule 4: Split the sample, and touch out-of-sample once.**

```
In-sample:      2005-01 → 2016-12   (parameter inspection)
Out-of-sample:  2017-01 → 2026-06   (single evaluation, no iteration)
```

If you look at the out-of-sample period, adjust something, and look again, the split has been consumed and you are back to pure in-sample. Log every out-of-sample evaluation with a timestamp in `backtest/oos_access_log.md` so this cannot happen accidentally.

### 14.6 Stress windows — evaluated individually

Each window is reported separately with tracks A, B, C, D, E, and the realized `regime_scalar` path. These are the episodes that determine whether the overlay earns its complexity.

| Window | Event | Question it answers |
|---|---|---|
| **2009-03 → 2009-05** | Momentum crash, ~−70% for the academic factor | **The decisive test.** Does the overlay reduce the drawdown? If A−B is not strongly positive here, the overlay has no reason to exist |
| **2008-09 → 2009-02** | Bear market, slow decline | Does the overlay de-risk with reasonable lag? |
| **2011-08 → 2011-10** | Volatility spike, no sustained bear | Whipsaw test — does the overlay incur cost for no benefit? |
| **2015-08 → 2016-02** | Two corrections, rapid reversals | Whipsaw test |
| **2018-10 → 2018-12** | Fast drawdown, sharp recovery | Overlay likely hurts here; quantify the cost |
| **2020-02 → 2020-04** | COVID crash, 23 sessions peak-to-trough | Overlay expected to fail (too fast). Confirm and size the failure |
| **2020-11** | Vaccine rotation, ~−15% momentum month | Overlay cannot help; measures raw factor fragility |
| **2022-01 → 2022-10** | Slow bear market | Second decisive test — the overlay's best-case scenario |
| **2023-01 → 2023-12** | MTUM ~+9% vs S&P ~+26% | Rebalance-timing failure after a bear year. Does the 20-name version with monthly rebalancing recover faster than the semi-annual ETF? |

Interpretation guidance: the overlay is expected to **help materially in 2009 and 2022**, **hurt modestly in 2011, 2015–16, 2018, and 2020**, and be **neutral in 2020-11**. If that pattern does not appear, either the implementation is wrong or the overlay is not doing what it is designed to do. A version that helps everywhere is a version with a bug — most likely look-ahead in the regime signal.

### 14.7 Transaction cost sensitivity

Run the full backtest at four slippage levels:

```
slippage_bps: [0, 3, 10, 25]
```

- **0 bps** — theoretical ceiling, for reference only
- **3 bps** — base case for $2,500 orders in S&P 500 names
- **10 bps** — pessimistic
- **25 bps** — stress; approximates a mid-cap universe or a much larger sleeve

**Acceptance test: if the edge over track C disappears between 3 and 10 bps, the strategy is not viable.** A real edge should degrade gracefully, not fall off a cliff. This test is more informative than the base-case return, because it directly measures how much of the apparent edge is an artifact of frictionless assumptions.

Additionally compute a **cost-neutral turnover curve**: re-run with `hold_rank` in `[20, 40, 60, 100, 150]` at 10 bps and plot net return against realized turnover. The maximum of that curve is the economically correct hysteresis setting. If it lands far from 60, update the specification — this is one of the few parameters worth tuning, because it trades directly against a cost that is measurable rather than estimated.

### 14.8 Required metrics

Computed per track (A, B, C, D, E) and per stress window:

**Mechanical — primary output, bias-resistant**
- Annualized turnover (two-way), by month and full-period distribution
- Position count: mean, min, and percentage of months below 20
- Sector concentration: max sector weight, Herfindahl index over time
- Filter attrition: names removed by F1–F6, monthly, absolute and percentage
- `regime_scalar` histogram, and number of state transitions
- Holding period: median, mean, distribution
- Trade size distribution and commission as a percentage of trade value
- Count of `WARN_UNDERFILLED`, `MANUAL_REVIEW`, `EXIT_DELISTED` events

**Risk — secondary, moderately bias-resistant**
- Annualized volatility
- Maximum drawdown, drawdown duration, time to recovery
- Rolling 12-month tracking error versus track C
- Beta and correlation versus SPY, rolling 24-month
- Worst month, worst quarter
- Downside deviation

**Return — report with the §14.3 banner, or suppress entirely**
- CAGR, Sharpe, Sortino, information ratio versus C
- Rolling 36-month excess return over C
- Hit rate versus C, monthly

### 14.9 Phase 0 acceptance criteria

Phase 0 passes when all of the following hold. Failures point to specific fixes, not to abandoning the project.

| # | Criterion | Threshold | Response on failure |
|---|---|---|---|
| B1 | Median annualized turnover | 140–210% | Re-tune `hold_rank` via §14.7 curve |
| B2 | Months with < 20 positions | < 20% of months | Raise `fallback_pool_rank` above 40 |
| B3 | Months with < 12 positions | < 3% of months | Sector cap too tight; consider 5 |
| B4 | Max sector weight breach | Never | Bug in `enforce_sector_cap` |
| B5 | Overlay benefit in 2009-03→05 | A−B > +5% | Overlay not functioning; check regime lag |
| B6 | Overlay benefit in 2022 | A−B > +3% | Same |
| B7 | Overlay cost in whipsaw windows | A−B > −4% each | Overlay too twitchy; raise `vix_threshold` |
| B8 | Edge over C survives 10 bps | Yes | **Stop. Buy QDVA.** |
| B9 | Runtime, full backtest | < 30 minutes | Optimize before scheduling |
| B10 | Two identical runs → identical output | Bit-identical | Non-determinism present; find and fix before proceeding |

B10 deserves emphasis. Any non-determinism — dictionary ordering, unstable sorts on ties, unfrozen data — will make the live pipeline impossible to debug later. Enforce deterministic tie-breaking (sort by `signal` descending, then by ticker ascending) everywhere.

### 14.10 Deliverables

```
backtest/
  data/prices.parquet              # frozen panel, with fetch timestamp
  data/constituents.parquet        # point-in-time if Option B
  results/base_case/
    nav_daily.csv                  # all 5 tracks
    trades.jsonl
    monthly_metrics.csv
    stress_windows.md
  results/robustness/
    grid_729.csv                   # one row per configuration
    marginal_surfaces.png          # per-parameter plateau plots
    deflated_sharpe.md
  results/cost_sensitivity/
    slippage_0_3_10_25.csv
    turnover_net_return_curve.png
  oos_access_log.md
  PHASE0_REPORT.md                 # acceptance table + banner
```

`PHASE0_REPORT.md` leads with the §14.9 acceptance table and the §14.3 bias banner. It does not lead with performance.

### 14.11 What Phase 0 does not resolve

Even a flawless Phase 0 leaves the central question untouched. A backtest cannot tell you whether the momentum factor will pay over your holding period; it can only tell you that your implementation of it is mechanically sound and economically survivable at realistic costs. The factor has been published since 1993, is widely harvested, and has delivered materially weaker returns post-2009 than in its published sample. No amount of backtesting addresses that. Phase 0 answers "does my code work"; nothing answers "will this work."

---

## 15. Deployment Sequence

**Phase 0 — backtest.** See §14. Passes on the §14.9 acceptance table.

**Phase 1 — dry run (1 month).**
Full pipeline, all trades logged only. Inspect every `results/p21_momentum/` file by hand. Focus on ticker normalization and on the adjusted-series bookkeeping in §10.1 behaving as intended.

**Phase 2 — paper (12 months).**
Normal operation. Reports are read; parameters are **not changed**. Any rule change resets the 12-month counter and is recorded in `config/pipeline/p21_changelog.md` with date and rationale.

**Phase 3 — decision per §12.6.** IBKR bid/ask reconciliation (§10.2) and the go-live share-count
reconciliation (§17) are candidates to build at this point, once Phase 2 has produced 12 months of data to
validate the slippage assumption against.

### Parameter freezing
After Phase 1, write all constants to `config/frozen_params.json` with a hash. The agent verifies the hash on every run and reports mismatches prominently. This guards against the most common way to ruin a factor strategy: adjusting rules after a bad quarter.

---

## 16. Parameter Summary

```yaml
signal:
  lookback_days: 252
  skip_recent_days: 21
  min_history_days: 260
  vol_method: weekly_std_annualized
  rank_by: risk_adjusted          # raw_return / vol

filters:
  min_adv_usd: 50_000_000
  adv_window_days: 60
  gap_filter_top3_share: 0.40
  earnings_blackout_days: 5

selection:
  entry_rank: 20
  hold_rank: 60
  max_per_sector: 4
  target_count: 20
  fallback_pool_rank: 40
  tie_break: [signal_desc, ticker_asc]

sizing:
  nav_total_usd: 250_000
  sleeve_target_pct: 0.20
  max_position_pct: 0.01         # of total NAV
  weighting: inverse_vol_capped
  min_trade_usd: 150

regime:
  bear_lookback_days: 252
  ma_days: 200
  vix_smoothing_days: 20
  vix_threshold: 28
  scalar_normal: 1.00
  scalar_bear_lowvol: 0.60
  scalar_bear_highvol: 0.25
  upgrade_confirmation_months: 2

execution:
  fill_at: next_open
  slippage_bps: 3.0
  commission_min_usd: 0.35
  commission_per_share: 0.0035
  commission_max_pct: 0.01

risk:
  catastrophic_stop_pct: -0.35

benchmark:
  proxy_ticker: MTUM
  ter_adjustment_annual: 0.0005  # MTUM 0.15% → QDVA 0.20%
  market_reference: SPY

backtest:
  start: 2005-01-01
  end: 2026-06-30
  in_sample_end: 2016-12-31
  slippage_grid: [0, 3, 10, 25]
  frozen_price_panel: backtest/data/prices.parquet
```

---

## 17. Known Limitations

1. **Survivorship bias in the backtest.** Current S&P 500 membership applied to history overstates returns by 1–3% annually. Live paper trading is free of this — that is its principal advantage over any backtest.
2. **yfinance fundamentals are unreliable.** Filter F4 operates at half strength. If it proves material, a paid source is required.
3. **No tax modeling.** Largely correct for a Swiss resident with tax-free capital gains; dividends (~1% of sleeve, 15% US withholding, reclaimable via DA-1) are ignored as immaterial.
4. **The slippage model is an assumption, unverified for now.** 3 bps is plausible for S&P 500 names at $2,500 order size. IBKR bid/ask reconciliation (§10.2) would supply empirics but is deferred to Phase 3+ — the existing `IBKRDownloader` doesn't cover live bid/ask, only historical bars, so this is new integration work, not a quick win.
5. **MTUM ≠ QDVA.** Same index, different domicile, different internal dividend tax treatment, different settlement currency. Expected divergence within 0.3% annually.
6. **12 months provides no statistical power.** See §0. This is a limitation of the method, not the implementation, and it cannot be engineered away.
7. **Deflated Sharpe is approximate.** The §14.5 adjustment assumes independent trials; grid parameters are correlated, so the true haircut is somewhat smaller. It errs conservative, which is the correct direction.
8. **`shares` is notional, not a literal share count (§10.1, provisional).** Adjusted-close-only bookkeeping means `dollar_allocation / adjusted_price` shifts retroactively whenever yfinance revises a name's adjustment factor (e.g. a newly declared dividend). Fine for paper P&L; a live migration needs a one-time reconciliation step converting notional shares to real order sizes off actual same-day raw prices at the moment of the first live order — not before. This whole tradeoff is marked for revisit per §10.1, not settled.
9. **No corporate-action audit trail (§10.1, provisional).** Because splits/dividends are absorbed into the adjusted series rather than logged as discrete events, there is no `CORPORATE_ACTION_SPLIT`-style record of *when* a held name split or paid a dividend — only the net effect on its adjusted price. If that audit trail becomes valuable (e.g. for tax reporting beyond §17.3's immateriality assumption), it requires the raw-price/`actions` fetch this design deliberately avoided (§2.1) — see §10.1's callout for when to add it back.

---

*This specification governs a paper simulation. Verify §12.6 in full before placing any live order. Educational material; not personalized investment advice.*
