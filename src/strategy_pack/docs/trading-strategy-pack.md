# 📊 Trading Strategy Pack (6–12M Horizon)

## Overview

**Target domains:**
- US / global equities & indices (SPY, QQQ, global ETFs)
- Crypto majors plus liquid altcoins (BTC, ETH, top 10–20 by market cap)

**Core constraints:**
- Systematic, formal rules only — no gut feeling or signal-chat copy-paste
- Prefer momentum, trend-following, lazy rebalancing
- **All execution is manual** — the system emits notifications (email / Telegram) when conditions are met; you decide whether and when to act

**Avoid:**
- Complex mean-reversion / pairs strategies
- Over-optimized multi-parameter systems
- Ultra-high-frequency / nanosecond arbitrage

---

## 🔔 Notification-First Execution Model

> **No automated buy/sell orders are placed by the system.**
> Instead, every signal triggers a notification sent to the trader via **email** and/or **Telegram**.

### Notification contents (for every signal)
Each alert should include:

| Field | Example |
|---|---|
| **Strategy name** | Momentum-Growth Portfolio |
| **Asset** | BTC-USD |
| **Signal type** | BUY / SELL / REBALANCE |
| **Trigger condition** | Price crossed above SMA(200) |
| **Current price** | $84,200 |
| **Suggested action** | Enter long — target weight 5% of portfolio |
| **Stop-loss level** | $71,570 (15% trailing) |
| **Timestamp (UTC)** | 2026-04-14 00:00 UTC |
| **Urgency** | Low (weekly batch) / Medium (daily) / High (intraday) |

### Delivery channels

**Email:**
- Use any SMTP-compatible service (Gmail, SendGrid, Mailgun, etc.)
- Subject format: `[SIGNAL] {STRATEGY} — {ASSET} — {SIGNAL_TYPE}`

**Telegram:**
- Create a private bot via @BotFather
- Send alerts to your personal chat or a private channel
- Format: plain text or Markdown with the fields above

### Recommended tooling

**Minimal / generic:** `smtplib`, Telegram Bot API over `requests`, host `cron`, or Airflow. No live order routing is required for this pack — alerts only.

**This codebase (reuse for integration):**

| Concern | Where to reuse |
|--------|----------------|
| Alerts (email / Telegram) | `src/notification/service/client.py` — `NotificationServiceClient`, `MessageType`, `MessagePriority` (same integration pattern as `src/trading/strategy_instance.py`, `src/screeners/logic/notifier.py`) |
| Logging | `src/notification/logger.py` — `setup_logger` |
| Job scheduling | `src/scheduler/scheduler_service.py` — APScheduler with DB-backed schedules and `CronTrigger`; align jobs to candle closes |
| Market data | `src/data/` — providers / `DataManager` / feeds so research and scheduled jobs read the same shapes |
| Vectorised research & JSON rules | `src/vectorbt/pipeline/engine.py` — `StrategyEngine`, `LogicEvaluator`; CLI `src/vectorbt/main.py`; Plotly HTML `src/vectorbt/pipeline/plotter.py`; JSON configs under `src/vectorbt/configs/` |
| Optuna | `src/vectorbt/pipeline/manager.py`, `src/vectorbt/tools/promoter.py`, reporting `src/vectorbt/pipeline/reporter.py` |
| Event-driven backtests / `CustomStrategy` | `src/strategy/custom_strategy.py`, `src/strategy/base_strategy.py` (Backtrader); orchestration `src/backtester/optimizer/` |

> **Import caveat:** the repo contains a **local** tree `src/vectorbt/` (pipeline code) that can shadow the **PyPI** `vectorbt` package if `sys.path` ordering is wrong. Use documented entrypoints from project root so `import vectorbt` resolves to the library for `vbt.Portfolio`. See `.ai/AGENTS.md`.

---

## Operational model (decision clock & scheduling)

**Bar timeframe** is the OHLCV frequency you compute on (e.g. daily, 4h). **Evaluation schedule** is when jobs run and may emit notifications.

| Principle | Guidance |
|-----------|----------|
| Match schedule to decisions | Do not run hourly jobs for strategies that only change on **daily** or **weekly** closes — redundant work and alert fatigue. |
| Align to bar completion | Run a few minutes **after** the official candle close so the bar is final (crypto: UTC; US equities: after regular session using your provider’s daily bar). |
| Edge-trigger alerts | Prefer notifying on **state changes** (e.g. first close above SMA(200)), plus optional slower **digests** for “no change” strategies. |
| Intraday (5–6) | Schedule at the **same** cadence as the signal timeframe (e.g. 4h strategy → job every 4h at bar close), not a faster poll “just in case.” |

---

## Backtesting & component reuse

**Two complementary paths already in the repo:**

1. **Vectorbt (research, grids, multi-symbol matrices)** — Express rules as boolean entry/exit time series (or custom order functions for allocator-style strategies). The JSON-driven `StrategyEngine` / `LogicEvaluator` in `src/vectorbt/pipeline/engine.py` fits MA/RSI/ATR-style rules; extend `IndicatorRegistry` when an indicator is missing (e.g. SuperTrend for Strategy 6).

2. **Backtrader + `CustomStrategy` (live / bar-event parity)** — `src/strategy/custom_strategy.py` wires `entry_logic` / `exit_logic` to mixins under `src/strategy/entry/` and `src/strategy/exit/`. Use this when you want one codepath for **cerebro** backtests and live-like stepping. Mixins assume a `bt.Strategy` instance; they are **not** drop-in for raw pandas. Prefer a shared **signal spec** (JSON or dataclass) or **pure functions** on DataFrames that both vectorbt and Backtrader wrappers call.

**Strategy 1** is a **cross-sectional allocator** (ranks, weights, turnover), not only per-asset long/short flags — implement with portfolio-style simulation (`order_func` / weight targets) in vectorbt or a dedicated small module; keep Backtrader only if you already model basket execution there.

---

## Visualization (per strategy)

| Strategy | Suggested views |
|----------|-----------------|
| **1 — Momentum-Growth** | Top-K membership over time; target vs current weights (stacked bars); turnover; crypto sleeve % of NAV. |
| **2 — Daily MA trend** | Price + SMA(200)/SMA(50); shaded **regime** (trend OK vs flat); markers on **crosses** only. |
| **3 — Weekly lazy** | Weekly OHLC + SMA(52)/SMA(26); regime shading; optional toy equity curve for “in trend = invested”. |
| **4 — Rebalance** | Weight drift vs targets; implied buy/sell notionals; tag assets below SMA(200) for “skip add”. |
| **5 — Swing** | OHLC, Donchian / breakout levels, volume; SL/TP; optional R-multiple histogram from research. |
| **6 — EMA + SuperTrend** | Price, EMA, SuperTrend; discrete position state; in-sample vs out-of-sample equity; Optuna parameter importance when using `src/vectorbt/pipeline/reporter.py`. |

**Tooling:** Plotly HTML from `src/vectorbt/pipeline/plotter.py`; Backtrader charting via `src/backtester/plotter/` for mixin-driven runs.

---

## Signal persistence / audit trail

Persist every **decision** the job computed (even when flat or no notification), for replay, audits, and deduplicated alerts.

**Recommended JSON / JSONL fields:**

| Field | Purpose |
|-------|---------|
| `strategy_id`, `variant` | Pack strategy + variant |
| `symbol` | Ticker / pair |
| `signal` | `BUY` / `SELL` / `REBALANCE` / `FLAT` / etc. |
| `ts_utc` | Job completion time |
| `bar_timeframe`, `bar_close_ts` | Which bar the decision belongs to |
| `price` (or reference close) | Context |
| `reason_code`, `metadata` | Trigger text + structured extras (weights, ranks, SL/TP) |
| `idempotency_key` | Stable hash of `strategy_id` + `symbol` + `bar_close_ts` + `signal` — skip duplicate Telegram/email on retries |

**Storage:** append-only **JSONL** under e.g. `results/signals/` to start; migrate to SQLite or Parquet partitions if volume grows. Before `send_notification`, check last-sent idempotency keys (pattern similar to batching in `src/screeners/logic/notifier.py`).

---

## Simulation assumptions (backtests)

Keep these **fixed and documented** per study so Sharpe/Calmar comparisons stay meaningful:

- **Fees and slippage** — per-side bps; wider for illiquid alts.
- **Signal vs fill bar** — signal on bar **close** vs fill on **next open** avoids lookahead; mixing them inflates results.
- **Adjusted prices** — use split/dividend-adjusted series for equity momentum ranks.
- **Calendars** — do not mix 24/7 crypto with RTH-only equity in one momentum basket without explicit session handling.
- **Leverage (Strategy 3)** — if modelling 1–2x, include funding and realistic margin constraints in simulation code.

---

## Strategy 1 — Momentum-Growth Portfolio (Global + Crypto)

**Style:** Momentum-based, long-only, relatively lazy
**Goal:** Capture winners in US/global equities and crypto over 6–24 months, with monthly or quarterly rebalancing

**Implementation shape:** cross-sectional **rank → weights** pipeline (portfolio allocator), not a single-symbol entry/exit mixin. Prefer dedicated allocation + turnover logic; validate in vectorbt with weight targets or a small custom simulator, and optionally mirror in Backtrader only if you need bar-by-bar basket execution.

### Universe

**Equities:**
- ETFs: SPY, QQQ, VTI, VPL (global), or a custom basket of 20–50 large-cap stocks

**Crypto:**
- Top 10–20 coins by market cap (BTC, ETH, SOL, BNB, etc.)

### Logic

**Universe selection (at rebalance date T):**
1. Compute 3-month and 6-month total return for each asset
2. Keep only those above a market-cap threshold (e.g., top-N coins + large-cap stocks)
3. Rank assets by 6-month momentum (or geometric average of 3- and 6-month returns)
4. Select the top-K (e.g., 10–20) assets as candidates

**Weighting options — choose one:**

- **Equal-weight:** `weight_i = 1 / K` for the top-K
- **Momentum-proportional:** `weight_i ~ momentum_score_i` (normalised)
- **Cap-weighted hybrid:** majority in large-cap ETFs (SPY/QQQ), minority in top crypto momentum coins

### Variants

| Variant | Description |
|---|---|
| **A — Price-momentum only** | No indicators; pure past return over 3- and 6-month windows |
| **B — SMA/EMA filter** | Only include assets where price > SMA(200) or EMA(200) on daily data; optional: price > SMA(50) on weekly |
| **C — Volatility-scaled momentum** | Use risk-adjusted momentum (`momentum / recent_vol`) for ranking |

### Risk Management
- Max single-asset weight cap: 5–10%
- Max crypto-only share of total portfolio: 20–40%
- Individual trailing stop per asset: 15–20%

### Monitoring & Notifications

| Parameter | Value |
|---|---|
| **Rebalance frequency** | Monthly (preferred) or quarterly (conservative) |
| **Data needed** | Daily OHLCV for each asset |
| **Live feed required?** | ❌ No — batch is sufficient |
| **Recommended setup** | Cron job running Sunday 00:00 UTC; fetches daily closing prices; computes rankings |

**Notification trigger:** After each rebalance run, send a single summary alert listing:
- Which assets to add / increase (with suggested weights)
- Which assets to reduce / exit
- Overall portfolio drift from targets

---

## Strategy 2 — Simple Trend-Following System (Slow MA / EMA)

**Style:** Price-above-slow-MA, "trend is your friend"
**Goal:** Ride long-term trends in US/global indices and selected crypto with minimal optimisation

### Universe
- **Equities (ETFs):** SPY, QQQ, VTI, VPL
- **Crypto:** BTC, ETH, and optionally 1–2 mega-cap tokens (BNB, SOL)

### Logic

**Trend filter (long only) — checked daily:**
1. Compute daily close
2. Compute SMA(200) or EMA(200) over 200 trading days
3. If `Close > SMA(200)` AND optionally `Close > SMA(50)` → asset is in uptrend

**Entry / exit conditions:**

| Event | Condition |
|---|---|
| **Entry signal** | Both conditions met AND no position currently open |
| **Exit signal** | `Close < SMA(200)` (or EMA(200)) for that asset |
| **Optional tighter exit** | `Close < SMA(50)` for 2 consecutive days |

### Variants

| Variant | Description |
|---|---|
| **A — Pure SMA(200)** | No EMA, no extra filters |
| **B — SMA(200) + EMA cross** | Confirm with EMA(20) > EMA(50) on same timeframe |
| **C — Weekly timeframe** | Use weekly closes and SMA(52) / EMA(52) for a lazier approach |

### Risk Management
- Fixed position size per asset (no leverage, or max 1–2x)
- Optional trailing stop: 15–20%, or fixed-percentage stop per position
- Max assets in portfolio: 5–10; avoid over-diversifying

### Monitoring & Notifications

| Parameter | Value |
|---|---|
| **Evaluation frequency** | Daily |
| **Live feed required?** | ❌ No |
| **Recommended setup** | Script runs once daily after market close (US stocks: ~21:00–22:00 UTC; crypto: 00:00 UTC) |

**Notification triggers:**
- **Entry alert:** asset crosses above SMA(200) → "Consider opening a long on {ASSET}"
- **Exit alert:** asset crosses below SMA(200) → "Consider closing long on {ASSET}"

---

## Strategy 3 — Weekly "Lazy" Trend-Following (1–2x Leverage Optional)

**Style:** Weekly decision-making, multi-month trends, minimal noise
**Goal:** Designed for the "I can wait 2 years" horizon; suitable for moderate leverage (1–2x) if desired

### Universe
- **Equities:** SPY, QQQ, VTI
- **Crypto:** BTC, ETH

### Logic

**Timeframe:** Weekly candles — evaluated once per week (e.g., Sunday evening UTC)

**Trend conditions:**
- `Close_weekly > SMA(52)` (1-year weekly moving average)
- Optionally: `Close_weekly > SMA(26)` (6-month)

**Entry / exit conditions:**

| Event | Condition |
|---|---|
| **Entry signal** | Asset in uptrend at weekly close |
| **Exit signal** | `Close_weekly < SMA(52)` (trend broken) |
| **Optional trailing stop** | 15% measured on weekly closes |

### Variants

| Variant | Description |
|---|---|
| **A — Pure weekly SMA(52)** | Simplest approach |
| **B — EMA(52)** | Smoother, faster reaction |
| **C — Mixed timeframe** | Daily SMA(200) confirms uptrend AND weekly SMA(52) confirms; then enter |

### Risk Management
- Max leverage per asset: 1–2x (via futures or margin, if used)
- Max combined exposure: 100–200% of equity across 4–5 assets
- Per-asset risk cap: 1–2% of equity per position

### Monitoring & Notifications

| Parameter | Value |
|---|---|
| **Evaluation frequency** | Once per week (Sunday) |
| **Live feed required?** | ❌ No |
| **Recommended setup** | Weekly batch; aggregate daily candles into weekly; run Sunday evening |

**Notification trigger:** Sunday evening summary — one alert per asset with status (in uptrend / exited trend) and suggested action for the coming week

---

## Strategy 4 — Long-Term Buy-and-Hold + Dynamic Rebalancing

**Style:** Core portfolio + satellite crypto, periodic rebalancing
**Goal:** "Marathon, not sprint" — control risk and avoid over-exposure over 1–2+ year horizon

### Universe

**Core (equities):**
- One or two broad ETFs: SPY, VTI, VPL, or a custom large-cap basket

**Crypto satellite:**
- BTC + ETH + 1–2 blue-chip alts (e.g., SOL, BNB)

### Initial Allocation (example)

| Bucket | Target weight |
|---|---|
| Equities (core) | 60–80% |
| Crypto (satellite) | 20–40% |
| — BTC (within crypto) | 50–60% |
| — ETH (within crypto) | 20–30% |
| — Alts (within crypto) | 10–20% |

### Rebalancing Rules

Triggered once every 3–6 months:
1. Compute current portfolio weights
2. If any asset or bucket deviates by ±5–10% from target → rebalance
3. Optional momentum filter: if asset is in strong downtrend (price below SMA(200)), skip adding to it rather than buying to rebalance

### Variants

| Variant | Description |
|---|---|
| **A — Time-based** | Rebalance every 3 months, no conditions |
| **B — Threshold-based** | Rebalance only when any asset deviates by more than X% (e.g., 5–10%) |
| **C — Momentum-aware** | Time-based rebalance + slow-MA filter: only add to asset if it's above SMA(200) or EMA(200) |

### Risk Management
- No leverage (pure long-only)
- Cap per asset (e.g., BTC max 20–25% of total portfolio)
- No stop-loss used (true long-term hold); redistribute allocations if trend breaks sharply

### Monitoring & Notifications

| Parameter | Value |
|---|---|
| **Evaluation frequency** | Monthly or quarterly |
| **Live feed required?** | ❌ No |
| **Recommended setup** | Quarterly batch script: reads latest prices, recalculates weights, emits trade list |

**Notification trigger:** Quarterly rebalance report alert listing:
- Current vs target weights per asset
- Suggested buys and sells to return to target
- Any assets flagged as below SMA(200) (skip rebalancing into them)

---

## Strategy 5 — Simple Swing System (1–2 Instruments)

**Style:** Swing / breakout on a small number of liquid instruments
**Goal:** Formal, tested swing system on SPY, BTC-USD, or ETH-USD — not microsecond arbitrage

### Universe
- **Equities:** SPY or QQQ (one instrument)
- **Crypto:** BTC-USD or ETH-USD (futures or spot)

### Timeframes
- Swing: 4-hour or 1-hour charts
- Scalping (optional, higher ops-load): 5- or 15-minute charts

### Signal Variants

**Variant A — Breakout + Volume (hourly)**

Every hour:
1. Compute `High_20` and `Low_20` from the last 20 candles
2. If price breaks above `High_20` AND volume of the breakout candle is above recent average → **Entry signal (long)**
3. Stop-loss: below `Low_20`
4. Take-profit: `entry + 1.5 × ATR(14)` or `entry + 2 × entry-risk`

**Variant B — SMA(20) + Price Pullback (1h / 4h)**

1. If `price > SMA(20)` AND `SMA(20)` is rising → long-only mode
2. Entry signal: price pulls back close to SMA(20) (within ~1%)
3. Exit signal: `price < SMA(20)` OR after fixed N-candle timeout (e.g., 24 × 1h candles)

**Variant C — EMA(9) + EMA(21) Crossover (trend-only)**

1. Long signal: `EMA(9) > EMA(21)` AND `EMA(9)` is rising
2. Exit signal: `EMA(9)` crosses back below `EMA(21)`
3. Shorts: only if explicitly allowed; ideally avoided in this simple design

### Risk Management
- Per-trade risk: 0.5–1% of equity
- Max open positions: 1–2 at a time
- No over-leverage: 1–2x if using futures
- Daily loss cap: 3–5% of equity; stop trading for the day if breached

### Monitoring & Notifications

| Parameter | Value |
|---|---|
| **Evaluation frequency** | Every 1–4 hours (swing); every 5–15 min (scalping) |
| **Live feed required?** | Useful but not mandatory |
| **Recommended setup (low-ops)** | Every 1–4 hours, agent pulls last N candles, checks for active signals, sends notification |

**Notification triggers:**
- **Entry alert:** Signal conditions met → "Entry signal on {ASSET} — {VARIANT} — Price: {X}, SL: {Y}, TP: {Z}"
- **Exit alert:** Exit condition triggered → "Exit signal on {ASSET} — Condition: {reason}"
- **Daily loss cap alert:** If simulated P&L for the day exceeds -3% → "Daily loss cap reached — consider pausing for the day"

---

## Strategy 6 — EMA + SuperTrend on BTC/USDT Spot (Optuna-Optimised)

**Style:** Dual trend-confirmation — both indicators must agree before signalling
**Asset:** BTC/USDT spot
**Goal:** Enter only when EMA and SuperTrend point in the same direction. No reversal guessing, no counter-trend trading.
**Backtest window:** 2024-03-21 → 2026-03-21 (2 full years, no cherry-picked window)

---

### Logic

**Entry conditions:**

| Direction | EMA condition | SuperTrend condition |
|---|---|---|
| **Long** | `Close > EMA(period)` | SuperTrend is **bullish** (price above SuperTrend line) |
| **Short** | `Close < EMA(period)` | SuperTrend is **bearish** (price below SuperTrend line) |

Both conditions must be true simultaneously. If they disagree → no position, stay flat.

**Exit conditions:**
- Either indicator flips → close the position and wait for a new confirmed signal
- Optional: ATR-based trailing stop (separate from SuperTrend's ATR — see parameters below)

---

### Optuna Parameter Search Space

All parameters below are fed to Optuna as a single trial. The optimiser searches for the combination that maximises the objective function on the in-sample period.

| Parameter | Type | Search range | Step / choices |
|---|---|---|---|
| `timeframe` | Categorical | `1h`, `2h`, `4h`, `6h`, `12h`, `1d` | — |
| `ema_period` | Integer | 20 – 300 | 5 |
| `supertrend_atr_period` | Integer | 7 – 50 | 1 |
| `supertrend_multiplier` | Float | 1.5 – 5.0 | 0.25 |
| `long_only` | Boolean | `True`, `False` | — |
| `atr_stop_multiplier` | Float | 1.0 – 3.0 | 0.25 |

**Notes on parameter ranges:**
- `ema_period` 20–300 covers fast (short-term reaction) to slow (primary trend filter) regimes
- `supertrend_multiplier` below 1.5 produces excessive noise; above 5.0 the band barely reacts to price
- `long_only = True` tests whether skipping short signals improves risk-adjusted performance on BTC (historically asymmetric asset)
- `atr_stop_multiplier` is independent of SuperTrend's ATR — it governs the exit trailing stop width

---

### Optimisation Objective

**Primary objective: Sharpe Ratio**
- Preferred over raw return to avoid Optuna selecting over-leveraged lucky streaks
- Alternative: **Calmar Ratio** (`annualised return / max drawdown`) — recommended for crypto due to large drawdown events

**Optuna settings (suggested):**
```
n_trials: 300–500
sampler: TPE (default)
pruner: MedianPruner (prune unpromising trials early)
direction: maximize
```

---

### Train / Validation Split (Overfitting Guard)

> Never expose the validation period to Optuna. Parameters are selected on in-sample data only, then tested once on out-of-sample.

| Period | Dates | Role |
|---|---|---|
| **In-sample (train)** | 2024-03-21 → 2025-09-21 | Optuna optimises here |
| **Out-of-sample (test)** | 2025-09-21 → 2026-03-21 | Single blind validation — never touched during search |

**Acceptance criterion:**
Accept the parameter set only if:
```
out_of_sample_sharpe >= in_sample_sharpe * 0.70
```
If out-of-sample Sharpe drops by more than ~30% relative to in-sample, the parameters are likely overfit — re-examine the search space or widen the train window.

---

### Risk Management
- Position size: fixed fraction of equity per trade (e.g., 5–10% of spot portfolio)
- No leverage (spot only)
- Max 1 open position at a time (BTC/USDT only)
- If `atr_stop_multiplier` is active: trailing stop moves with price, closes position when price retraces by `atr_stop_multiplier × ATR(14)` from the peak

### Monitoring & Notifications

| Parameter | Value |
|---|---|
| **Evaluation frequency** | Matches chosen `timeframe` (e.g., every 4h if `timeframe=4h`) |
| **Live feed required?** | ⚠️ Useful for 1h–2h; batch sufficient for 4h–1d |
| **Recommended setup** | Scheduled script runs at candle close; recomputes EMA and SuperTrend; checks entry/exit conditions |

**Notification triggers:**

- **Entry alert (Long):** `Close > EMA` AND SuperTrend bullish → `"LONG signal — BTC/USDT — Price: {X} — SL: {Y} (ATR stop) — Timeframe: {TF}"`
- **Entry alert (Short):** `Close < EMA` AND SuperTrend bearish → `"SHORT signal — BTC/USDT — Price: {X} — SL: {Y} — Timeframe: {TF}"`
- **Exit alert:** Either indicator flips → `"EXIT signal — BTC/USDT — Reason: {EMA/SuperTrend flip} — Price: {X}"`
- **Flat alert:** Indicators disagree → `"No confirmed signal — staying flat"` (optional, send once per disagreement event, not every candle)

---

## 📌 Summary Table

| # | Strategy | Timeframe | Eval Frequency | Live Feed? | Data Granularity |
|---|---|---|---|---|---|
| 1 | Momentum-Growth Portfolio | 6–24 months | Monthly / quarterly | ❌ No | Daily OHLCV |
| 2 | Simple Trend-Following (SMA/EMA) | Multi-month | Daily | ❌ No | Daily OHLC |
| 3 | Weekly Lazy Trend-Following | Multi-month | Weekly | ❌ No | Weekly (from daily) |
| 4 | Buy-and-Hold + Rebalancing | 12–24 months | Quarterly | ❌ No | Daily / weekly |
| 5 | Swing System (1–2 instruments) | Days–weeks | Every 1–4 hours | ⚠️ Useful | 1h / 4h OHLCV |
| 6 | EMA + SuperTrend BTC/USDT (Optuna) | Hours–days | Per candle close | ⚠️ Useful for ≤2h | 1h–1d OHLCV |

---

## Implementation status (tracked)

| Item | Status | Location / notes |
|------|--------|------------------|
| Signal schema (`PackSignal`) | Done | `src/strategy_pack/models.py` |
| JSONL audit log (daily file) | Done | `src/strategy_pack/io.py` → `results/signals/pack_signals_YYYYMMDD.jsonl` |
| Dedup store for notifications | Done | `results/signals/dedup_cache.json` |
| Default universes / parameters | Done | `config/strategy_pack/default.json` |
| Data loading | Done | `DataManager.get_ohlcv` via `src/strategy_pack/strategies.py` |
| Strategies 1–6 signal rules (v1) | Done | `src/strategy_pack/strategies.py` (simplified vs prose spec; see per-strategy notes below) |
| CLI runner | Done | `python -m src.strategy_pack` → `src/strategy_pack/run.py` |
| Logging | Done | `setup_logger` on all modules |
| Notifications | Done | `NotificationServiceClient` + `src/strategy_pack/notify.py` (`MessageType.ALERT`) |
| APScheduler / DB job registration | Not done | Wire `run` subcommand from `src/scheduler/scheduler_service.py` when you are ready (same pattern as other batch jobs). |
| Vectorbt / Optuna wiring for pack | Not done | Still use existing `src/vectorbt/` pipelines separately; Strategy 6 Optuna search not duplicated here. |
| Unit tests | Partial | `tests/test_strategy_pack_models.py` |

**Per-strategy implementation notes (v1):**

- **SP-1:** Momentum rank on configured universe; equal-weight top-K in one `REBALANCE` row (`symbol`: `__PORTFOLIO__`). No live market-cap filter (universe is config-driven).
- **SP-2:** Daily SMA(200); optional fast SMA confirm via `use_sma_fast_confirm` in JSON. `BUY`/`SELL` only on cross; `STATUS` rows are audit-only (`notify_recommended: false`).
- **SP-3:** Daily data resampled to `W-SUN`; SMA(52) on weekly close; same cross vs `STATUS` pattern as SP-2.
- **SP-4:** Emits one `REBALANCE` advisory with targets + per-symbol SMA(200) skip flags. Optional `current_weights` in config for future drift math (not required for v1).
- **SP-5:** Variants **A** (Donchian + volume), **B** (SMA20 pullback), **C** (EMA9/21 cross). Single `symbol` in config. Pass `-v B` etc.
- **SP-6:** EMA + SuperTrend (SuperTrend ported from `run_plotter` logic). Signals `LONG` / `SHORT` (if not `long_only`) / `EXIT` / `STATUS`. Params in `strategy_6` JSON block.

---

## How to run each strategy (CLI)

From the **repository root** (`e-trading`), with Python env activated and data keys / cache paths configured like the rest of the app:

```bash
python -m src.strategy_pack run [flags...]
# equivalent:
python -m src.strategy_pack [flags...]
```

**Global flags**

| Flag | Meaning |
|------|---------|
| `--config` / `-c` | JSON config (default: `config/strategy_pack/default.json`) |
| `--strategy` / `-s` | `1`–`6`, comma list (`2,3`), or `all` |
| `--variant` / `-v` | Variant label; for Strategy 5 use `A`, `B`, or `C` |
| `--dry-run` | Log signals only; no JSONL, no notifications |
| `--no-notify` | Write JSONL but skip `NotificationServiceClient` |
| `--no-jsonl` | Skip append to JSONL |

**Prerequisites**

- `DataManager` providers (e.g. Yahoo / Binance) must work for your symbols; see `config/data/provider_rules.yaml`.
- Notifications: set `NOTIFICATION_SERVICE_URL` if not using default `http://localhost:5003`, or rely on client DB fallback as elsewhere in the project.

**Strategy 1 — Momentum-Growth (SP-1)**

```bash
python -m src.strategy_pack run -s 1 -v A
```

Edit `strategy_1` in `config/strategy_pack/default.json` (`symbols`, `top_k`, `lookback_days`, `use_vol_scaled`).

**Strategy 2 — Daily SMA trend (SP-2)**

```bash
python -m src.strategy_pack run -s 2 -v A
python -m src.strategy_pack run --dry-run -s 2
```

Edit `strategy_2` (`symbols`, `sma_slow`, `sma_fast`, `use_sma_fast_confirm`).

**Strategy 3 — Weekly lazy trend (SP-3)**

```bash
python -m src.strategy_pack run -s 3 -v A
```

Edit `strategy_3` (`weekly_sma`, `weekly_sma_fast`, `symbols`).

**Strategy 4 — Rebalance advisory (SP-4)**

```bash
python -m src.strategy_pack run -s 4 -v A
```

Edit `strategy_4.targets` and optional `current_weights` object (same keys as targets) when you track a real portfolio.

**Strategy 5 — Swing (SP-5)**

```bash
python -m src.strategy_pack run -s 5 -v A
python -m src.strategy_pack run -s 5 -v B
python -m src.strategy_pack run -s 5 -v C
```

Edit `strategy_5` (`symbol`, `timeframe`, `donchian`, etc.).

**Strategy 6 — EMA + SuperTrend (SP-6)**

```bash
python -m src.strategy_pack run -s 6 -v A
```

Edit `strategy_6` (`timeframe`, `ema_period`, `supertrend_*`, `long_only`, ATR stop fields).

**Run several at once**

```bash
python -m src.strategy_pack run -s 2,3,6 -v A
python -m src.strategy_pack run -s all --no-notify
```

**Windows PowerShell (same commands)**

```powershell
Set-Location c:\dev\cursor\e-trading
python -m src.strategy_pack run -s 2 --dry-run
```

**Scheduler (manual integration)**

Until a DB schedule exists, use Windows Task Scheduler or cron to invoke the same `python -m src.strategy_pack run ...` line at the cadence described in **Operational model** above. For in-app scheduling, register a job that runs this module with the desired `-s` / `-v` flags.

---

## Implementation plan (e-trading)

Phased work so scheduling, logging, notifications, data, and research stay aligned with existing modules. Each phase should ship a thin vertical slice (config + one strategy working end-to-end) before expanding breadth.

### Phase 0 — Contracts & plumbing

- **Signal schema** — Done: `PackSignal` in `src/strategy_pack/models.py`.
- **JSONL writer** — Done: `src/strategy_pack/io.py`, files under `results/signals/pack_signals_YYYYMMDD.jsonl`.
- **Logging** — Done: `setup_logger` in pack modules.
- **Config** — Done: `config/strategy_pack/default.json`.

### Phase 1 — Scheduling

- **Persisted cron** — Register jobs via `src/scheduler/scheduler_service.py` / DB schedules (todo).
- **CLI entrypoint** — Done: `python -m src.strategy_pack run ...` (`src/strategy_pack/run.py`, `__main__.py`).
- **Cadence table** — Documented above; operator maps cron to `-s` values.

### Phase 2 — Data + signal computation

- **OHLCV** — Done: `DataManager` in `strategies.RunContext`.
- **Pure functions** — Done: `run_strategy_*` in `src/strategy_pack/strategies.py` (I/O only in loader helpers used by runners).
- **Strategy 1** — Done (v1 rank / top-K weights).
- **Strategies 2–4** — Done (v1 rules). Strategy 5–6 intraday rules implemented (v1).

### Phase 3 — Notifications (notification-first)

- **Client** — Done: `send_pack_notifications` in `src/strategy_pack/notify.py`.
- **Formatting** — Done: `format_signal_message`.
- **Dedup** — Done: `DedupStore` JSON file `dedup_cache.json` beside JSONL.

### Phase 4 — Research & optimisation

- **Vectorbt** — Encode Strategies 2, 5 (subsets), and 6 as JSON logic + engine where possible; add missing indicators to `IndicatorRegistry` only when justified.
- **Optuna** — Strategy 6: reuse `src/vectorbt/pipeline/manager.py` study flow; enforce train/holdout split from this doc in study config, not in trial code scattered ad hoc.
- **Backtrader parity (optional)** — For strategies you might later trade via `CustomStrategy`, add a JSON → mixin mapping test that asserts the same signals as the pandas path on a fixed CSV fixture.

### Phase 5 — Visualization & operator UX

- **HTML** — Pipe research outputs through `src/vectorbt/pipeline/plotter.py` (or notebooks) using the **Visualization (per strategy)** table above as a checklist.
- **Dashboard (optional)** — Later: read JSONL and render markers; keep out of scope until Phases 0–3 are stable.

### Phase 6 — Execution bridge (optional, out of pack default)

- If you later **enable** execution, route through `src/trading/` / `StrategyInstance` / brokers — **not** required for this document’s notification-only model.

### Success criteria (minimal)

- Scheduled runs for at least **Strategy 2** (daily) and **Strategy 3** (weekly) produce JSONL + deduped notifications using production notification + logging paths.
- One **vectorbt** report for a pack-derived config saved under `results/vectorbt/` with documented fee/slippage assumptions.

---

> **All strategies operate in notification-only mode. The system never places orders automatically. You receive an alert, review it, and decide whether to act.**
