# CR-2026-11-06 — P17 signals arrive after the price already jumped

| | |
|---|---|
| Status | Phase 1 deployed to the Pi 2026-11-07 (`fee6fb5`, `ab04148`); Phase 2 (stale OHLCV) implemented 2026-11-07, not yet deployed |
| Reported by | Owner, 2026-11-06 |
| Scope | `src/ml/pipeline/p17_penny_stocks/`, `src/ml/pipeline/signal_ledger/` |
| Evidence base | 112 prod runs, 2026-05-15 → 2026-11-05 (`R:\apps\e-trading\results\p17_penny_stocks`) |

## 1. Problem

Tier A/B and EXPLOSIVE notifications for penny stocks arrive when the stock has already run, so entering on the
alert rarely makes sense. Goal: get actionable signals **before** the price jumps, or at least stop alerting on
names whose move is over.

## 2. Findings

### 2.1 Signals fire after the move
78 Tier A/B/explosive signal rows across the 112 runs collapse to 45 episodes (first signal per ticker, 14-day
cooldown) on 40 tickers.

| At first signal | 25th pct | median | 75th pct |
|---|---|---|---|
| 5-day return already made | +27% | **+43%** | +90% |
| Run-up from 20-day low | +44% | **+67%** | +128% |
| Relative volume | 3.5× | 6.4× | 61× |

73% of signals had already gained > 30% in 5 days.

### 2.2 Entering at the next open loses money
42 episodes with prices, entry at the next session's open after the run date:

| Horizon | median | win rate |
|---|---|---|
| +5 sessions | −4% | 37% |
| +10 sessions | −9% | 28% |
| +20 sessions | −12% | 22% |

The overnight gap from signal close to next open is ~0% (median −0.7%), so **delivery timing (08:00 UTC email) is not
the cause** — the loss comes from the multi-day run before the signal and the mean reversion after it.

### 2.3 Root causes in the code
1. **Mandatory EXPLOSIVE / Tier A conditions describe the breakout day itself** (`scoring_agent.py`): RVOL ≥ 3,
   close above the prior 20/50-day high, above SMA50. These are true only once the move is visible.
2. **Score rewards completed momentum.** Momentum (weight 0.25) saturates at +50% 20-day return and is penalised only
   above +300%; volume (0.20) saturates at 5× RVOL. 45% of the score pays for a move that already happened.
3. **Universe filter excludes quiet names** (`universe_agent.py`). Yahoo's 3-month average volume crosses 500k only
   after a name gets active: `Avg volume ≥500000: 612 → 158` on 2026-11-05. Five runs before their signal, of 26
   traceable episodes: 11 were removed by universe filters (market cap 4, dollar volume 3, float 2, average volume
   2), 8 were missing from the fetched snapshot (Yahoo fetch failures), 7 were scored but only Tier W/C.
4. **Daily-close only** — `target_date` = yesterday; the earliest possible alert is the morning after the breakout bar.
5. **Stale OHLCV** (`market_agent.py`) — the unified cache is accepted when its last bar is within **5 calendar days**
   of `target_date`. Matching 400 sampled prod rows against fresh prices: 53% used the target-date bar, **47% used a
   bar 1–3 sessions old** (lag 1: 66, lag 2: 45, lag 3: 39 of 328 matched). Signals were therefore computed on old
   data and arrived later still.

## 3. Options considered

| # | Option | Decision |
|---|---|---|
| 1 | Extension gate: no alert once the move is done | **Done** (Phase 1) |
| 2 | Pre-breakout "setup" score | **Done, shadow mode** (Phase 1) — no edge in replay |
| 3 | Scheduled catalysts known in advance (PDUFA / AdCom, earnings dates, Form 4 buys, 13D) | Planned |
| 4 | Two-tier universe: looser liquidity for a watchlist, strict for the alert | Planned |
| 5 | Intraday trigger via P19 on the watchlist | Planned |
| — | Fix stale / partial OHLCV from the shared cache (root cause 5) | **Phase 2, implemented** |

## 4. Phase 1 — delivered (`fee6fb5`, docs `ab04148`)

### 4.1 Extension gate
`extended = price_5d_return > 0.30 OR (close − SMA20) / ATR14 > 3.0`
(`P17ScoringConfig.max_alert_5d_return`, `max_alert_extension_atr`). Extended candidates are capped at Tier C (they stay
on the P19 intraday watchlist, which reads A/B/C) and are never EXPLOSIVE; signal `extended_5d_<ret>_<atr>atr`.

### 4.2 Pre-breakout setup score (shadow)
0–100 score: coil 0.30 (tight 10-day range, Bollinger squeeze bonus), proximity to prior 50-day high 0.25, early
volume pickup (RVOL 1.3–2.0) 0.20, accumulation 0.15, short-squeeze fuel 0.10. `setup_candidate` requires score ≥ 60,
5-day return ≤ 15%, 20-day return in [−15%, +30%], within 15% below the 50-day high, RVOL ≤ 4, dilution ≤ 30, not
extended. Written to CSV / JSON / Markdown and the signal ledger (`p17_setup` stream); **not emailed**.

New Candidate fields: `dist_to_50d_high`, `extension_atr`, `range_10d_pct`, `extended`, `setup_score`,
`setup_candidate`. Scheduler result adds `extended_count`, `setup_count`.

### 4.3 Point-in-time replay (`replay.py`)
For each saved run date D: universe, fundamentals, short interest, dilution and catalyst score come from that day's
candidates CSV; technicals are recomputed from OHLCV truncated at D (same 100-day window as prod). Entry at next open.

| Set | Episodes | 5d run-up at signal | fwd10 median / mean | fwd10 win |
|---|---|---|---|---|
| Universe baseline (all rows) | 10,395 | −1% | −3.6% / −2.4% | 38% |
| Prod A/B as recorded | 43 | +42% | −9.1% / −5.8% | 30% |
| Replay, gate off | 59 | +35% | **−8.0%** / −4.6% | 34% |
| **Replay, gate on** | 20 | +11% | **+0.4% / +4.0%** | **53%** |
| Setup (shadow) | 59 | +4% | −2.2% / −0.8% | 46% |

Split-half check (split 2026-08-12), fwd10 median: gate on +0.1% / +0.4%; gate off −8.4% / −2.1%; setup +1.5% / −3.0%.

- **Gate:** consistent improvement in both halves; removes about two thirds of alerts. n = 20 — encouraging, not proven.
- **Setup:** no edge. Sign flipped between halves; only 7/59 later jump episodes were preceded by a setup flag within 14
  days; setups reached +20% within 10 sessions *less* often than the universe (14.6% vs 19.1%). Daily-bar coil features
  do not anticipate penny-stock jumps. Thresholds were deliberately not tuned on this sample.
- Gate on vs off is the like-for-like comparison (same fresh OHLCV). Prod A/B differs from "gate off" (agreement on 36
  rows) mainly because prod scored many rows on stale bars (root cause 5).

### 4.4 Verification
302 tests pass (P17, signal ledger, P19); pyright and mypy clean; end-to-end smoke run on 9 tickers (CRCT, flagged
EXPLOSIVE by prod at +45% 20-day, now Tier C with `extended_*`). Module README / Requirements / Design / Tasks added.

## 5. Phase 2 — stale and partial OHLCV (implemented 2026-11-07)

### 5.1 Two defects found
1. **Stale bars.** A cached series was accepted when its last bar was within 5 calendar days of `target_date`;
   ~47% of sampled prod rows were scored on 1–3-session-old bars (§2.3 item 5).
2. **Partial intraday bars.** The shared UnifiedCache stores whatever a job downloaded; a daily bar fetched during
   market hours is a half-day bar and later looks "fresh". Observed locally for 2026-11-06 (fetched 20:02 UTC, before
   the close): ANGI cached volume 680k vs 1.39M final and close $7.27 vs $6.92; CRCT 1.57M vs 2.42M; PSIG 377k vs
   743k. RVOL, breakouts and returns computed on such a bar are wrong. The cache has no per-bar "written after close"
   marker, so a freshness check cannot detect this. On prod, 95 of 139 tickers were served from this cache on
   2026-11-05 (the other 44 were downloaded).

A related look-ahead issue: downloads end at "now" with no cut at `target_date`, so historical `--target-date` runs
could see later bars.

### 5.2 Change (`agents/market_agent.py`)
- OHLCV for the post-filter universe (~150 names) is **always batch-downloaded** — about a second per 10 tickers, one
  batch call. Fresh downloads are written back to the cache, which overwrites partial bars (`keep="last"` merge) and so
  also repairs the cache for other pipelines.
- The cache is only a **fallback** for tickers whose download fails, and then only if it reaches the most recent
  NYSE session ≤ `target_date` (XNYS calendar via `pandas_market_calendars`; weekends and holidays roll back).
- Bars after `target_date` are dropped (no look-ahead).
- Logged: tickers that still end before the expected session after download (halts / Yahoo lag) — kept, but flagged.
- New helpers `last_session_on_or_before`, `last_bar_date`, `truncate_to`; tests in `tests/test_market_agent.py`
  (partial-bar regression, stale fallback rejected, weekend/holiday sessions, look-ahead cut, lagging names).

### 5.3 Verification
311 tests pass (P17, P19, signal ledger); pyright and mypy clean. Smoke run on 10 tickers for 2026-11-06: one batch
download, all fresh; the cached ANGI / CRCT / PSIG bars were overwritten with the final Yahoo values.

### 5.4 Not yet verified
The prod cache lives under the Pi's `DATA_CACHE_DIR` (not the `R:\apps\e-trading\data\cache` copy, which holds 18
symbols), so the share of partial bars on prod is unmeasured. Expect the first post-deploy runs to log a full batch
download of ~140 tickers and no cache hits.

## 6. Monitoring after deploy
- `pipeline.log` `Result:` line shows `extended=… setups=…`; expect fewer Tier A/B emails.
- Signal-ledger scorecard: `p17_tier_ab` (gated from 2026-11-07) and `p17_setup` once ≥ 20 cohorts mature.
- Re-run `replay.py` after ~2 months of new runs and compare against the table above.

## 7. Artefacts
- Code: `agents/scoring_agent.py`, `agents/technical_agent.py`, `config.py`, `models/candidate.py`,
  `agents/reporting_agent.py`, `p17_pipeline.py`, `run_p17.py`, `replay.py`; ledger `harvesters.py`, `scorecard.py`.
- Spec: `docs/pipeline-specification.md` §11.1–11.2. Tasks: `docs/Tasks.md`.
