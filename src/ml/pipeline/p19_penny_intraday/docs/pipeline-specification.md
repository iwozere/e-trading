# P19 — Intraday Penny-Stock Spike Monitor — Pipeline Specification

Status: **design / pre-implementation**
Home: `src/ml/pipeline/p19_penny_intraday/`
Seeds from: `brainstorming1.md` (free-data research) and the P17
`intraday_monitor_design.md` proposal.

This pipeline detects **explosive intraday moves in penny stocks while they are
happening** (minute/few-minute cadence) and emits a **single, de-duplicated,
human-readable alert** per name per day at the moment of breakout — something the
daily P17 batch screener structurally cannot do.

> Motivating case — **SCAG (2026-06-24)**: ripped 0.367 → **1.11 intraday
> (+200%)** on ~1,100× volume, then faded to close 0.711, all in one session.
> P17's daily run only "saw" it the next morning (a Tier-B non-alert). Catching it
> requires detecting the break **intraday**.

---

## 1. Locked design decisions (this session, 2026-06-28)

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Intraday data feed | **IBKR Gateway (delayed, free)** primary; Finnhub real-time price cross-check; yfinance fallback | *Updated 2026-06-28 after the feed probe (§13.1):* no free REST tier gives real-time intraday **volume**. IBKR delayed bars **include volume** (1m/5m OHLCV, ~15-min delayed — acceptable), giving real RVOL-so-far. Gateway already deployed (paper, same Pi). See §13.2. |
| 2 | Watchlist universe | **P17 daily output + pre-market gappers/most-active < $5** | Catches overnight movers P17 didn't rank, while staying capped/affordable. |
| 3 | Build sequencing | **Shadow-mode logger first**, then calibrate, then alert | No historical intraday penny data exists; thresholds must be tuned on accumulated data, not guessed. |
| 4 | Social/news sentiment | **Context/enrichment only** (not a trigger) | Sentiment is noisy, laggy, gameable. Trigger on volume + price + fresh 8-K; attach sentiment for color; test its lead later. |

---

## 2. Goals / non-goals

**Goals**
- Detect explosive intraday moves on a capped **watchlist** in near-real-time.
- Fire **one** de-duplicated alert per name per day at breakout (or on escalation).
- Combine a volume/price tripwire with **fresh-8-K** catalyst awareness.
- **Reuse** P17 catalyst / short-squeeze / dilution agents as enrichment.
- Accumulate an intraday **shadow-mode dataset** for later threshold calibration.

**Non-goals**
- Not a universe-wide ranker (P17's job).
- Not an execution/trading system — **alerting only**.
- Not tick-level HFT — minute/few-minute polling suffices for the pump regime.
- Social sentiment is **not** a trigger (context only — decision #4).

## 3. Why a separate pipeline (not a P17 mode)

| Axis | P17 (daily) | P19 (intraday) |
|---|---|---|
| Data | daily EOD bars | 1m/5m bars, live quotes, RVOL-so-far |
| Core logic | rank ~4,000-name universe once | watch a small list for live tripwires |
| Style | stateless batch | **event-driven, stateful** (already-alerted today) |
| Cadence | 1×/day cron | every few minutes during market hours |
| Universe | exhaustive | pre-selected, capped watchlist |

Opposite shapes; forcing a fast stateful loop into P17's 7-stage batch orchestrator
would harm both. They share **components**, not the orchestrator: **P17 produces
the daily watchlist; P19 watches it live**, importing P17 agents as libraries.

---

## 4. Architecture

```
   (daily, pre-market)
  ┌───────────────────────────────────────────────┐
  │ Watchlist Builder                             │
  │  P17 Tier B/C + explosive (results/p17/{date})│──► watchlist.json
  │  + pre-market gappers/most-active  < $5        │     (per trading day,
  │  + dedup, hard filters, cap to N               │      with baseline ctx)
  └───────────────────────────────────────────────┘
                         │
   (intraday, every poll_interval_minutes, market hours)
                         ▼
  ┌──────────────┐  ┌────────────────────┐  ┌──────────────────────┐
  │ Intraday Feed│─►│ Trigger Engine     │─►│ Enrichment (P17)     │
  │ DataManager  │  │ • RVOL-so-far      │  │ catalyst (fresh 8-K) │
  │ Finnhub/Poly │  │ • % from open/prev │  │ short-squeeze        │
  │ 1m/5m + quote│  │ • fresh 8-K today  │  │ dilution (fade flag) │
  └──────────────┘  │ • severity score   │  └──────────────────────┘
        │           └────────────────────┘            │
        ▼                     │                        ▼
  ┌──────────────┐            ▼              ┌──────────────────────┐
  │ Shadow Logger│   ┌────────────────┐     │ Alert Manager        │
  │ (every poll, │   │ State Store    │◄────┤  dedup per name/day  │
  │  all names)  │   │ alerted_today  │     │  + sentiment context │
  └──────────────┘   └────────────────┘     │  Telegram (+email)   │
                                            └──────────────────────┘
```

### 4.1 Watchlist Builder (runs once, pre-market)
- **Sources**: (a) latest P17 dated output (`results/p17_penny_stocks/{date}/` —
  Tier B/C + explosive), (b) a **pre-market gappers / most-active** list filtered to
  the penny range, (c) optional manual pins.
- **Hard filters** (§7) applied here; **cap to N** (e.g. 150–300) — you cannot poll
  thousands of tapes every few minutes (see §13 rate budget).
- **Output**: `results/p19_penny_intraday/{date}/watchlist.json` with per-name
  **baseline context**: avg 30d volume, float, prior close, dilution penalty, short
  interest, any known catalyst, and the intraday **volume-profile baseline** (§4.2).

### 4.2 Intraday Feed
- **Primary: IBKR Gateway** (`IBKRLiveDataFeed` / `IBKRDataDownloader`), delayed mode
  (`reqMarketDataType(3)`), **streaming** 5m bars via `reqHistoricalData(keepUpToDate=
  True)` for the watchlist — one subscription per name within the ~100 market-data
  line budget (§13.2). Delayed bars **carry volume**, so RVOL-so-far is real (just
  ~15-min late). Connects to the same-Pi paper Gateway (`raspberrypi:4002`).
- **Optional cross-check**: Finnhub `/quote` (real-time price, no volume) for a faster
  read on price thrust; **fallback**: yfinance/Polygon via `DataManager`.
- **RVOL-so-far-today** = cumulative volume to now ÷ *typical cumulative volume by
  this minute-of-day*. The intraday **volume profile** (U-shaped) is built from
  recent IBKR/cached intraday history; until enough exists, approximate as
  `daily_avg_volume × intraday_cdf(minute_of_day)`. Shadow mode (§12) accumulates the
  real profile.
- Also compute **% move from today's open** and **% from prior close**.

### 4.3 Trigger Engine (event detection, stateful)
A name fires when configured tripwires cross (all tunable in config):
- **Volume surge**: `rvol_so_far ≥ intraday_rvol_trigger` (e.g. 3–5×) **and**
  cumulative `$-volume ≥ dollar_volume_floor` (liquidity gate).
- **Price thrust**: `|pct_from_open| ≥ intraday_move_trigger` (e.g. +20%).
- **Fresh catalyst**: a bullish 8-K filed **today** → escalates severity and lowers
  the volume/price thresholds.
- **Gate**: require volume **AND** (price thrust **OR** fresh catalyst) to avoid
  pure-illiquidity noise.
- Combine into an **intraday severity score** (§8), reusing P17 sub-scores where
  applicable.

### 4.4 Enrichment (reused P17 agents — §6)
- `CatalystAgent` → fresh 8-K driving the move?
- `ShortSqueezeAgent` → squeeze fuel (SI/float, days-to-cover).
- `DilutionAgent` → **suppress/annotate** names with active ATM/shelf — "pump into a
  dilution wall" is a **fade**, not a long.

### 4.5 Sentiment (context only — §10)
Attach, do not trigger: ticker-mention spikes and FinBERT news sentiment from the
existing `src/common/sentiments/` adapters, surfaced in the alert for color.

### 4.6 Alert Manager + State Store
- **State**: `results/p19_penny_intraday/{date}/alerted.json` — names alerted today
  and at what level → alert **once**, or only on **escalation** to a higher tier.
- **Delivery**: real-time **Telegram** (primary), optional email, via
  `NotificationService` (recipient resolved from `user_id`, as P17 Stage 8).
- **Hard daily alert cap** to prevent storms on chaotic days.

### 4.7 Shadow Logger (Phase 1 — ships first)
Every poll, for **every** watchlist name (whether or not it triggers), append a row
to the shadow store (§12): timestamp, price, RVOL-so-far, % from open/prev close,
$-volume, fresh-8-K flag, sentiment metrics, plus EOD open/high/low/close
backfilled after the close. This is the dataset that makes threshold calibration
(§15) possible.

---

## 5. Pipeline run modes

| Mode | Trigger | Behaviour |
|---|---|---|
| `build-watchlist` | once, pre-market cron | produce `watchlist.json` |
| `run-once` (shadow) | intraday cron (Phase 1) | poll, **log only**, no alerts |
| `run-once` (live) | intraday cron (Phase 2+) | poll, trigger, enrich, alert, persist state |
| `eod-backfill` | once, post-close cron | fill EOD O/H/L/C into shadow rows |

`run-once` is **stateless across invocations** — it loads state/watchlist, polls,
acts, persists state, exits. Crash-safe; reuses the existing scheduler (§13).

---

## 6. Reuse map (do not rebuild)

| Capability | Reuse | Path |
|---|---|---|
| Catalyst (8-K) | `CatalystAgent` | `p17_penny_stocks/agents/catalyst_agent.py` |
| Short squeeze | `ShortSqueezeAgent` | `p17_penny_stocks/agents/short_squeeze_agent.py` |
| Dilution / fade flag | `DilutionAgent` | `p17_penny_stocks/agents/dilution_agent.py` |
| Technical sub-scores | `TechnicalAgent` | `p17_penny_stocks/agents/technical_agent.py` |
| Candidate model | `Candidate` | `p17_penny_stocks/models/candidate.py` |
| OHLCV + provider routing | `DataManager` / `ProviderSelector` | `src/data/data_manager.py` |
| Real-time-ish quotes | Finnhub / Polygon / Tradier / Alpaca downloaders | `src/data/downloader/` |
| Social + news + FinBERT sentiment | `adapter_manager` + async adapters | `src/common/sentiments/adapters/` |
| EDGAR 8-K (intraday + index) | `EdgarDownloader` (`download_8k_filings`, `_efts_search`, `get_recent_filings`) | `src/data/downloader/edgar_downloader.py` |
| AI explain/gate (optional) | P05 Claude synthesizer pattern (Anthropic SDK) | `p05_ai_selector/stages/stage3_llm_synthesizer.py` |
| Alert delivery | `NotificationService` / `UsersService` | `src/data/db/services/` |
| Backtest / Optuna optimize | P17 `backtest.py` / `strategy_sim.py` patterns | `p17_penny_stocks/` |

**New code:** Watchlist Builder, Intraday Feed + RVOL-so-far/volume-profile,
Trigger Engine, State Store, Alert Manager, Shadow Logger, config, models, runner.

---

## 7. Hard filters (watchlist eligibility)

Applied in the Watchlist Builder (penny-pump regime, per brainstorm §2):
- **Price** < `$5` (configurable; brainstorm uses < $5).
- **Float** < `~25M` shares (low float = explosive; `< 10M` flagged "ultra-low").
- **Min liquidity**: daily volume > `~500k` shares (avoid un-tradeable illiquidity).
- US exchanges; exclude ETFs / test issues (reuse P17 universe hygiene).

## 8. Intraday severity score

Normalised 0–100 composite, used for alert tiering and dedup-escalation. Weighted
blend of: RVOL-so-far, % move from open, fresh-8-K catalyst (binary boost),
short-squeeze fuel, with a **dilution penalty** (active ATM/shelf → mark as fade).
Weights/thresholds live in config and are calibrated against the shadow dataset
(§15) — **not hand-tuned at launch**.

## 9. Catalyst — intraday 8-K

- Phase 1: reuse the **daily 8-K index** for "filed today" awareness.
- Phase 2: poll EDGAR **EFTS/RSS intraday** for watchlist CIKs so a same-session
  8-K escalates severity within minutes of filing.
- Bearish items (1.02/1.03/3.01/4.02) never count as bullish (reuse `CatalystAgent`
  classification).

## 10. Sentiment — context only

- Adapters: Reddit (`async_reddit`/`async_pushshift`), StockTwits, ApeWisdom,
  Google Trends (`async_trends`), NewsAPI/Finnhub news, FinBERT (`async_hf_sentiment`).
- Captured **per poll into the shadow log** and **attached to alerts**, never used to
  trigger (decision #4). Once shadow data exists, test mention-spike **lead time**
  vs price move; promote to a trigger only if it demonstrably leads.

## 11. Data model (sketch)

```python
@dataclass
class IntradaySignal:
    ticker: str
    ts: datetime                 # detection time (UTC)
    price: float
    pct_from_open: float
    pct_from_prev_close: float
    rvol_so_far: float
    dollar_volume_so_far: float
    fresh_catalyst: bool         # bullish 8-K filed today
    catalyst_signals: list[str]
    short_squeeze_score: float
    dilution_penalty: float      # >0 → fade risk
    sentiment: dict              # context: mention counts, finbert score, trends
    severity: float              # composite 0–100
    trigger_reason: str          # which tripwire(s) fired
    tier: str                    # alert tier (escalation)
```

## 12. Storage / shadow-mode dataset

- **Per-day artefacts**: `results/p19_penny_intraday/{date}/`:
  `watchlist.json`, `alerted.json` (state), `signals.csv` (fired), `report.md`.
- **Shadow store** (the calibration dataset): append-only per-poll rows for **all**
  watchlist names → SQLite/Parquet table `intraday_shadow`:
  `ts, ticker, price, rvol_so_far, pct_from_open, pct_from_prev_close,
  dollar_volume, fresh_8k, mention_count, finbert_score, trends_score`,
  plus EOD-backfilled `open, high, low, close, day_max`.
- This directly implements brainstorm §3A (3–6 month accumulation sandbox).

## 13. Scheduling, cadence & rate budget

- **Model**: stateless `run-once` on a **short intraday cron** (decision: not a
  daemon). Reuses `job_schedules`; scheduler auto-injects `--user-id`.
- **Window**: US market hours + a pre-market gapper slot; weekdays only.
  Cron in UTC (scheduler is UTC) — encode market hours accordingly, DST-aware.
- **Poll interval**: configurable `5 / 15 / 30 min` (start at 5 during a focused
  window, widen if rate-limited).
- **Rate budget (critical)**: provider free tiers are limited — e.g. Finnhub ~60
  req/min, Polygon free ~5 req/min. With N=300 names and per-symbol calls you cannot
  poll every minute. Mitigations: **cap N**, prefer **batch/snapshot** endpoints,
  stagger symbols across the interval, and cache aggressively in DataManager. The
  watchlist cap, poll interval, and provider limit must be sized together.

### 13.1 Feed probe findings (measured 2026-06-28, `tools/latency_probe.py`)

Empirical free-tier capability matrix (off-hours run; re-run during market hours for
true latency). **This reshapes the trigger design** vs the original "real-time
RVOL-so-far" assumption:

| Provider (free) | Real-time price | Intraday volume bars | Rate limit |
|---|---|---|---|
| **Finnhub `/quote`** | ✅ o/h/l/c/prev-close, ~140 ms, no limit at 15 calls | ❌ `/stock/candle` is **403 premium** | ~60/min |
| **Polygon `/aggs`** | ❌ snapshot **403 premium** | ⚠️ yes but **~15-min delayed** | **~5/min** (429 at 6 calls) |

**Consequence — no free tier gives real-time intraday volume.** Therefore:
- **Trigger on price** via **Finnhub `/quote`** (real-time): `/quote` returns today's
  open/high/low/current + prev-close, so **% from open**, **% from prev close** and
  the intraday high are available live **without candles**. Penny pumps are
  price-led, so this is the primary live tripwire.
- **Treat RVOL/volume as ~15-min-delayed *confirming context***, polled on a slower
  cadence (Polygon `/aggs` or yfinance), not as the millisecond trigger. Backfill
  true volume at EOD for the shadow dataset / calibration.
- **Watchlist sizing:** Finnhub 60/min → a full quote sweep of N≈60 fits a 1-min
  poll, or N≈300 fits a 5-min poll. Polygon 5/min makes per-name volume polling
  impractical at scale — fetch volume for only the *price-triggered* subset.

### 13.2 IBKR Gateway feed (chosen primary — 2026-06-28)

IBKR's **free delayed** market data (`reqMarketDataType(3)`, ~15-min late) returns
**1m/5m OHLCV with volume** — the one thing the free REST tiers lacked — making it
p19's primary feed. The Gateway is already deployed (Docker, same Pi, paper port
**4002**); reuse `IBKRLiveDataFeed` / `IBKRDataDownloader` and `ib_insync`.

**The binding constraint shifts from REST rate limits to IBKR data limits:**

| IBKR limit | Default (scales w/ equity & commissions) | p19 handling |
|---|---|---|
| Market-data **lines** (concurrent streams) | **~100** | **Cap watchlist N ≤ 100** and **stream** (`keepUpToDate`) rather than poll. |
| **Historical** request pacing | **~60 / 10 min**; no identical req < 15 s; ≤6 identical / 2 s | Don't drive the loop with `reqHistoricalData`; use streaming subs. Tail beyond 100 names (if any) rotates through historical within this budget. |
| Market-data type | must set `reqMarketDataType(3)` for delayed | already done in `IBKRDataDownloader`. |

**Operational notes:** Gateway must be up (Docker on Pi); **daily re-auth/restart** →
p19 must handle reconnects; use a **unique `clientId`** (p19 = 19, scanner = 20)
distinct from other bots. Connect via **`127.0.0.1:4002`** (the gateway API), verified
2026-06-28. **15-min delay = discovery, not entry** — for one-session pump-and-fades
you'll often see the move mid/late; treat alerts as awareness.

**Gateway deployment gotcha (resolved 2026-06-28):** the gnzsnz `ib-gateway` container
binds the API to localhost *inside* the container; publishing `4002:4002` over the
Docker bridge makes connections arrive as a non-localhost (untrusted) source, so the
TCP socket connects but the **API handshake silently times out** — for *any* client
(both `ib_insync` and `ib_async` fail identically). Fix: run the paper container with
**`network_mode: host`** (drop the `ports:` block) so the Pi reaches the API as true
localhost. Use the project venv (`ib_async` lives there, not system Python).

**Probe:** `tools/latency_probe.py` has an `--ibkr` mode that connects to the paper
Gateway, sets delayed data, pulls a few penny names' 5m bars, and reports **bar
staleness + volume presence**. Run it **on the Pi during market hours** to confirm the
real delay and that volume comes through before building Phase 1.

## 14. Alerting

- **Dedup**: one alert per name/day; re-alert only on **escalation** to a higher
  severity tier. State in `alerted.json`.
- **Daily cap**: hard maximum alerts/day; drop or batch beyond it.
- **Message** (Telegram, human-readable): ticker, price, % from open, RVOL-so-far,
  fresh-catalyst flag, **dilution/fade warning**, short-squeeze note, sentiment
  snapshot, and *why it fired*.
- **Delivery**: `NotificationService` (Telegram primary, email optional).

## 15. Backtesting & optimization (forward-test model)

Because granular historical intraday penny data is not freely available, follow the
brainstorm's **forward-test** model rather than buying history:

1. **Accumulate** via shadow mode (§12) for ~3–6 months.
2. **Backtest** entry/exit rules on the accumulated data, with realism rules from
   brainstorm §3B: **no-fill realism** (enter at open or +1–2% over trigger),
   **dilution filter** (drop trades with ATM/warrant/424B within 48h), and **risk
   management** (trailing 5–10% stop; sell ½ at +20% with a hard stop) — reuse the
   P17 `strategy_sim.py` engine and its **Optuna** optimizer.
3. **Optimize** thresholds: `intraday_rvol_trigger` (3× vs 5×), float caps
   (< 10M vs 10–25M), severity weights, and (later) any sentiment lead threshold.

## 16. Phased implementation plan

| Phase | Scope | Size |
|---|---|---|
| **0** | This spec + submodule scaffold (`README`, `docs/{Requirements,Design,Tasks}.md`, `tests/`, config, models) | small |
| **1** | Watchlist Builder (P17 + gappers) + `run-once` loop + **Shadow Logger** (no alerts) + EOD backfill. Proves the loop, state file, and starts data accumulation. | medium |
| **2** | Intraday Feed via Finnhub/Polygon + RVOL-so-far/volume profile + Trigger Engine + dedup state + **Telegram alerts** + daily cap. | medium |
| **3** | Enrichment: `CatalystAgent` (fresh 8-K, intraday EFTS), `ShortSqueezeAgent`, `DilutionAgent` fade-flagging; sentiment context attach. | medium |
| **4** | Calibrate thresholds on shadow data via Optuna (§15); optional LULD halt detection; optional LLM alert summarizer. | medium |

**Ship order honours decision #3: shadow logging (Phase 1) before alerting (Phase 2).**

## 17. Risks & open questions

**Risks**
- **Data latency / quota** — minute bars for hundreds of small caps can exhaust free
  tiers (§13). Mitigate via capped watchlist + batch snapshots + caching.
- **Chasing fades** — a +200% spike is often a sell by detection; mitigate by
  surfacing dilution/squeeze context and treating "pump into active ATM/shelf" as a
  fade, not a long.
- **Alert spam** — dedup state, severity gating, daily cap.
- **Operational surface** — a market-hours intraday job is new (monitoring, failure
  handling) vs the once-daily batch.

**Open questions (for Phase 0/1)**
1. Exact free-tier limits & best **batch/snapshot** endpoints for Finnhub vs Polygon
   on small-cap tickers — measure real latency before committing the poll interval.
2. **Gappers/most-active** source — provider screener API, or derive from a
   pre-market DataManager scan of the P17 universe?
3. Volume-profile bootstrap — how many days of cached intraday history before the
   real profile replaces the `daily_avg × intraday_cdf` approximation?
4. Shadow store backend — SQLite (simple) vs Parquet (analytics-friendly) vs the
   planned DuckDB layer (see deferred P15 GDELT work).

---

## 18. Submodule deliverables (per repo conventions)

Phase 0 must also create, under `src/ml/pipeline/p19_penny_intraday/`:
`README.md`, `docs/Requirements.md`, `docs/Design.md`, `docs/Tasks.md`, and
`tests/`. Cross-module dependencies (P17 agents, DataManager, sentiments, EDGAR,
notification) documented in `Requirements.md`.
