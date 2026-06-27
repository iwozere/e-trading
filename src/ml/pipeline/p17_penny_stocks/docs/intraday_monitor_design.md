# Intraday Explosive-Move Monitor — Design Doc

Status: **proposal** (no code yet)
Proposed home: a **new, separate pipeline** (e.g. `src/ml/pipeline/p19_intraday_monitor/`)
Seeds from / reuses: P17 penny-stock screener, `CatalystAgent`, `ShortSqueezeAgent`,
`DilutionAgent`, `DataManager`, `EdgarDownloader`, the notification service.

---

## 1. Why this exists

P17 is a **daily end-of-day batch ranker**. As the SCAG case showed, that design
**structurally cannot** catch a same-session pump-and-fade:

> SCAG (2026-06-24): ripped 0.367 → **1.11 intraday (+200%)** on ~1,100× normal
> volume, then collapsed to close 0.711 — all in one session. P17's daily run
> "saw" it the next morning (and was a Tier-B non-alert anyway). By then the move
> was over.

Catching that requires detecting the volume/price break **while it is happening**,
intraday — a fundamentally different cadence and data model from a daily batch
screen. This monitor fills that gap.

## 2. Goals / non-goals

**Goals**
- Detect explosive intraday moves on a **watchlist** in near-real-time (minutes, not
  next-morning).
- Fire a **single, de-duplicated** alert per name per day at the moment of breakout.
- Combine an intraday volume/price tripwire with **fresh-8-K** catalyst awareness.
- **Reuse** P17's catalyst / short-squeeze / dilution logic as enrichment.

**Non-goals**
- Not a universe-wide ranker (that stays P17's job).
- Not an execution/trading system — alerting only.
- Not tick-level HFT — minute-bar / few-minute polling is sufficient for the
  penny-stock pump regime.
- Social sentiment is **not** a trigger (additive context only, per P17 spec §8.7).

## 3. Why separate pipeline, not a P17 mode

| Axis | P17 (daily) | Intraday monitor |
|---|---|---|
| Data | daily bars, EOD | 1m/5m bars, live quotes, "RVOL-so-far-today" |
| Core logic | rank ~4,000-name universe once | watch a small list for real-time tripwires |
| Style | stateless batch | **event-driven, stateful** (already-alerted today) |
| Cadence | 1×/day cron | every few minutes during market hours |
| Universe | exhaustive | pre-selected watchlist |

These are opposite shapes. Forcing a fast, stateful event loop into the slow,
stateless 7-stage batch orchestrator would complicate both. They share **components**
(data access, catalyst/squeeze/dilution agents, the `Candidate` model), so the clean
split is: **P17 produces the daily watchlist; the intraday monitor watches it live**,
importing P17's agents as libraries.

## 4. Architecture

```
                ┌─────────────────────────────────────────────┐
   (daily)      │ Watchlist Builder                           │
 P17 output  ──►│  P17 Tier B/C + explosive candidates        │──► watchlist.json
 gappers/most-  │  + daily "gappers / most-active < $10" list  │     (per trading day)
 active feed ──►│  + dedup, cap to N names                     │
                └─────────────────────────────────────────────┘
                                     │
        (intraday, every N minutes during market hours)
                                     ▼
   ┌──────────────┐   ┌───────────────────┐   ┌──────────────────────┐
   │ Intraday Feed│──►│ Trigger Engine     │──►│ Enrichment           │
   │ 1m/5m bars   │   │  • RVOL-so-far     │   │  catalyst (fresh 8-K)│
   │ live RVOL    │   │  • % move from open│   │  short-squeeze       │
   │ (DataManager)│   │  • halt / resume   │   │  dilution penalty    │
   └──────────────┘   │  • fresh 8-K today │   └──────────────────────┘
                      └───────────────────┘              │
                                     │                    ▼
                                     ▼          ┌──────────────────────┐
                            ┌────────────────┐  │ Alert Manager        │
                            │ State Store    │◄─┤  dedup per name/day  │
                            │ alerted_today  │  │  Telegram (+email)   │
                            └────────────────┘  └──────────────────────┘
```

### 4.1 Watchlist Builder (runs once pre-market)
- **Sources**: (a) P17's latest dated output (`results/p17_penny_stocks/{date}/` —
  Tier B/C + explosive), (b) a daily **gappers / most-active** list filtered to the
  penny range (price < $10, min liquidity), (c) optional manual pins.
- **Output**: `results/p19_intraday_monitor/{date}/watchlist.json` — capped to N
  (e.g. 150–300) names with per-name baseline context (avg 30d vol, float, prior
  close, dilution penalty, short interest, any known catalyst).
- Capping matters: you cannot poll thousands of intraday tapes every few minutes.

### 4.2 Intraday Feed
- `DataManager.get_ohlcv(symbol, "1m"|"5m", ...)` for watchlist names (already
  supported timeframes).
- Compute **RVOL-so-far-today** = cumulative volume to now ÷ *typical cumulative
  volume by this time of day* (needs an intraday volume-profile baseline, built from
  recent intraday history — see Open Questions).
- Compute **% move from today's open** and **% from prior close**.

### 4.3 Trigger Engine (event detection)
A name fires when it crosses configured tripwires (all tunable in config):

- **Volume surge**: RVOL-so-far ≥ `intraday_rvol_trigger` (e.g. 5×) **and**
  cumulative $-volume ≥ floor (liquidity gate).
- **Price thrust**: |% move from open| ≥ `intraday_move_trigger` (e.g. +20%).
- **Fresh catalyst**: a bullish 8-K filed **today** (from the daily 8-K cache /
  catalyst plan) — escalates severity and lowers the volume/price thresholds.
- **Halt / resume**: LULD halt or resumption detected (provider-dependent; optional
  Phase 2).
- Combine into an **intraday severity score** reusing P17 sub-scores where
  applicable; require volume **and** (price thrust **or** fresh catalyst) to avoid
  pure-illiquidity noise.

### 4.4 Enrichment (reused P17 agents)
- `CatalystAgent` (with the 8-K improvements) → is there fresh news driving this?
- `ShortSqueezeAgent` → squeeze fuel (SI/float, days-to-cover).
- `DilutionAgent` → suppress/annotate names with active ATM/shelf (a "pump into a
  dilution wall" is a *fade* setup, not a long).

### 4.5 Alert Manager + State Store
- **State**: `results/p19_intraday_monitor/{date}/alerted.json` — names already
  alerted today and at what trigger level, so we alert **once** (or only on
  *escalation*, e.g. crossing a higher tier).
- **Delivery**: real-time **Telegram** (primary), optional email; concise message
  (ticker, price, % move, RVOL, fresh-catalyst flag, dilution warning, link).
- Hard daily cap on alert count to prevent storms on chaotic days.

## 5. Cadence & scheduling

- Runs **only during US market hours** (plus a short pre-market window for gappers),
  every `poll_interval_minutes` (e.g. 3–5 min).
- The existing APScheduler/`job_schedules` cron is **daily**; two options:
  1. **Intraday cron** entries (e.g. every 5 min, weekdays, market hours) invoking a
     stateless `run_once()` that loads state, polls, alerts, persists state. Fits the
     existing scheduler model; simplest operationally.
  2. **Long-running intraday loop** process (market-hours daemon). More efficient for
     streaming but adds a new process lifecycle to manage.
  **Recommendation: start with option 1** (stateless `run_once()` on a short intraday
  cron) — reuses the scheduler, easy to reason about, naturally crash-safe via the
  state file.

## 6. Data model (sketch)

```python
@dataclass
class IntradaySignal:
    ticker: str
    ts: datetime               # detection time (UTC)
    price: float
    pct_from_open: float
    pct_from_prev_close: float
    rvol_so_far: float
    dollar_volume_so_far: float
    fresh_catalyst: bool       # bullish 8-K filed today
    catalyst_signals: list[str]
    short_squeeze_score: float
    dilution_penalty: float
    severity: float            # composite intraday score
    trigger_reason: str        # which tripwire(s) fired
```

## 7. Phased implementation plan

### Phase 0 — Decisions (this doc + Open Questions) — *blocking*
Pick: intraday data provider, poll interval, scheduling model (cron vs daemon),
watchlist size N, gappers source.

### Phase 1 — Watchlist + scaffold — *small*
- New pipeline dir `p19_intraday_monitor/` with config, models, `run_once()`.
- Watchlist Builder consuming P17 daily output (+ stub gappers source).
- Wire a market-hours intraday cron (no triggers yet) — proves the loop + state file.

### Phase 2 — Volume/price triggers + alerting — *medium*
- Intraday feed + RVOL-so-far (needs intraday volume-profile baseline).
- Trigger engine (volume surge + price thrust) with dedup state.
- Telegram alerts; daily cap.

### Phase 3 — Catalyst + squeeze/dilution enrichment — *medium*
- Consume the **daily 8-K cache** (from `catalyst_8k_improvement_plan.md`) for
  "fresh 8-K today" escalation.
- Reuse `ShortSqueezeAgent` / `DilutionAgent` for fuel/suppression.

### Phase 4 — Halt detection + tuning — *medium/optional*
- LULD halt/resume signals (provider-dependent).
- Backtest thresholds against recent explosive names (incl. SCAG 2026-06-24) to
  calibrate `intraday_rvol_trigger` / `intraday_move_trigger`.

## 8. Open questions (need a decision)

1. **Intraday data source & rate limits** — does the current `DataManager` 1m/5m
   path provide *near-real-time* bars for arbitrary small-cap tickers, and at what
   latency / quota? May need a dedicated intraday/quote provider.
2. **Intraday volume-profile baseline** — RVOL-so-far needs "typical cumulative
   volume by minute-of-day." Build from cached intraday history, or approximate
   from daily avg volume × an intraday distribution curve?
3. **Scheduling model** — short intraday cron `run_once()` (recommended) vs a
   market-hours daemon?
4. **Gappers / most-active feed** — which source for the non-P17 portion of the
   watchlist (provider screener API, or derive from a pre-market scan)?
5. **Alert philosophy** — alert once per name/day, or re-alert on escalation tiers?
6. **Scope of universe** — strictly P17 penny names, or any small-cap < $10 that
   gaps, regardless of P17 eligibility?

## 9. Risks

- **Data latency/quota**: minute bars for hundreds of small caps every few minutes
  can hit provider limits — mitigated by a capped watchlist and batch fetch.
- **Alert spam**: chaotic days → mitigated by dedup state, severity gating, and a
  daily cap.
- **Chasing fades**: a +200% spike is often a *sell* by the time it's detected —
  mitigate by surfacing dilution/short-squeeze context so the alert says *why*, and
  by treating "pump into active ATM/shelf" as a fade, not a long.
- **Operational**: a market-hours intraday job is new operational surface (monitoring,
  failure handling) vs the once-daily batch.
