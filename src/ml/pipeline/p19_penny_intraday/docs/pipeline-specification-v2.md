# P19 — Intraday Penny-Stock Spike Monitor — Pipeline Specification **v2**

Status: **Phase 1 shipped (shadow mode) / v2 rework pre-implementation**
Home: `src/ml/pipeline/p19_penny_intraday/`
Supersedes: v1 spec (2026-06-28). Seeds from `brainstorming1.md` and the P17
`intraday_monitor_design.md` proposal.

This pipeline detects **explosive intraday moves in penny stocks while they are
happening** and emits a **single, de-duplicated, human-readable alert** per name per
day — now with a **disposition** attached (is this a squeeze setup or a distribution
event?), not just a severity number.

> Motivating case — **SCAG (2026-06-24)**: ripped 0.367 → **1.11 intraday (+200%)** on
> ~1,100× volume, then faded to close 0.711. v1 correctly framed the detection
> problem. v2 addresses the harder half: **most such moves are structurally
> pre-determined to fade**, and that is knowable *before the market opens* from
> filings — at zero intraday rate-budget cost.

---

## 0. What changed in v2, and why

The v1 design is sound as a **detection** system. Its weakness is that it treats
structure as a single scalar (`dilution_penalty`) subtracted from a momentum score.
That conflates two orthogonal axes and destroys the information that matters most.

| # | Change | Rationale |
|---|---|---|
| A | **New Layer 0: Structural Integrity** (§4.0) computed pre-market from EDGAR XBRL + Form 4 + corporate actions | Structural signals move on filing events, not minutes. Computing them pre-market costs **zero intraday rate budget** — the binding constraint in §13. |
| B | **Severity score → 2-axis Disposition Matrix** (§8) | A momentum score minus a dilution penalty says a toxic +200% pump and a clean +60% squeeze are "the same". They are opposite trades. Momentum and structure must stay separate axes. |
| C | **Hard-filter set expanded** (§7): reverse-split history, share-count CAGR, floating-rate converts, going-concern, exchange deficiency notice | These are cheap, deterministic, high-precision disqualifiers absent from v1. Reverse-split history in particular is one of the strongest negative predictors available and was not modelled at all. |
| D | **Positive-signal catalog added** (§7.3) — insider open-market buys (Form 4 code P), share-count stability, institutional accumulation, cash-flow positivity | v1 had *no* positive structural signals. Clustered insider buying is the single most evidenced predictor in the microcap literature and is free from EDGAR. |
| E | **`dilution_urgency` derived metric** (§7.4) | "Has a shelf" is weak. "Has a shelf **and** 2 quarters of cash **and** used the ATM last quarter" is close to deterministic: they *will* sell into the pump. Runway is the multiplier that makes shelf data predictive. |
| F | **Shadow schema v2** (§12) — structural features snapshotted per day, **outcome labels** added to EOD backfill | ⚠️ **Time-critical.** See §0.1. |
| G | **New Phase 1.5** (§16), inserted before the calibration resume point | Consequence of F. |

### 0.1 ⚠️ Do this before more shadow data accumulates

Per v1 §19, the shadow loop is **live and collecting now**, with a resume point in
3–4 weeks. The current schema logs momentum + sentiment only.

**Every day collected without structural features and outcome labels is a day of data
that can calibrate momentum thresholds but *cannot* calibrate the fade/long
discrimination that determines whether an alert is worth acting on.** Structural
features cannot be reliably reconstructed retroactively at daily granularity (shelf
capacity, ATM usage, insider-buy recency and runway all drift, and EDGAR gives you
"as of filing", not "as of that Tuesday").

Phase 1.5 (§16) is ~2 days of work and unblocks the entire v2 thesis. It should
happen **before** the 3–4 week accumulation window elapses, not after.

---

## 1. Locked design decisions

Decisions 1–4 from v1 stand unchanged. New:

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Intraday data feed | **IBKR Gateway (delayed, free)** primary; Finnhub real-time price cross-check; yfinance fallback | *(v1, unchanged)* No free REST tier gives real-time intraday volume. IBKR delayed bars include volume. See §13.2. |
| 2 | Watchlist universe | **P17 daily output + pre-market gappers/most-active < $5** | *(v1, unchanged)* |
| 3 | Build sequencing | **Shadow-mode logger first**, then calibrate, then alert | *(v1, unchanged)* |
| 4 | Social/news sentiment | **Context/enrichment only** (not a trigger) | *(v1, unchanged)* |
| **5** | **Structural signals computed pre-market, never intraday** | Layer 0 refreshes daily (Form 4, new filings) + weekly (XBRL facts); cached per ticker | Filing-driven data has filing-driven cadence. Keeps the intraday loop inside the ~100 IBKR market-data line budget. |
| **6** | **Momentum and structure stay orthogonal** | Disposition matrix (§8), not a single blended severity | A fade signal and a long signal must not sum. Blending them makes the highest-momentum toxic names score like the best setups. |
| **7** | **Structural toxicity suppresses alerts but never suppresses shadow logging** | Grade-D names stay on the watchlist and in the shadow store | They are the **negative training set**. Excluding them from logging destroys the ability to prove the classifier works. |
| **8** | **EDGAR is the structural source of record** | XBRL `companyfacts` + `submissions` + EFTS full-text + Form 4 XML | Free, 10 req/s, no key. One `companyfacts` call per ticker per week yields share count, cash, and burn as time series. |

---

## 2. Goals / non-goals

**Goals** *(v1 goals retained, plus)*
- Detect explosive intraday moves on a capped **watchlist** in near-real-time.
- Fire **one** de-duplicated alert per name per day at breakout (or on escalation).
- **Classify each detected move by disposition** — squeeze candidate / watch / fade /
  suppress — rather than ranking it on a single severity scale.
- Maintain a **structural integrity profile** per watchlist name, refreshed pre-market
  from filings at zero intraday cost.
- Accumulate a shadow dataset **with structural features and forward outcome labels**
  sufficient to calibrate *both* axes.

**Non-goals** *(v1, unchanged, plus)*
- Not a universe-wide ranker (P17's job).
- Not an execution/trading system — **alerting only**. A `FADE` disposition is
  information, not a short recommendation.
- Not tick-level HFT.
- Social sentiment is **not** a trigger.
- **Not a fundamental valuation model.** Layer 0 measures *cap-structure integrity*
  and *dilution urgency*, not business quality. A structurally clean company can still
  be a bad business; that is out of scope.

---

## 3. Why a separate pipeline (not a P17 mode)

*(v1 §3 unchanged — P17 produces the daily watchlist; P19 watches it live.)*

| Axis | P17 (daily) | P19 (intraday) |
|---|---|---|
| Data | daily EOD bars | 1m/5m bars, live quotes, RVOL-so-far |
| Core logic | rank ~4,000-name universe once | watch a small list for live tripwires |
| Style | stateless batch | **event-driven, stateful** |
| Cadence | 1×/day cron | every few minutes during market hours |
| Universe | exhaustive | pre-selected, capped watchlist |

**v2 note:** Layer 0 (§4.0) is a *P19* component, not a P17 one, because it is scoped
to the ≤100-name watchlist. Running full structural profiling over P17's 4,000-name
universe is not rate-feasible and not needed. If it later proves cheap enough, promote
it to P17 as a pre-rank filter.

---

## 4. Architecture (v2)

```
   (daily, pre-market — no intraday rate cost)
  ┌───────────────────────────────────────────────┐
  │ Watchlist Builder                             │
  │  P17 Tier B/C + explosive + gappers < $5      │
  │  + dedup, momentum-eligibility filters, cap N │
  └───────────────────┬───────────────────────────┘
                      ▼
  ┌───────────────────────────────────────────────┐
  │ ★ LAYER 0 — Structural Integrity Profiler     │
  │   EDGAR XBRL companyfacts  → share count Δ,   │
  │       cash, burn, runway                      │
  │   EDGAR submissions        → S-3/424B5/S-1,   │
  │       8-K 3.01/3.02/5.03, 13G/13D             │
  │   EFTS full-text           → floating converts│
  │       going-concern language                  │
  │   Form 4 XML               → insider P vs S   │
  │   yfinance .splits         → reverse splits   │
  │                                               │
  │   → structural_grade  A / B / C / D           │
  │   → dilution_urgency  0–100                   │
  │   → insider_conviction 0–100                  │
  │   D-grade → alert-suppressed (still logged)   │
  └───────────────────┬───────────────────────────┘
                      ▼   watchlist.json (+ structural profile per name)
   (intraday, every poll_interval_minutes, market hours)
                      ▼
  ┌──────────────┐  ┌────────────────────┐  ┌──────────────────────┐
  │ Intraday Feed│─►│ Momentum Trigger   │─►│ Enrichment (P17)     │
  │ IBKR delayed │  │ • RVOL-so-far      │  │ catalyst (fresh 8-K) │
  │ Finnhub px   │  │ • % from open/prev │  │ short-squeeze        │
  │ 1m/5m + quote│  │ • momentum_tier    │  │ (dilution → Layer 0) │
  └──────┬───────┘  └─────────┬──────────┘  └──────────┬───────────┘
         │                    │                        │
         ▼                    ▼                        ▼
  ┌──────────────┐   ┌────────────────────────────────────────┐
  │ Shadow Logger│   │ ★ DISPOSITION ENGINE (§8)              │
  │ momentum +   │   │   momentum_tier × structural_grade     │
  │ structural + │   │   → LONG_CANDIDATE / WATCH /           │
  │ sentiment    │   │     FADE_CANDIDATE / SUPPRESS          │
  │ (ALL names)  │   └──────────────┬─────────────────────────┘
  └──────┬───────┘                  ▼
         │              ┌────────────────┐   ┌──────────────────────┐
         │              │ State Store    │◄──┤ Alert Manager        │
         ▼              │ alerted_today  │   │ dedup per name/day   │
  ┌──────────────┐      └────────────────┘   │ per-disposition cap  │
  │ EOD Backfill │                           │ Telegram (+email)    │
  │ ★ + outcome  │                           └──────────────────────┘
  │   labels §12 │
  └──────────────┘
```

### 4.0 ★ Structural Integrity Profiler (NEW — runs pre-market)

**Cadence.** Full refresh **weekly** per ticker (XBRL facts change quarterly).
**Daily delta check** via the EDGAR daily index: if any new filing landed for a
watchlist CIK, re-profile that name only. New names entering the watchlist are
profiled on first sight and cached.

**Cost.** One `companyfacts` call + one `submissions` call + one EFTS query per
ticker per refresh. At N=100 and a weekly cadence this is ~300 requests/week against
a 10 req/s limit — negligible, and entirely outside market hours.

**Failure mode.** Many microcaps are foreign private issuers (20-F/6-K, e.g. GRSD) or
have incomplete XBRL. **Missing data must grade as `C` (unknown ≠ clean), never `A`.**
An unprofilable name is not a safe name.

**Outputs per ticker:**

```python
@dataclass
class StructuralProfile:
    ticker: str
    cik: str | None
    as_of: date
    grade: str                      # A / B / C / D
    dilution_urgency: float         # 0–100
    insider_conviction: float       # 0–100
    # component evidence (all nullable → coverage tracked explicitly)
    reverse_splits_24m: int
    share_count_cagr_8q: float | None
    shares_outstanding: float | None
    cash: float | None
    quarterly_burn: float | None
    runway_quarters: float | None
    shelf_active: bool | None
    shelf_capacity_pct_mcap: float | None
    days_since_last_offering: int | None
    floating_convert_flag: bool | None
    going_concern_flag: bool | None
    exchange_deficiency_flag: bool | None
    warrant_overhang_pct_float: float | None
    insider_buys_90d: int           # Form 4 code P, open market
    distinct_insider_buyers_90d: int
    insider_sells_90d: int
    inst_holders_delta_2q: int | None
    recent_ipo_months: int | None
    coverage: float                 # fraction of fields resolved
    disqualifiers: list[str]        # human-readable, surfaced in alert
```

### 4.1 Watchlist Builder (pre-market)
- **Sources**: (a) latest P17 dated output (Tier B/C + explosive), (b) pre-market
  gappers / most-active filtered to penny range, (c) manual pins.
- **Momentum-eligibility filters** (§7.1) applied here; **cap to N ≤ 100** (IBKR line
  budget, §13.2).
- **Then** Layer 0 profiles every surviving name.
- **Output**: `results/p19_penny_intraday/{date}/watchlist.json` with baseline context
  (avg 30d volume, float, prior close, short interest, volume-profile baseline) **plus
  the full `StructuralProfile`**.

### 4.2 Intraday Feed
*(v1 §4.2 unchanged.)* IBKR Gateway delayed streaming 5m bars, `reqMarketDataType(3)`,
one subscription per name within the ~100-line budget; Finnhub `/quote` optional
faster price cross-check; yfinance/Polygon fallback via `DataManager`.
RVOL-so-far = cumulative volume ÷ typical cumulative volume by minute-of-day (U-shaped
profile from accumulated shadow data; linear approximation until then — see §19).

### 4.3 Momentum Trigger Engine (stateful)
Unchanged tripwires, but the output is now a **`momentum_tier`**, not a final severity:
- **Volume surge**: `rvol_so_far ≥ intraday_rvol_trigger` **and** cumulative
  `$-volume ≥ dollar_volume_floor`.
- **Price thrust**: `|pct_from_open| ≥ intraday_move_trigger`.
- **Fresh catalyst**: bullish 8-K filed today → escalates tier, lowers thresholds.
- **Gate**: volume **AND** (price thrust **OR** fresh catalyst).
- → `momentum_tier ∈ {T0 none, T1 elevated, T2 strong, T3 explosive}` (§8.1).

### 4.4 Enrichment (reused P17 agents)
- `CatalystAgent` → fresh bullish 8-K driving the move.
- `ShortSqueezeAgent` → squeeze fuel (SI/float, days-to-cover).
- `TechnicalAgent` → sub-scores where applicable.
- ~~`DilutionAgent`~~ → **superseded by Layer 0.** Keep the agent as a component but
  call it *from* the profiler, pre-market, and extend it per §7.2. Do not run dilution
  logic in the intraday loop.

### 4.5 Sentiment (context only)
*(v1 unchanged.)* Attach, do not trigger. Logged per poll for later lead-time testing.

### 4.6 Alert Manager + State Store
- **State**: `alerted.json` — name → disposition + tier alerted today.
- **Dedup**: one alert per name/day; re-alert only on **escalation** (higher momentum
  tier **or** disposition change, e.g. `WATCH → FADE_CANDIDATE` after an intraday
  424B5 lands).
- **Per-disposition daily caps**, not one global cap — a chaotic day generates mostly
  `FADE_CANDIDATE`s, and those must not crowd out the rare `LONG_CANDIDATE`.
- **Delivery**: `NotificationService` (Telegram primary, email optional).

### 4.7 Shadow Logger
Every poll, for **every** watchlist name including grade-D suppressed ones (decision
#7), append a row per §12 schema v2.

---

## 5. Pipeline run modes

| Mode | Trigger | Behaviour |
|---|---|---|
| `build-watchlist` | once, pre-market cron | produce `watchlist.json` |
| ★ `profile-structural` | pre-market, after build-watchlist | Layer 0 refresh (weekly full / daily delta) → structural profile cache |
| `run-once` (shadow) | intraday cron (Phase 1) | poll, **log only**, no alerts |
| `run-once` (live) | intraday cron (Phase 2+) | poll, trigger, classify, enrich, alert, persist state |
| `eod-backfill` | post-close cron | fill EOD O/H/L/C **+ outcome labels (§12.2)** |
| ★ `label-backfill` | T+10 cron | fill forward-return and dilution-event labels |

`run-once` stays **stateless across invocations**.

---

## 6. Reuse map (do not rebuild)

*(v1 §6 table retained in full.)* Additions for v2:

| Capability | Reuse / New | Path |
|---|---|---|
| EDGAR submissions + 8-K + EFTS | `EdgalDownloader` (`download_8k_filings`, `_efts_search`, `get_recent_filings`) | `src/data/downloader/edgar_downloader.py` |
| ★ XBRL `companyfacts` client | **NEW** — thin client, ETag-cached | `p19_penny_intraday/structural/xbrl_client.py` |
| ★ Form 4 parser | **NEW** — ownership XML → transactions | `p19_penny_intraday/structural/form4.py` |
| ★ Corporate actions (splits) | `yfinance .splits` via `DataManager` | existing |
| Dilution logic core | extend `DilutionAgent` | `p17_penny_stocks/agents/dilution_agent.py` |

**New code (v2):** Structural Integrity Profiler + its sub-clients, Disposition
Engine, shadow schema v2 migration, label backfill job. *(Plus v1's remaining new
code: Trigger Engine, State Store, Alert Manager.)*

---

## 7. Filters and signals

v1 had one flat filter list. v2 splits it into four: momentum eligibility (does this
name *move*?), structural disqualifiers (is the cap structure defensible?), positive
signals (is there evidence of conviction?), and the derived urgency metric.

### 7.1 Momentum eligibility — watchlist admission

*(v1 §7, unchanged.)*
- **Price** < `$5` (configurable).
- **Float** < `~25M` shares; `< 10M` flagged "ultra-low".
- **Min liquidity**: daily volume > `~500k` shares.
- US exchanges; exclude ETFs / test issues (P17 universe hygiene).

### 7.2 ★ Structural disqualifiers — negative signals

Two severities. **Grade D = alert-suppressed** (logged only). **Grade C = eligible,
but classified `FADE_CANDIDATE` on any momentum trigger.**

| # | Signal | Grade | Detection | Source |
|---|---|---|---|---|
| N1 | **Reverse split within 24 months** | **D** | `yf.Ticker(t).splits` ratio < 1.0; corroborate 8-K item 5.03 | yfinance / EDGAR |
| N2 | Any reverse split in listed history | C | as N1, no window | yfinance |
| N3 | **Share-count CAGR > 25% over 8 quarters** | **D** | XBRL `dei:EntityCommonStockSharesOutstanding` time series | companyfacts |
| N4 | Share-count CAGR 10–25% | C | as N3 | companyfacts |
| N5 | **Floating-rate / toxic convertible outstanding** | **D** | EFTS phrase match scoped to CIK: *lowest VWAP*, *conversion price equal to*, *discount to the market price*, *variable conversion* | EFTS full-text |
| N6 | **Going-concern qualification in latest annual** | **D** | EFTS: *substantial doubt about its ability to continue as a going concern* | EFTS |
| N7 | **Exchange deficiency notice active** (bid price / MVLS / equity) | **D** | 8-K **item 3.01** in last 12m without a subsequent compliance-regained 8-K | submissions |
| N8 | 424B5 / S-1 / prospectus supplement filed within 5 trading days | C (→ escalate to D if within 2 days) | submissions form-type + date | submissions |
| N9 | Effective S-3 shelf with remaining capacity > 30% of market cap | C | S-3 + prospectus text parse; fall back to *shelf exists* flag | submissions + text |
| N10 | ATM used in either of last 2 quarters | C | XBRL `ProceedsFromIssuanceOfCommonStock` > 0 with no discrete offering 8-K | companyfacts |
| N11 | **Cash runway < 3 quarters** | C (→ D if < 1.5 and shelf active) | `Cash…AtCarryingValue` ÷ \|quarterly `NetCashProvidedByUsedInOperatingActivities`\| | companyfacts |
| N12 | Warrant overhang > 20% of float with strike within 50% of price | C | 10-Q/10-K warrant table (text parse; nullable) | filings |
| N13 | 8-K **item 3.02** (unregistered equity sale) within 30 days | C | submissions | submissions |
| N14 | Clustered insider **sales** / Form 144 in last 90 days | C | Form 4 code S, disposal | Form 4 |
| N15 | IPO within 18 months **and** float < 5M **and** foreign private issuer reporting | C | listing date + 20-F/6-K form types + float | submissions |
| N16 | Auditor absent from whitelist, or PCAOB-sanctioned | C | audit opinion signature block | 10-K/20-F |
| N17 | **Layer 0 coverage < 0.4** (unprofilable) | C | `coverage` field | derived |

**Design note on N17.** v1 had no concept of unknown-vs-clean. Grading unknowns as
neutral-good is exactly how a monitor ends up alerting confidently on the least
transparent names in the universe. Unknown grades to C.

### 7.3 ★ Positive signals — structural conviction

Absent entirely from v1. These lift a name from B toward A and feed
`insider_conviction`. **No positive signal ever overrides a D disqualifier** — that
asymmetry is deliberate: dilution mechanics are near-deterministic, conviction signals
are probabilistic.

| # | Signal | Weight | Detection |
|---|---|---|---|
| P1 | **Clustered insider open-market buying** — ≥3 distinct insiders, code `P`, within 30 days | **highest** | Form 4 XML: `transactionCode == P`, `acquiredDisposedCode == A`, non-derivative |
| P2 | Any officer/director open-market buy in last 90 days, sized meaningfully vs. their comp | high | Form 4 |
| P3 | **Share count flat or declining over 8 quarters** | high | companyfacts — a microcap that hasn't printed stock in 2 years doesn't need to |
| P4 | Buyback authorised **and executed** | high | `PaymentsForRepurchaseOfCommonStock` > 0 |
| P5 | Positive operating cash flow (any magnitude) | medium | `NetCashProvidedByUsedInOperatingActivities` > 0 |
| P6 | Cash runway > 8 quarters, or net cash positive | medium | derived (§7.4) |
| P7 | No dilution event in 24 months | medium | absence of 424B5 / S-1 / 8-K 3.02 |
| P8 | Institutional accumulation — new/increased 13G/13D positions over 2 consecutive quarters | medium | submissions (13F reverse-lookup deferred to Phase 3) |
| P9 | No debt maturity inside 24 months | medium | XBRL maturity schedule (nullable) |
| P10 | Insider ownership > 20% and stable | low | Form 3/4 aggregate |
| P11 | High SI% + rising days-to-cover **conditional on grade A/B** | context | `ShortSqueezeAgent` |

**P11 is deliberately conditional.** Short interest is squeeze fuel on a clean
structure and *distribution fuel* on a toxic one — the same raw number, opposite
meaning. v1 added squeeze score unconditionally into severity, which systematically
over-scored the worst names. This is the single highest-impact scoring bug the rework
fixes.

### 7.4 ★ `dilution_urgency` — the derived metric

Shelf existence alone is weakly predictive; most microcaps have one. What predicts
*sale into strength* is **capacity × need × demonstrated willingness**:

```
dilution_urgency = 100 × clip(
      w_r · runway_pressure          # 1 - clip(runway_quarters / 8, 0, 1)
    + w_c · shelf_capacity_pct_mcap  # clipped at 1.0
    + w_h · recent_usage_rate        # offerings in trailing 8 quarters / 8
    + w_d · dilution_history         # share_count_cagr_8q, clipped
    , 0, 1)
```

Weights are **config, calibrated against the `dilution_event_within_5d` label**
(§12.2) — not hand-tuned. Bootstrap suggestion only:
`w_r=0.35, w_c=0.25, w_h=0.25, w_d=0.15`.

**Interpretation:** `dilution_urgency > 70` on a +100% intraday move means the pump is
a financing window and the company has both the means and the motive to use it.
That is the `FADE_CANDIDATE` core case.

### 7.5 Grade assignment

```
D  ← any D-severity disqualifier fires
C  ← any C-severity disqualifier fires, or coverage < 0.4
B  ← no disqualifiers, insider_conviction < 40
A  ← no disqualifiers, insider_conviction ≥ 40, dilution_urgency < 30
```

---

## 8. ★ Disposition Engine (replaces the flat severity score)

### 8.1 Momentum tier

Normalised 0–100 composite of **momentum evidence only** — RVOL-so-far, % from open,
% from prior close, $-volume, fresh-bullish-8-K boost. **No structural term.**
Thresholds → `T0 / T1 / T2 / T3`, calibrated from shadow data (§15), not hand-set.

### 8.2 Disposition matrix

| | **A** (clean + conviction) | **B** (clean, neutral) | **C** (dilution risk) | **D** (toxic) |
|---|---|---|---|---|
| **T3 explosive** | `LONG_CANDIDATE` 🟢 | `WATCH` 🟡 | `FADE_CANDIDATE` 🔴 | `SUPPRESS` ⚫ |
| **T2 strong** | `LONG_CANDIDATE` 🟢 | `WATCH` 🟡 | `FADE_CANDIDATE` 🔴 | `SUPPRESS` ⚫ |
| **T1 elevated** | `WATCH` 🟡 | `WATCH` 🟡 | log only | `SUPPRESS` ⚫ |
| **T0** | log only | log only | log only | log only |

**Escalation rules:**
- A fresh **bullish** 8-K raises momentum tier by one.
- An intraday **424B5 / S-1 / 8-K item 3.02** immediately forces `FADE_CANDIDATE` and
  re-alerts as an escalation regardless of prior state. This is the highest-value
  intraday structural event and justifies its own EFTS poll (§9).
- Grade D never escalates to an alert. It escalates only in the shadow log.

**Alert content differs by disposition.** A `FADE_CANDIDATE` message leads with *why
it is a fade* — the named disqualifiers, runway, shelf capacity, days since last
offering — not with the price move. The disqualifier list is the payload.

---

## 9. Catalyst — intraday filings

- Phase 1: daily 8-K index for "filed today" awareness.
- Phase 2: EDGAR **EFTS/RSS intraday** poll over watchlist CIKs. **v2 extends the
  watched form set beyond 8-K:** `424B5`, `S-1`, `S-3`, `8-K item 3.02`, `8-K item
  3.01`. A same-session dilution filing is at least as actionable as a same-session
  bullish 8-K, and is the direct trigger for disposition escalation (§8.2).
- Bearish 8-K items (1.02 / 1.03 / 3.01 / 4.02) never count as bullish
  (`CatalystAgent` classification).

---

## 10. Sentiment — context only

*(v1 §10 unchanged.)* Adapters: Reddit, StockTwits, ApeWisdom, Google Trends,
NewsAPI/Finnhub news, FinBERT. Captured per poll, attached to alerts, never a trigger.

**v2 addition — a specific hypothesis worth testing once data exists:** mention-spike
magnitude conditional on `structural_grade`. If promotional mention spikes cluster on
grade C/D names, mention velocity becomes a useful *fade* confirmer even though it is
useless as a directional trigger. Log the interaction term.

---

## 11. Data model (v2)

```python
@dataclass
class IntradaySignal:
    ticker: str
    ts: datetime
    price: float
    # --- momentum axis ---
    pct_from_open: float
    pct_from_prev_close: float
    rvol_so_far: float
    dollar_volume_so_far: float
    momentum_score: float            # 0–100, momentum evidence ONLY
    momentum_tier: str               # T0 / T1 / T2 / T3
    # --- catalyst ---
    fresh_catalyst: bool
    catalyst_signals: list[str]
    fresh_dilution_filing: bool      # ★ 424B5 / S-1 / 8-K 3.02 today
    # --- structural axis (from Layer 0, pre-market) ---
    structural_grade: str            # A / B / C / D
    dilution_urgency: float          # 0–100
    insider_conviction: float        # 0–100
    runway_quarters: float | None
    disqualifiers: list[str]         # ★ human-readable, drives alert copy
    structural_coverage: float
    # --- squeeze (conditional, see P11) ---
    short_squeeze_score: float
    # --- context ---
    sentiment: dict
    # --- output ---
    disposition: str                 # LONG_CANDIDATE / WATCH / FADE_CANDIDATE / SUPPRESS
    trigger_reason: str
```

Note the removal of `severity` and `dilution_penalty`. Nothing in the model collapses
the two axes into one number; the disposition is a **classification**, and the alert
carries both scores separately so a human can override.

---

## 12. Storage / shadow dataset — **schema v2**

Per-day artefacts unchanged: `watchlist.json`, `alerted.json`, `signals.csv`,
`report.md` under `results/p19_penny_intraday/{date}/`.

**Backend decision (resolves v1 open question 4): DuckDB over Parquet files.** The
calibration workload is analytical (distribution fits, label joins, group-bys over
months of per-poll rows), the dataset is append-only and single-writer, and the
planned P15 layer already points there. SQLite is fine for the state store; it is the
wrong engine for the calibration query pattern.

### 12.1 `intraday_shadow` — per-poll rows (all watchlist names)

```
ts, ticker, price,
rvol_so_far, pct_from_open, pct_from_prev_close, dollar_volume,
momentum_score, momentum_tier,
fresh_8k, fresh_dilution_filing,
★ structural_grade, ★ dilution_urgency, ★ insider_conviction,
★ runway_quarters, ★ shelf_capacity_pct_mcap, ★ share_count_cagr_8q,
★ days_since_last_offering, ★ insider_buys_90d, ★ distinct_insider_buyers_90d,
★ reverse_splits_24m, ★ floating_convert_flag, ★ going_concern_flag,
★ structural_coverage, ★ disqualifiers (json),
short_squeeze_score,
mention_count, finbert_score, trends_score,
★ disposition
```

Structural fields are **denormalised into every row** rather than joined from a
profile table. They are the point-in-time snapshot; a later join against a mutable
profile table would leak future information into calibration.

### 12.2 ★ Outcome labels — the missing half

v1 backfilled `open, high, low, close, day_max`. That supports "how big was the move"
but **not** "should you have acted on it". Add, per ticker/day:

| Label | Definition | Answers |
|---|---|---|
| `high_time` | minute-of-day of intraday high | Was the alert before or after the top? |
| `close_retention` | `(close − open) / (high − open)` | **Primary fade measure.** ~1.0 held, ~0 round-tripped |
| `mae_from_alert` | max adverse excursion from alert price, same session | Would a stop have survived? |
| `mfe_from_alert` | max favourable excursion from alert price | Was there tradeable follow-through after detection? |
| `ret_t1, t3, t5, t10` | forward close-to-close returns | Multi-day continuation vs. decay |
| ★ `dilution_event_within_5d` | 424B5 / S-1 / 8-K 3.02 filed within 5 sessions | **Directly validates the fade thesis.** Trains `dilution_urgency` weights |
| ★ `reverse_split_within_180d` | corporate action | Long-horizon structural-decay label |

`mae_from_alert` / `mfe_from_alert` require the alert price, so during shadow mode
compute them against the **would-have-alerted** price from the simulated trigger.
Store the simulated trigger point even when no alert is sent — otherwise Phase 2's
first live alerts have no historical analogue to compare against.

---

## 13. Scheduling, cadence & rate budget

*(v1 §13, §13.1, §13.2 retained in full — feed probe findings, IBKR line limits,
historical pacing, `network_mode: host` gotcha, `clientId` 19, `127.0.0.1:4002`,
15-min delay caveat, `tools/latency_probe.py --ibkr`.)*

**v2 additions:**

| Layer | Window | Budget | Notes |
|---|---|---|---|
| Layer 0 structural | pre-market, 06:00–08:00 ET | EDGAR 10 req/s; ~3 calls/ticker/week | ETag + local cache; weekly full, daily delta on new-filing detection |
| Intraday momentum | market hours | IBKR ~100 lines, streaming | unchanged from v1 |
| Intraday filings poll | market hours, 10-min cadence | EFTS, scoped to ≤100 CIKs | new in v2; small, and the escalation payoff is large |

**The key budget property of v2: the entire structural layer sits outside market
hours.** It adds analytical power without competing for the intraday constraint that
v1 correctly identified as binding.

---

## 14. Alerting

- **Dedup**: one alert per name/day; re-alert on escalation (higher tier **or**
  disposition change).
- **Per-disposition daily caps** (v2) rather than one global cap.
- **Message**, differentiated by disposition:
  - `LONG_CANDIDATE` 🟢 — ticker, price, % from open, RVOL-so-far, catalyst, **grade A
    evidence** (insider buys, runway, share-count stability), squeeze note, sentiment.
  - `WATCH` 🟡 — same, framed as unresolved; note which structural fields are unknown.
  - `FADE_CANDIDATE` 🔴 — **leads with disqualifiers**: named toxic items, runway,
    shelf capacity, days since last offering, `dilution_urgency`. Price move is
    secondary.
- Every alert states **why it fired** and **what would change the disposition**.
- **Delivery**: `NotificationService` (Telegram primary, email optional).

---

## 15. Calibration & forward-test model

*(v1 §15 forward-test approach retained: accumulate 3–6 months, then backtest with
no-fill realism, dilution filter, trailing stops, via `strategy_sim.py` + Optuna.)*

**v2 reframes what calibration must answer.** Four questions, in priority order:

1. **Does `structural_grade` separate `close_retention`?** If grade A/B names retain
   materially more of their intraday move than C/D names, the whole v2 thesis holds
   and the matrix is worth tuning. If not, revert to a single severity and say so.
2. **Does `dilution_urgency` predict `dilution_event_within_5d`?** This is a clean
   supervised problem with a hard label. Fit the §7.4 weights against it directly.
3. **Where should `momentum_tier` thresholds sit?** *(v1's original question —
   RVOL-so-far and |%-from-open| distributions vs `day_max`.)*
4. **Does `insider_conviction` add anything beyond the absence of disqualifiers?**
   Plausibly it does not at intraday horizons. Test it; drop it if flat.

**Guard against the obvious trap:** grade D names will be rare in the surviving
watchlist and their outcomes extreme. Report per-grade sample counts alongside every
result, and treat any grade with n < 30 name-days as non-conclusive.

---

## 16. Phased implementation plan

| Phase | Scope | Size | Status |
|---|---|---|---|
| **0** | Spec + submodule scaffold | small | ✅ done |
| **1** | Watchlist Builder + `run-once` shadow loop + Shadow Logger + EOD backfill | medium | ✅ done |
| **★ 1.5** | **Schema v2 migration + Layer 0 profiler (core subset) + outcome labels.** See below. | **small–medium** | **✅ code done 2026-08-18 (see tasks-v2.md); not yet run against live EDGAR / deployed on the Pi** |
| **2** | Momentum Trigger Engine + Disposition Engine + dedup/escalation state + Telegram alerts + per-disposition caps | medium | after calibration |
| **3** | N5,N6,N15,N16 / P8,P9,P11, intraday EFTS dilution poll, sentiment context attach | medium | **✅ code done 2026-08-18 (see tasks-v2.md); not yet run against live EDGAR / deployed on the Pi. N12, P10, and N9 above $75M float still deferred — see design-v2.md §Roadmap for why.** |
| **4** | Optuna calibration of both axes (§15), LULD halt detection, optional LLM alert summarizer | medium | |

### ★ Phase 1.5 — scope (the time-critical insert)

Deliberately minimal. Ship the **deterministic, high-coverage** structural signals
first; defer everything requiring fragile text parsing to Phase 3.

1. **Migrate `intraday_shadow` to schema v2** (§12.1) — additive columns, backfill
   `NULL` for existing rows.
2. **Layer 0 core subset** — only what is cheap and reliable:
   - reverse splits (yfinance `.splits`) → N1, N2
   - share-count series (XBRL `companyfacts`) → N3, N4, P3
   - cash + operating cash flow → runway → N11, P5, P6
   - offering filings (`submissions` form types) → N8, N13, P7
   - Form 4 code P/S in last 90d → P1, P2, N14
   - 8-K item 3.01 → N7
   - EFTS going-concern + floating-convert phrase match → N5, N6
   - → provisional `structural_grade` + `dilution_urgency` (bootstrap weights)
3. **Denormalise the profile into every shadow row.**
4. **Outcome labels** (§12.2) in `eod-backfill` + new `label-backfill` at T+10.
5. **Store the simulated trigger point** on every poll, even in shadow mode.

Phases 2–4 proceed on the v1 sequencing. **Decision #3 still governs: shadow logging
before alerting, calibration before thresholds.**

---

## 17. Risks & open questions

**Risks** *(v1 retained, plus)*
- **Data latency / quota** — mitigated via capped watchlist, streaming, caching.
- **Chasing fades** — *this is what v2 principally addresses.* Residual risk: 15-min
  delayed discovery means even a correct `LONG_CANDIDATE` is often un-actionable.
  Treat alerts as awareness (v1 §13.2 caveat stands).
- **Alert spam** — dedup, disposition gating, per-disposition caps.
- **★ Structural staleness** — a profile is only as fresh as the last filing. A shelf
  takedown filed at 16:05 is invisible until the next EFTS poll. Mitigate with the
  intraday filings poll (§9); accept residual gaps.
- **★ False confidence from grade A** — grade A means *no detected disqualifier*, not
  *verified clean*. Alert copy must say "no disqualifiers found (coverage 0.82)", never
  "clean". Surfacing `coverage` in the message is a hard requirement, not a nicety.
- **★ Survivorship in calibration** — the watchlist is already momentum-filtered, so
  the shadow dataset cannot answer "does structure predict returns in general", only
  "does it predict outcomes *given* a momentum event". Do not over-claim from it.
- **Operational surface** — market-hours job monitoring, IBKR reconnects, daily
  re-auth.

**Open questions**
1. *(v1 Q1, still open)* Real market-hours latency for Finnhub vs Polygon vs IBKR —
   measure before fixing the poll interval.
2. *(v1 Q2)* Gappers source — provider screener API vs. pre-market DataManager scan.
3. *(v1 Q3)* Volume-profile bootstrap — days of cached intraday history before the
   real U-shaped profile replaces `daily_avg × intraday_cdf`.
4. ~~*(v1 Q4)* Shadow store backend~~ → **resolved: DuckDB** (§12).
5. **★ NEW** — Ticker→CIK mapping coverage for the penny universe, especially foreign
   private issuers and recent uplistings. SEC's `company_tickers.json` is the base;
   measure the miss rate before trusting `coverage`.
6. **★ NEW** — EFTS phrase-match precision for floating converts. Needs a hand-labelled
   sample of ~50 filings to set the phrase set; false positives here downgrade good
   names to D and are costly.
7. **★ NEW** — Shelf *remaining capacity* almost certainly requires text parsing of
   prospectus supplements. Is the boolean *shelf exists + recently used* good enough
   for the `dilution_urgency` fit, or does N9 need the number? Test against
   `dilution_event_within_5d` before investing in the parser.

---

## 18. Submodule deliverables

Under `src/ml/pipeline/p19_penny_intraday/`: `README.md`,
`docs/{Requirements,Design,Tasks}.md`, `tests/`. Cross-module dependencies (P17
agents, DataManager, sentiments, EDGAR, notification) documented in
`Requirements.md`.

**v2 additions:** `structural/` subpackage (`profiler.py`, `xbrl_client.py`,
`form4.py`, `disqualifiers.py`), `disposition.py`, `migrations/` for schema v2,
and `docs/StructuralSignals.md` documenting every N/P signal with its detection
method, source, and measured coverage.

---

## 19. Status & resume point

**Phase 1 built and committed** — monitor runs in shadow mode (logging only).
IBKR paper Gateway working (host networking, `127.0.0.1:4002`, `ib_async`).
Watchlist builder producing capped, enriched `{date}/watchlist.json`.
Shadow loop writing `shadow.sqlite`. `eod-backfill` + scheduler SQL in place.
`shadow_report.py` QA tool; 31 tests passing.

**Phase 1.5 and Phase 3 code-complete (2026-08-18)** — see tasks-v2.md for the
full per-component breakdown. Neither has been applied on the Pi or run
against live EDGAR yet; both are still pending the operational carry-overs
listed there. Phase 2 (Disposition Engine + alerting) deliberately still
waits — see decision #3 / §16: shadow logging and calibration come first.

**Operational carry-overs from v1 (still required):**
1. Verify a market-hours `run-once --mode shadow` logs rows; check `shadow_report` —
   fix `ibkr_volume_lot_size` if the vol-ratio flag fires.
2. `psql -d <db> < bin/scheduler/insert_p19_schedules.sql` to start daily collection.
3. Use a dedicated paper IBKR username to avoid re-login churn dropping polls.

**★ Revised resume order (changed from v1):**

1. ~~Phase 1.5~~ and ~~Phase 3~~ code done (this round) — **apply
   `bin/scheduler/insert_p19_v2_schedules.sql` on the Pi** (adds Structural
   Profile, Label Backfill, and the new Intraday Filings Poll jobs) and run
   both against live EDGAR before trusting any of it.
2. Health-check collection: `shadow_report --all`, plus the new
   `by_grade`/coverage/FPI-share stats — what fraction of watchlist names
   resolved a CIK and a full structural profile, and what share are FPIs.
3. **Calibrate, in §15 priority order** — grade-vs-`close_retention` separation first
   (it validates or kills the v2 thesis), then `dilution_urgency` against
   `dilution_event_within_5d`, then momentum tiers, then insider conviction.
4. **Phase 2** — Trigger Engine → Disposition Engine → dedup/escalation state → Alert
   Manager. The `--mode live` runner stub is already wired. Deliberately sequenced
   *after* Phase 3 and calibration (2026-08-18 decision — see design-v2.md
   §Roadmap): thresholds are meant to be fit from shadow data, not hand-set, and the
   §8.2 escalation rule depends on the intraday filings poll that Phase 3 now provides.
5. Replace the linear RVOL session-fraction (`metrics.py`) with the real U-shaped
   volume profile from accumulated shadow data.

**Known approximations to revisit first:** real U-shaped volume profile, IBKR lot
size, gapper coverage *(v1)* — plus ticker→CIK coverage, EFTS phrase precision, and
shelf-capacity parsing *(v2)*.

---

*This document specifies a monitoring and research tool. Alert dispositions are
classifications of structural and momentum evidence, not investment advice; nothing
here is a recommendation to buy, sell, or short any security. Microcap and
sub-$5 equities carry a real probability of total loss.*
