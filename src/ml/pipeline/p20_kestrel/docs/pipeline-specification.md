# Kestrel (P20) — Trading Intelligence Pipeline — Technical Specification

**Version:** 1.2 · **Date:** 2026-07-02
**Changelog v1.2 (resolves review findings #1–#14):**
- #1 Interim Sleeve A scoring without revisions feed: weight renormalization (§4.2.1)
- #2 `k20_sentiment_daily` schema reconciled with §7.2; per-source field semantics table (§3.2)
- #3 `k20_alias_blocklist` table defined (§3.1, §7.2)
- #4 `universe_refresh` job added (§11); initial seed in Phase 1
- #5 `/pos` command grammar + position flow specified (§9.4)
- #6 `gdelt_process` is a **standalone P20 script** reading P15's GKG cache; P15 stays a pure downloader (§7.2)
- #7 All P20 tables use **`k20_` prefix** — no collisions with `trading_positions` / `trading_trades` in model_trading.py
- #8 `ingest_filings` cooperates with P15's `edgar_8k_index`: discovery from P15 cache, body fetch only in P20 (§11)
- #9 `k20_catalysts` gains alert-state columns; idempotent T−10/T−3 firing (§3.1, §5.1)
- #10 `k20_watchlist.state` enumerated; `llm_verdict` documented as denormalized cache with FK (§3.1)
- #11 GDELT cold start: one-time 60-day backfill job + min_periods warm-up rule (§7.2.1)
- #12 Morning jobs converted from clock-time to a dependency DAG via `k20_job_runs` (§11.1)
- #13 LLM providers named, token/cost estimates and monthly budget cap added (§8.3)
- #14 Crowding-score fallback made explicit with staleness rules and `components_used` audit (§7.6)

**Mode:** Semi-automated. The agent screens, monitors, scores, and alerts. The human reviews and places all orders manually.
**Objective:** Surface candidates with +50% potential over a 6–12 month horizon across three strategy sleeves, while requiring ≤ 30–60 minutes of human attention per day.

---

## 1. Scope & Operating Model

Research and monitoring pipeline, not an execution system.

- **Agent:** data ingestion, screening, catalyst tracking, sentiment monitoring, LLM filings analysis, scoring, watchlist management, position risk monitoring, notifications, weekly reporting.
- **Human:** final candidate approval, manual order placement at broker, position confirmation via `/pos` commands (§9.4).
- **Attention budget:** one daily digest + allowlisted push alerts only.

### 1.1 Existing infrastructure (not in scope)

| Component | Status | Notes |
|---|---|---|
| Nasdaq ticker list (file) | ✅ | loaded into `k20_universe` by `universe_refresh` (§11) |
| Free OHLCV data (EOD) | ✅ | |
| Fundamentals dataset | ✅ | |
| SEC EDGAR access, incl. **P15 `edgar_8k_index`** (daily universe-wide 8-K index) | ✅ | P20 reads this cache for discovery (#8) |
| FINRA TRF daily short-sale volume | ✅ | |
| PostgreSQL | ✅ | shared DB; P20 namespace = `k20_` prefix (#7) |
| `p15_daily.py` | ✅ | **pure downloader** — P20 adds no processing inside it (#6) |
| GDELT events + GKG daily download (P15, 14-day file retention) | ✅ | processed by standalone P20 `gdelt_process` |
| CNN Fear & Greed, CBOE put/call | ✅ | daily loads |
| Scheduler framework | ✅ | P20 jobs registered as P20-namespaced tasks |
| Telegram + email notifications | ✅ | |
| Connectors: Alpha Vantage, Finnhub, StockTwits, Reddit | ⚠️ | free-tier constraints per §7.1 |

---

## 2. Design Principles

1. **Digest-first, push-rarely.**
2. **Human-in-the-loop**; every actionable alert carries a full order-ticket draft.
3. **Process at ingest, not at query**; heavy transforms live in batch jobs, queries hit small aggregates.
4. **Budget-aware connectors** with an enforced `k20_request_budget` allocator.
5. **Idempotent, resumable jobs** with run-state in `k20_job_runs`; full audit trail.
6. **Stale-data guards**; degraded feeds are flagged, never silently used (see §7.6 for the trends fallback).
7. **Sentiment is an overlay, not a source of ideas.**
8. **Namespace isolation:** every P20 table, job, and metric is `k20_`-prefixed; P20 never writes to P15/P17/P18 tables.

---

## 3. Data Model (PostgreSQL, all tables `k20_`-prefixed)

### 3.1 Tables

```sql
k20_universe(
  ticker text PRIMARY KEY, exchange text, sector text, industry text,
  mcap numeric, adv_20d numeric,
  status text CHECK (status IN ('active','delisted','suspended')),
  updated_at timestamptz)

k20_company_aliases(
  ticker text, alias text, alias_type text CHECK (alias_type IN
    ('legal_name','short_name','brand','former_name')),
  PRIMARY KEY (ticker, alias))

k20_alias_blocklist(                                   -- #3, load-bearing for GDELT precision
  alias text PRIMARY KEY,                              -- normalized collision-prone alias, e.g. 'apple','target','visa'
  ticker text,                                         -- which ticker it would have mapped to
  match_policy text CHECK (match_policy IN
    ('legal_name_only',    -- only 'Apple Inc.' counts, bare 'Apple' ignored
     'name_plus_context',  -- bare name counts only if a finance theme co-occurs in V2Themes
     'never')),            -- alias fully disabled
  reason text, added_at timestamptz)

k20_signals(ticker text, date date, signal_type text, value numeric, sleeve text,
  PRIMARY KEY (ticker, date, signal_type))

k20_sentiment_daily(                                   -- #2 reconciled with §7.2
  ticker text, date date,
  source text CHECK (source IN ('gdelt','stocktwits','reddit','apewisdom','trends','av_news')),
  mentions numeric,            -- semantics per source, see §3.2
  avg_tone numeric,            -- NULL where source has no tone
  tone_std numeric,            -- GDELT only (§7.2 step 3)
  pos_score numeric,           -- GDELT V2Tone positive; AV positive share
  neg_score numeric,           -- GDELT V2Tone negative; AV negative share
  bullish_ratio numeric,       -- StockTwits only: bull/(bull+bear); NULL elsewhere
  top_domains jsonb,           -- GDELT only, audit (§7.2 step 3)
  mention_z20 numeric,         -- NULL during warm-up (§7.2.1)
  tone_z20 numeric,
  PRIMARY KEY (ticker, date, source))

k20_catalysts(                                         -- #9 alert-state columns added
  id bigserial PRIMARY KEY, ticker text, event_type text, event_date date,
  confidence text, source text, notes text,
  state text CHECK (state IN ('upcoming','date_changed','passed','cancelled')) DEFAULT 'upcoming',
  t10_alerted_at timestamptz,                          -- idempotent T−10 firing
  t3_alerted_at timestamptz,                           -- idempotent T−3 firing
  datechange_alerted_at timestamptz)

k20_watchlist(                                         -- #10 states enumerated
  ticker text, sleeve text, score numeric,
  llm_verdict text,            -- DENORMALIZED CACHE of the latest dossier verdict; source of truth is k20_llm_runs
  dossier_run_id bigint REFERENCES k20_llm_runs(id),   -- FK to the run that produced llm_verdict
  thesis_short text, added_at timestamptz,
  state text CHECK (state IN ('screening','candidate','active_position','rejected','expired')),
  PRIMARY KEY (ticker, sleeve))

k20_positions(                                         -- distinct from existing trading_positions (#7)
  id bigserial PRIMARY KEY, ticker text, sleeve text,
  entry_date date, entry_px numeric, size_pct numeric,
  stop_px numeric, t1_px numeric, t2_px numeric, trail_pct numeric,
  realized_thirds int DEFAULT 0, status text DEFAULT 'open', notes text)

k20_llm_runs(
  id bigserial PRIMARY KEY, ts timestamptz, ticker text, task_type text,
  input_ref text,              -- accession number / input hash → cache key (task_type, input_ref) UNIQUE
  output_json jsonb, model text,
  tokens_in int, tokens_out int, cost_usd numeric,     -- #13 spend tracking
  verdict text)

k20_request_budget(source text, date date, quota int, used int,
  PRIMARY KEY (source, date))

k20_job_runs(                                          -- #12 dependency signaling
  job text, run_date date,
  status text CHECK (status IN ('running','ok','failed','skipped')),
  started_at timestamptz, finished_at timestamptz, rows_out int, error text,
  PRIMARY KEY (job, run_date))

k20_alerts_log(ts timestamptz, ticker text, trigger text, payload jsonb, channel text)
```

### 3.2 `k20_sentiment_daily` — per-source field semantics (#2)

| source | mentions | avg_tone / tone_std / pos / neg | bullish_ratio | top_domains |
|---|---|---|---|---|
| gdelt | article count | from V2Tone | NULL | populated (audit) |
| stocktwits | message count | NULL | bull/(bull+bear) tagged msgs | NULL |
| reddit | post mention count | NULL | NULL | NULL |
| apewisdom | mention count (cross-check) | NULL | NULL | NULL |
| trends | anchor-normalized interest (0–100 rescaled) | NULL | NULL | NULL |
| av_news | article count | AV sentiment aggregates | NULL | NULL |

---

## 4. Sleeve A — Turnaround / Fallen Angels (40% of sleeve capital)

### 4.1 Screen (weekly, Sunday)

Hard filters: drawdown −40%…−75% from 2-year high; mcap ≥ $500M; 20d ADV ≥ $10M; net cash OR net debt/EBITDA < 3 OR interest coverage > 4×; revenue YoY ≥ −15%; positive GM, erosion < 500 bps.

Inflection triggers (scored): forward-EPS revisions up 60d *(gap 10.1)*; insider net buying 90d (Form 4); buyback authorization (8-K/10-Q); EPS+rev beat; activist 13D/13G; TRF short-volume ratio declining. Sentiment overlay: attention-vacuum bonus / hype penalty (§7). Technical confirmation: price above rising 50DMA OR ≥2 higher weekly lows; never below a falling 50DMA.

### 4.2 Scoring

Full formula (revisions feed available):
`score = 30·revisions + 20·insider + 15·balance_sheet + 15·technical_base + 10·buyback + 5·short_covering + 5·attention_vacuum`

#### 4.2.1 Interim mode without revisions feed (#1)

**Decision: renormalize, don't zero.** A neutral zero would deflate every candidate by up to 30 points, silently turning the ≥60 dossier threshold into ≥60/70≈86-percentile-equivalent and starving the funnel. Instead:

- Config flag `REVISIONS_FEED_AVAILABLE` (default false until 10.1 is wired).
- When false: `score_interim = round(score_partial × 100/70)` where `score_partial` is the sum over the remaining 70 points. Thresholds (60 dossier / 75 push) stay unchanged.
- Every dossier and order ticket produced in interim mode carries the tag `⚠ revisions:n/a` so the human knows the largest single signal is missing.
- Optional soft proxy while the gap is open: LLM guidance-language delta from the last two earnings 8-Ks may add/subtract up to ±5 points, clearly labeled `revisions_proxy`. It never fills the full 30-point weight.
- When the feed goes live: flag flips, formula reverts, and the weekly report shows both scores for two weeks (calibration overlap).

### 4.3 Exit rules

Stop −25%; scale out ⅓ @ +35%, ⅓ @ +60%, remainder 20% trail; breakeven after first third. Thesis-invalidation push alerts from LLM 8-K classification (§8).

---

## 5. Sleeve B — Event Catalysts (30% of sleeve capital)

### 5.1 B1: FDA run-up

PDUFA/AdCom/readout calendar *(gap 10.2)*; mcap $300M–$10B, cash runway > 12 months, event 30–90 days out. **Run-up only; exit 100% by T−3.** Max 2%/name, 4% aggregate. Crowding skip: social mention spike > 3σ before T−10.

**Idempotent countdown enforcement (#9):** `catalyst_sync` fires the T−10 alert only when `days_to_event ≤ 10 AND t10_alerted_at IS NULL`, then stamps the column; same for T−3. A slipped `event_date` (detected on re-verify) sets `state='date_changed'`, stamps `datechange_alerted_at`, emits one digest line, and **resets** `t10/t3_alerted_at` to NULL so countdowns re-arm for the new date.

### 5.2 B2: Spin-offs

EDGAR Form 10/10-12B monitor + external calendars. Entry window day 20–60 post-spin after volume normalization (20d vol < 50% of first-week avg) and 5 days without a new low. Mandatory LLM Form-10 dossier before watchlist entry.

### 5.3 B3: Index events & activists

S&P/Nasdaq index-change RSS; curated-activist 13D monitor via EDGAR full text. Watchlist candidates only.

---

## 6. Sleeve C — Momentum in Live Themes (30% of sleeve capital)

RS = 0.5·(3m pct) + 0.5·(6m pct); eligible: top decile, price > 50DMA > 200DMA, ADV ≥ $20M, positive revenue growth, breakout volume ≥ 1.5×. Theme tags: industry classification + LLM keyword extraction from 10-K item 1. Regime filter: SPY < 200DMA → no new entries, trails 12%; breadth < 35% → exposure halved. Crowding overlay per §7.6. Exits: 15–20% trail OR 3 closes below 50DMA.

---

## 7. Sentiment Layer

### 7.1 Source matrix — verified free-tier status (2026-07)

| Source | Free-tier reality | Role | Schedule |
|---|---|---|---|
| GDELT GKG (+events for macro flags) | free, already downloading (P15) | backbone: news attention & tone, full universe | daily, §7.2 |
| Fear & Greed, CBOE put/call | free, ingested | regime context lines | daily (existing) |
| Google Trends (pytrends) | free, unofficial, throttled | retail attention, watchlist-scale | weekly, §7.3 |
| StockTwits public symbol streams | unofficial endpoints readable with polite limits; may break | msg volume + bull/bear tags, watchlist | daily, ≤1 req/2s |
| Reddit OAuth free tier | ~60–100 q/min per client, personal use | r/wsb + r/stocks mentions | daily |
| ApeWisdom | fully free, no key | Reddit cross-check / fallback | daily, 1 req |
| Alpha Vantage NEWS_SENTIMENT | endpoint on free key, **25 req/day total** | precision instrument, positions + top candidates only | §7.5 allocator |
| Finnhub | 60/min free, but news- & social-sentiment **premium-locked** | free endpoints only (headlines, earnings calendar, recommendation trends) | daily |

### 7.2 GDELT processing — standalone P20 job (#6)

**Placement decision:** `gdelt_process` is a standalone script in the P20 codebase that **reads P15's GKG file cache** (path from shared config). P15 remains a pure downloader; no P15 code changes; no risk to P15/P17/P18.

Steps (unchanged logic from v1.1, schema references updated):
0. **Aliases:** weekly `alias_refresh` rebuilds `k20_company_aliases` from fundamentals + EDGAR names; `k20_alias_blocklist` (#3) enforces per-alias match policy for collision-prone names (bare "Apple", "Target", "Visa" → `legal_name_only` or `name_plus_context`).
1. **Filter GKG:** keep rows where V2Organizations matches an allowed alias (exact normalized; fuzzy ≥0.93 second pass logged separately) or V2Themes carry finance codes. Retention typically <2% of rows.
2. **Tone:** V2Tone fields → tone, pos, neg, polarity; keep source domain + URL for audit.
3. **Aggregate:** per (ticker, date): mentions, avg_tone, tone_std, top_domains → `k20_sentiment_daily(source='gdelt')`.
4. **Signals:** rolling 20d `mention_z20`, `tone_z20` (warm-up rule §7.2.1). Spike >3 → digest + crowding input; sustained <−1 + Sleeve A price criteria → attention-vacuum bonus; tone_z20 <−3 on a position → pairs with LLM 8-K check.
5. **Events file:** macro/geopolitical regime flags only; no ticker mapping.

Raw-file retention 14 days (P15 policy); aggregated rows kept forever; weekly precision spot-check sample in the Sunday report (target >90% true-positive matches).

#### 7.2.1 Cold start & backfill (#11)

- **One-time `gdelt_backfill` job (Phase 3):** GDELT's public archive allows fetching historical GKG beyond P15's 14-day retention. Backfill **60 days** of GKG for the current universe before enabling signals — primes the 20-day baselines with margin and gives immediate usable z-scores.
- **Warm-up rule (defense in depth):** z-scores are computed only with `min_periods = 15` days of history for that (ticker, source); otherwise stored as NULL. Digest renders NULL as `warming up (d/20)` and no sentiment bonus/penalty is applied to scores. This also covers newly listed tickers and any future source additions.

### 7.3 Google Trends via pytrends

Unchanged from v1.1: `"<TICKER> stock"` query form; anchor term in every ≤5-term batch with rescale `norm_i = raw_i × (anchor_ref / anchor_batch)`; 30–60s jittered sleeps, abort on repeated 429 and retry next day; weekly 12-month window for baselines; watchlist-scale (~13 requests / 50 tickers). Failure degrades gracefully per §7.6.

### 7.4 Social polling (watchlist-scale, daily)

StockTwits public streams (circuit breaker, 1 req/2s), Reddit OAuth scan of new posts (cashtag/ticker regex), ApeWisdom daily table. Composite social z = max of per-source z-scores.

### 7.5 Alpha Vantage budget allocator

20 calls/day (5 reserved for retries) via `k20_request_budget`: positions first, then top candidates by score, then rotating watchlist coverage; unserved names lead tomorrow's queue.

### 7.6 Crowding score — explicit fallback contract (#14)

Implemented in `sentiment_aggregator.py`, not prose:

```
components = {
  'social':  z_social,        # max(stocktwits, reddit, apewisdom) z
  'gdelt':   mention_z20,
  'trends':  z_trends,
}
STALENESS_DAYS = {'social': 3, 'gdelt': 2, 'trends': 10}
usable = {k: v for k, v in components.items()
          if v is not NULL and age_days(k, ticker) <= STALENESS_DAYS[k]}
crowding = mean(usable.values())          # equal weights over usable set
row.components_used = list(usable.keys()) # audit column on the aggregate row
```

Rules: if `trends` exceeds 10-day staleness (pytrends broken/blocked) it drops out silently from the score but **loudly** from the digest — `data_health` emits one line "trends stale since <date>". If fewer than 2 components are usable for a ticker, crowding is NULL and the Sleeve C crowding gate does not fire (fail-open for blocking, fail-closed for bonuses: attention-vacuum bonus also requires ≥2 usable components).

---

## 8. LLM Analysis Layer

Funnel position: quant screens (universe, pennies) → LLM dossiers (10–20 names/week) → human (≤3/day).

### 8.1 Tasks

| Task | Trigger | Output |
|---|---|---|
| 8-K / PR classification | new filing, watchlist+positions, every 2h | `{event_type, materiality, thesis_impact, one_liner}` → invalidation alerts |
| 10-K/10-Q risk diff | new annual/quarterly on watchlist | added/removed risks, red_flags[] |
| Earnings-call tone | transcript available | confidence/evasiveness markers, guidance delta |
| Form 10 dossier | new spin-off registration | 1-page SpinCo summary |
| Candidate dossier | quant score ≥ 60, weekly batch | JSON per §8.2 |
| Guidance-delta proxy (interim, §4.2.1) | Sleeve A candidate in interim mode | `revisions_proxy` ∈ [−5, +5] |

### 8.2 Candidate dossier contract

Unchanged from v1.1 (thesis, bull/bear cases, red_flags with source pointers, catalysts_ahead, verdict ∈ {advance, watch, reject} with mandatory reject reason, confidence, invalidation line, sources). Stored in `k20_llm_runs.output_json`; `k20_watchlist.llm_verdict` + `dossier_run_id` cache the latest verdict (#10).

### 8.3 Providers, token volumes, and budget (#13)

| Task | Model | Est. tokens/run (in/out) | Volume | Est. cost/month* |
|---|---|---|---|---|
| 8-K classification | Anthropic **claude-haiku-4-5** | ~6k / 0.3k | ~20–40 filings/day | ~$3–8 |
| Guidance-delta proxy | claude-haiku-4-5 | ~10k / 0.5k | ~10/week | ~$1 |
| 10-K/Q risk diff | Anthropic **claude-sonnet-4-6** | ~40k / 2k | ~5–10/week | ~$8–20 |
| Form 10 dossier | claude-sonnet-4-6 | ~60k / 2k | ~1–3/month | ~$2–5 |
| Candidate dossier | claude-sonnet-4-6 | ~30k / 2k | ~10–20/week | ~$10–25 |
| **Total** | | | | **≈ $25–60/month** |

\*Order-of-magnitude estimates for budget validation; verify against current API pricing at build time (docs.claude.com) — pricing and model names change.

Controls:
- `LLM_MONTHLY_BUDGET_USD` (default 75). `cost_usd` accumulated in `k20_llm_runs`; at 80% of budget → digest warning; at 100% → dossiers pause (classification — the risk-critical task — keeps running until 120%, then full stop + push alert).
- Cache by `(task_type, input_ref)` unique key: a filing is analyzed exactly once.
- Input trimming: pre-extract only relevant filing sections (Risk Factors, MD&A, Item 1) before the LLM call; never send whole 10-Ks.
- Structured-JSON prompting with schema validation, one retry on parse failure.
- Weekly calibration: verdicts vs 4/12-week forward returns.
- LLM never overrides quant hard filters.

---

## 9. Notification Policy

### 9.1 Daily digest — 07:30 Europe/Zurich

Regime line (SPY/200DMA, breadth, VIX, F&G, put/call) · open positions with distance-to-stop and sentiment flags · catalysts next 5 days · new candidates (max 3) with score + LLM verdict + thesis line · top-3 sentiment anomalies · data-health warnings (incl. trends staleness, budget usage, warm-up notices).

### 9.2 Push allowlist

Stop/target touched · T−10/T−3 countdowns (idempotent per §5.1) · candidate score ≥75 + `advance` (with order ticket) · position −12% intraday · LLM `thesis_impact: invalidates` + `materiality: high` · circuit-breaker · LLM budget 120%.

### 9.3 Weekly pack — Sunday 18:00

Performance vs SPY/QQQ, sleeve attribution, funnel stats, 2-week catalyst calendar, LLM calibration + spend, GDELT alias-precision sample, interim-mode score overlap report when 10.1 goes live.

### 9.4 `/pos` command grammar (#5)

Telegram bot commands writing to `k20_positions` (numbers are examples):

```
/pos add TICKER SLEEVE ENTRY_PX SIZE_PCT [stop=PX] [t1=PX] [t2=PX] [trail=PCT]
    e.g. /pos add SNAP A 8.42 3.0 stop=6.30
    Defaults if omitted: stop = entry×0.75, t1 = entry×1.35, t2 = entry×1.60, trail = 20
    Bot replies with a full echo card; position becomes 'open' only after user taps ✅ Confirm.
/pos scale TICKER THIRD_N PX      -- records a realized third; auto-moves stop to breakeven after third 1
/pos stop TICKER PX               -- manual stop adjustment
/pos close TICKER PX [reason]     -- closes position, logs realized P&L
/pos list                         -- current open positions with distance-to-stop
```

Fallback input path: a watched YAML file (`positions.yml`) ingested by `risk_check` for bulk edits. `k20_watchlist.state` transitions to `active_position` on confirmed add, back to `expired` on close. Ships in **Phase 2** together with `risk_check` — the digest's positions section renders "no positions tracked" until then (documented, not a bug).

---

## 10. Data Gaps — remaining

| # | Gap | Priority | Note |
|---|---|---|---|
| 10.1 | Earnings estimate revisions / forward consensus | CRITICAL | Sleeve A runs in interim renormalized mode (§4.2.1) until wired; candidates tagged `revisions:n/a` |
| 10.2 | Event calendars (earnings dates, PDUFA, spin-offs, index changes) | CRITICAL | Finnhub earnings calendar free; PDUFA scrapes + weekly manual verify |
| 10.3 | NYSE/AMEX universe extension | CRITICAL | Nasdaq-only halves Sleeve A/B opportunity set |
| 10.4 | Official FINRA short interest (bi-monthly) | HIGH | TRF ≠ outstanding SI |
| 10.5 | PR wire RSS | MEDIUM | 8-K lag mitigation; GDELT partially covers |
| 10.6 | Corporate actions / sector classification | MEDIUM | verify OHLCV adjustments; SIC fallback |
| 10.7 | Options IV / flow | LOW | phase 2, broker API |

---

## 11. Jobs (P20 namespace; #6 #8 #12 applied)

| Job | Schedule / trigger | Description |
|---|---|---|
| `universe_refresh` (#4) | Weekly Sun 08:30 + one-time seed in Phase 1 | Load Nasdaq ticker file → `k20_universe`; refresh mcap/ADV from fundamentals+OHLCV; mark delisted via corporate-actions feed |
| `ingest_eod` | Daily 23:30 | OHLCV, TRF, corporate actions |
| `ingest_filings` (#8) | Every 2h, 13:00–00:00 | **Discovery from P15 `edgar_8k_index` cache** (no duplicate index download); P20 fetches filing **bodies** only for watchlist+positions tickers; Form 4/13D/Form 10 deltas |
| `llm_classify_filings` | On `ingest_filings` completion | 8-K/PR classification |
| `gdelt_backfill` (#11) | One-time, Phase 3 | 60-day GKG archive backfill to prime z-score baselines |
| `gdelt_process` (#6) | DAG: after P15 GDELT download flag | Standalone P20 script over P15's GKG cache; steps §7.2 |
| `social_poll` | Daily, DAG start 05:30 | StockTwits/Reddit/ApeWisdom, watchlist-scale |
| `av_sentiment_budgeted` | Daily, DAG start 05:30 (parallel) | AV NEWS_SENTIMENT ≤20 tickers |
| `sentiment_aggregate` (#12) | **DAG: on completion of the three jobs above** (see §11.1) | composite z-scores, crowding (§7.6), anomalies |
| `risk_check` | Daily 23:45 | stops/targets/invalidations |
| `momentum_rank` | Daily 23:50 | RS, regime, crowding overlay |
| `catalyst_sync` | Daily 06:30 | calendar refresh, idempotent T-minus (§5.1) |
| `digest_send` | Daily 07:30 hard deadline | daily digest |
| `screen_turnaround` | Weekly Sun 10:00 | Sleeve A screen (§4.2/4.2.1) |
| `screen_spinoffs` | Weekly Sun 11:00 | SpinCo monitor |
| `llm_dossiers` | Weekly Sun 12:00 | dossiers for score ≥ 60 |
| `trends_watchlist` | Weekly Sat 09:00 | pytrends batched pull |
| `alias_refresh` | Weekly Sun 09:00 | aliases + blocklist review sample |
| `weekly_report` | Weekly Sun 18:00 | performance + calibration + spend |
| `data_health` | Daily 07:00 | freshness checks, budget usage, staleness lines |

### 11.1 Morning DAG instead of clock times (#12)

The 05:30–06:15 fixed-clock chain from v1.1 is replaced by dependency signaling through `k20_job_runs`:

```
P15 gdelt download ─▶ gdelt_process ──┐
social_poll ──────────────────────────┼─▶ sentiment_aggregate ─▶ (07:00 gate) ─▶ digest_send 07:30
av_sentiment_budgeted ────────────────┘
```

- Each job writes `running → ok/failed` to `k20_job_runs`. `sentiment_aggregate` starts when all three upstreams report `ok` for today's `run_date`, polling every 5 min.
- **Deadline rule:** at 07:00, `sentiment_aggregate` runs with whatever upstreams completed, marks missing sources per the staleness contract (§7.6), and the digest carries an explicit `⚠ partial sentiment: gdelt pending` line. The digest itself is never delayed past 07:30 — a partial digest on time beats a complete digest at noon.
- Early/long GDELT runs are therefore harmless: first runs (backfill, cold caches) can take hours without breaking the morning chain.

---

## 12. Build Phases

| Phase | Deliverable |
|---|---|
| 1 | `k20_` schema + `universe_refresh` seed + `ingest_eod` + momentum ranking + daily digest |
| 2 | Risk tracker + `/pos` bot commands (§9.4) + push-alert engine + `k20_job_runs` DAG plumbing |
| 3 | `gdelt_backfill` + `gdelt_process` + alias table & blocklist |
| 4 | Sleeve A screen in interim mode (§4.2.1) + Form 4/13D parsers; revisions feed (10.1) when sourced |
| 5 | LLM layer: 8-K classifier first, then dossiers; budget tracking |
| 6 | Sleeve B calendars + idempotent T-minus + spin-off monitor |
| 7 | Social polling + pytrends + crowding overlay (§7.6) |
| 8 | Weekly reporting, calibration, NYSE/AMEX extension |
| 9 (opt.) | Backtests, options data |

---

## 13. Non-Goals & Boundaries

No order execution; no intraday logic; sentiment never generates entries alone; LLM never overrides quant hard filters; P20 never modifies P15/P17/P18 code or tables. Output is research/monitoring for the operator's own decisions — not investment advice; operator verifies current data before acting.
