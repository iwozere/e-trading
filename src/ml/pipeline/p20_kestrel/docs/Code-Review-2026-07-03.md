# P20 Kestrel — Code Review Findings (2026-07-03)

Full review of all business-logic modules after the first production deployment
attempts on the Raspberry Pi surfaced two runtime bugs (universe_loader provider
chain, alias_builder per-ticker asyncio.run). This document lists every further
issue discovered, ordered by severity.

Status legend: 🔴 broken in production · 🟠 wrong/incomplete behavior · 🟡 performance or hygiene

---

## 🔴 CRITICAL

### C1. Hardcoded `R:/data-cache` Windows paths (4 files) — FIXED 2026-07-03

| File | Line | Path |
|---|---|---|
| `reporting/data_health.py` | 40 | `R:/data-cache/gdelt/gkg` |
| `sentiment/gdelt_processor.py` | 39 | `R:/data-cache/gdelt/gkg` |
| `ingest/filings_ingest.py` | 37–39 | `R:/data-cache/edgar/{8k/index,13f/form4,13f/13dg}` |
| `llm/classifier_8k.py` | 68 | `R:/data-cache/edgar/8k/…` |

On the production server the cache root is `/share/data-cache` (via
`DATA_CACHE_DIR` in donotshare). The `R:` drive doesn't exist on Linux, so:
- data_health always reports "gdelt stale"
- gdelt_processor always finds 0 GKG files → 0 sentiment rows
- filings_ingest reads no 8-K/Form4/13D-G caches → 0 signals
- classifier_8k never finds cached filing bodies

**Worst property: nothing crashes.** Every module degrades silently to empty
results. **Fix**: all four now derive paths from `config.DATA_CACHE_PATH`.

### C2. Crowding score (§7.6) can never be computed

`sentiment_aggregator._compute_crowding_for_ticker` reads `mention_z20` from
the latest sentiment row for each of: stocktwits/reddit/apewisdom (social),
gdelt, trends. **Only `gdelt_processor` ever computes `mention_z20`.**
`social_poll` and `trends_poll` store raw `mentions` only.

Crowding requires ≥ 2 usable components → with only gdelt ever populated,
`crowding_score` is **always NULL** for every ticker. This silently disables
the §7.6 crowding penalty in all three sleeves.

**Fix options** (needs a decision):
- (a) Compute z-scores inside the aggregator from `get_sentiment_history`
  (same warm-up rule as gdelt: ≥ 15 days), or
- (b) add z-score computation to social_poll and trends_poll after upsert.
Option (a) is preferred: one implementation, and pollers stay write-only.

### C3. `calendar_sync` and `risk_checker` never send push alerts

Both call `log_alert(..., channel="push")` which **only inserts a row into
k20_alerts_log**. No `NotificationServiceClient` call is made. Only
`daily_digest` actually sends anything. The T-10/T-3 catalyst countdown and
stop/target alerts — the pipeline's most time-critical outputs — currently go
nowhere except the database.

**Fix**: after `log_alert`, send via
`src.notification.service.client.NotificationServiceClient.send_to_admins`
(same pattern as `daily_digest.py:182-190`).

---

## 🟠 HIGH

### H1. risk_checker: no alert deduplication + stale prices

- Cron is `*/30 9-17 * * 1-5` (17 runs/day) but there is **no dedup**: the same
  `stop_hit` alert is logged on every run while the condition holds — up to
  17 duplicate alerts per position per day. `get_today_alerts()` exists in the
  repo/service precisely for this and is never called.
- The "intraday" check reads `get_latest_signal(ticker, "close")` — that's
  **yesterday's EOD close** written by `ingest_eod` at 20:00 UTC. All 17
  intraday runs evaluate the same stale price. Either wire a real intraday
  quote source or reduce the cron to once daily after EOD ingest.

### H2. social_poll: Reddit credentials use wrong env var names

`social_poll._get_reddit_headers()` reads `REDDIT_CLIENT_ID`,
`REDDIT_CLIENT_SECRET`, `REDDIT_USERNAME`, `REDDIT_PASSWORD`.
The project's `donotshare.py` defines `REDDIT_API_KEY`, `REDDIT_API_SECRET`,
`REDDIT_USER_AGENT` (no username/password — it was designed for
client-credentials flow). Reddit polling is therefore **always skipped** in
production. Decide: add the 4 password-grant vars to `.env`, or rewrite
`_get_reddit_headers` for the app-only OAuth flow with the existing vars.

### H3. filings_ingest: 13D/G activist matching is dead code

`_process_13dg_activist` filters `f.get("ticker")` — but per its own module
docstring (line 114-121), the P15 13D/G cache columns describe the **filer**
(the activist fund), not the subject company, and contain **no ticker column**.
`activist_matches` is always 0. Either parse the filing subject from the
document body (expensive) or match `entity_name` against the curated
`ACTIVISTS_JSON` list and log an unattributed activist-activity alert.

### H4. Form 4 signal semantics: `insider_buy_value_90d` holds a 1-day sum

`_process_form4` writes signal rows named `insider_buy_value_90d` with the
**single-day** buy total for that date. Sleeve A's scorer
(`_score_interim`) reads the latest value and treats it as a 90-day
aggregate against a $5M full-score threshold. Effect: insider scoring
(20 of 100 pts) is dramatically understated. Fix: sum
`get_signals(ticker, "insider_buy_value_90d", start=−90d)` in sleeve_a, or
aggregate the trailing 90 days at ingest time.

---

## 🟡 MEDIUM

### M1. sentiment_aggregator: ~35k sequential DB transactions per run

5 `get_latest_sentiment` calls per ticker (each its own `@with_uow`
transaction) × all active tickers (~7,000 post-filter) ≈ 35,000 round-trips.
Social data only exists for watchlist tickers anyway. Fix: restrict to
watchlist ∪ positions (matches what pollers actually populate), or add a
batched repo method (one query per source, `DISTINCT ON (ticker)`).

### M2. eod_ingest: per-ticker OHLCV fallback

If `get_ohlcv_batch` misses a ticker, `dm.get_ohlcv` is called per ticker —
for ~7k tickers with cold cache this can take hours. Acceptable once the
DataManager cache is warm; monitor first EOD run duration.

### M3. data_health.check_sentiment_staleness checks job status, not data age

The warning text says "no recent ok run (>3d)" but the code only checks
*yesterday's* job-run status. A job that succeeded 2 days ago (within
staleness limits) still triggers a warning. Harmless but misleading; align
the check with `STALENESS_DAYS` semantics.

### M4. trends_poll stores normalized interest in the `mentions` column

Semantic reuse of the column (documented nowhere else). Fine, since the
aggregator would z-score it uniformly — but note it when implementing C2.

---

## Verified non-issues

- `av_budgeted`: `ALPHA_VANTAGE_API_KEY` env name matches donotshare, and
  donotshare's `load_dotenv()` populates `os.environ` before use (transitively
  imported via `config`). ✓
- LLM client budget gates and JSON-fence stripping logic are correct. ✓
- Sleeve hard-filter early-return chains match spec §4.1. ✓
- `llm/client.py` caches only successful parses (`output_json` not NULL). ✓

---

## Recommended fix order

1. **C1** — done (this commit); unblocks GDELT/filings testing on the Pi.
2. **C3** — small, high value: wire NotificationServiceClient into
   calendar_sync + risk_checker.
3. **H1** — dedup via `get_today_alerts` (one guard clause), decide on
   intraday price source separately.
4. **C2** — z-scores in aggregator (option a).
5. **H4** — 90-day insider aggregation in sleeve_a.
6. **H2 / H3** — need product decisions (Reddit auth flow; 13D/G matching).
7. **M1** — aggregate scope reduction, trivial win.
