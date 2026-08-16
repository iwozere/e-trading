# Tasks

## Implementation Status

### ✅ COMPLETED FEATURES

- [x] Database migration `002_kestrel_schema.py` — all 12 k20_* tables
- [x] `src.data.db.models.model_kestrel` — ORM models (in shared data layer, git-tracked)
- [x] `src.data.db.repos.repo_kestrel` — Session-based KestrelRepo class
- [x] `src.data.db.services.kestrel_service` — KestrelService(BaseDBService) with @with_uow
- [x] All P20 modules migrated to `KestrelService` — no direct session_scope() usage
- [x] `config.py` — All tuning constants
- [x] `ingest/universe_loader.py` — Nasdaq CSV → fundamentals → k20_universe
- [x] `ingest/eod_ingest.py` — EOD OHLCV + technicals → k20_signals
- [x] `ingest/filings_ingest.py` — Form 4, 8-K, 13D/G → k20_signals + llm_queue
- [x] `ingest/calendar_sync.py` — Catalyst T-10/T-3 alert countdown
- [x] `sentiment/alias_builder.py` — Company alias table + legal-name normalization
- [x] `sentiment/gdelt_processor.py` — GKG z-score pipeline with fuzzy matching
- [x] `sentiment/social_poll.py` — StockTwits/Reddit/ApeWisdom with rate limiting
- [x] `sentiment/trends_poll.py` — Google Trends with jitter and 429 abort
- [x] `sentiment/av_budgeted.py` — AlphaVantage priority queue with daily quota
- [x] `sentiment/sentiment_aggregator.py` — §7.6 crowding score
- [x] `screening/sleeve_a.py` — Hard filters + interim scoring (§4.2.1)
- [x] `screening/sleeve_b.py` — FDA run-up + activist screen
- [x] `screening/sleeve_c.py` — RS rank + regime filter + crowding overlay
- [x] `llm/prompts.py` — All prompt constants
- [x] `llm/client.py` — Budget-aware Anthropic client with caching
- [x] `llm/classifier_8k.py` — 8-K thesis impact classifier
- [x] `llm/dossier.py` — Candidate dossier generator
- [x] `llm/risk_diff.py` — 10-K/Q risk factor change detector
- [x] `risk/risk_checker.py` — Intraday stop/target/loss monitor
- [x] `pos/pos_commands.py` — /pos Telegram command parser
- [x] `reporting/daily_digest.py` — 07:30 digest builder + sender
- [x] `reporting/data_health.py` — 07:00 freshness guard
- [x] `reporting/weekly_report.py` — Sunday performance report
- [x] `jobs/register_jobs.py` — One-time job schedule registration (21 jobs)
- [x] 22 `run_*.py` scheduler entry scripts (21 scheduled + 1 manual backfill)
- [x] Test suite (16 test files, ~130 tests)
- [x] Module documentation (README, Requirements, Design, Tasks)
- [x] Sleeve B2 (spin-offs) — `screen_b2()` in sleeve_b.py; `get_past_spinoffs()` repo; B2 in run()
- [x] `llm/risk_diff.py` — wired: `run_llm_risk_diff.py` entry point + registered in jobs (Sunday 18:00 UTC)
- [x] `universe_loader.py` async batch — `_fetch_all_fundamentals()` with `asyncio.gather()` in 50-ticker batches
- [x] `gdelt_processor.py` multi-day backfill — `run_backfill(start, end)` + `run_gdelt_backfill.py`
- [x] `risk_checker.py` YAML fallback — removed; always reads k20_positions
- [x] Deploy runbook — added to README.md
- [x] Telegram bot hook example for /pos — added to README.md
- [x] Integration tests (2 test files: morning chain + /pos roundtrip)
- [x] Revisions feed ingest (gap 10.1, §4.2.1) — `ingest/revisions_ingest.py`
      populates `revisions_score` from FMP `analyst-estimates`/`grades` +
      Finnhub `recommendation` trends, registered as `p20_revisions_ingest`
      (20:50 UTC weekdays). Ships in **shadow mode**: writes signals but
      `REVISIONS_FEED_AVAILABLE` stays `False` (zero scoring impact) until
      reviewed.

### 🔄 IN PROGRESS

*(none)*

### 🚀 PLANNED ENHANCEMENTS

- [ ] Revisions feed flag flip — after reviewing shadow-mode `revisions_score`
      output for a few weeks, set `REVISIONS_FEED_AVAILABLE=True` in
      `config.py`. Verify `weekly_report.py` shows the §4.2.1 two-week
      dual-score calibration overlap before flipping.
- [ ] Recalibrate `REVISIONS_*` component weights in `config.py` once real
      shadow-mode data is available — current weights are a first-cut
      heuristic, not backtested.
- [ ] Sleeve A: EV/EBITDA relative valuation scoring when data available
- [ ] Performance attribution — realized P&L by sleeve in weekly report
- [ ] Backtester integration — validate sleeve screens against historical data

## Technical Debt

See [Code-Review-2026-07-03.md](Code-Review-2026-07-03.md) for full details.

- [x] **C2** — Crowding score (§7.6) never computed — fixed: z-scores derived in aggregator from history
- [x] **C3** — push alerts never sent — fixed: `notify.send_push()` wired into calendar_sync + risk_checker
- [x] **C4** — `get_signals(ticker, date)` arg misuse in sleeves A/C — fixed via `get_signals_for_date`
- [x] **C5** — sleeve_c regime filter treated float as dict — fixed
- [x] **C6** — sleeve_b B1 crowding check unreachable — fixed: applies to whole entry window
- [x] **C7** — `normalize_alias` left dangling punctuation — fixed
- [x] **H1** — risk_checker alert dedup — fixed: one (ticker, trigger) per day
- [x] **H4** — insider 90-day aggregation in sleeve_a — fixed
- [x] **M1** — aggregator scope reduced to watchlist ∪ positions — fixed
- [x] **H2** — Reddit polling — fixed: app-only OAuth via donotshare REDDIT_API_KEY/SECRET/USER_AGENT
- [x] **H3** — 13D/G matching — fixed: accession grouping + CIK→ticker + curated activists.json (Sleeve B3 live)
- [x] **H1b** — risk_checker intraday prices — fixed: yfinance delayed quote with EOD-close fallback
- [x] **M3** — data_health staleness — fixed: checks full STALENESS_DAYS window, not just yesterday
- [ ] **M2** — eod_ingest per-ticker fallback speed — monitor first production run
- [x] **C8** — Daily digest type conversion crash — fixed: reads float directly
- [x] **C9** — 8-K classifier key mismatch skips — fixed: uses accession_number and constructs URL
- [x] **C10** — Risk factor diff HTML index fetch — fixed: gets primaryDocument text
- [x] **H5** — Watchlist candidate drawdown fallback — fixed: calls get_latest_signal()
- [x] **L1** — LLM client cost fallback test failure — fixed

## Known Issues

- GDELT GKG alias fuzzy matching at 0.93 threshold may miss 2-char ticker typos
- Trends poll has no persistent state for anchor-term calibration
- `EdgarDownloader.download_form4_filings()` downloads ALL form 4s for a date — large payload on busy days
- ~~Hardcoded `R:/data-cache` paths in 4 modules~~ — fixed 2026-07-03 (C1)
- **Direct Reddit API (H2) is blocked at the platform level, not a code bug** — Reddit's Aug-2025 "Responsible
  Builder Policy" replaced self-serve app creation at `/prefs/apps` with an approval-gated request form;
  legitimate small non-commercial apps commonly get no response or a generic rejection. `social_poll.py`'s
  app-only OAuth code (H2) is correct and untouched; `REDDIT_API_KEY`/`SECRET`/`USER_AGENT` are left empty in
  `donotshare/.env` on purpose (2026-08-16). `_get_reddit_headers()` skips cleanly when unset, and ApeWisdom
  (`pipeline-specification.md` §social) already serves as the free, keyless Reddit-mention fallback in the
  composite `z_social` score — do not re-investigate the "not set; skipping" log line as a regression.

## Testing Requirements

- [x] Integration test: full morning chain with mock DB — `test_integration_morning_chain.py`
- [x] Integration test: /pos add → confirm_add → risk_checker roundtrip — `test_integration_pos_roundtrip.py`
- [x] Sleeve B: Index inclusion event screening (S&P/Nasdaq adds/removes) — Scraped via Wikipedia inside `p15_daily.py`, cached as CSV, and screened in `sleeve_b.py`
- [x] Unit tests for revisions_ingest.py (EPS-row selection, grades window, Finnhub momentum, score blending, shadow-mode run()) — `test_revisions_ingest.py`
- [ ] Performance test: universe_loader with 3000+ tickers (requires live data or large fixture)

## Documentation Updates

- [x] Add Telegram bot hook example for /pos handler — in README.md
- [x] Add deploy runbook (migration → register_jobs → enable jobs) — in README.md
