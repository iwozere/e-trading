# Tasks

## Implementation Status

### ✅ COMPLETED FEATURES

- [x] Database migration `002_kestrel_schema.py` — all 12 k20_* tables
- [x] `db/models.py` — SQLAlchemy ORM models
- [x] `db/repos.py` — Full repository layer
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
- [x] `jobs/register_jobs.py` — One-time job schedule registration (18 jobs)
- [x] 18 `run_*.py` scheduler entry scripts (was 20; consolidated to 18)
- [x] Test suite (9 test files, ~60 tests)
- [x] Module documentation (README, Requirements, Design, Tasks)

### 🔄 IN PROGRESS

- [ ] Sleeve B2 (spin-offs) — entry window 20-60 days post-spin not yet implemented
- [ ] `llm/risk_diff.py` — module implemented but not wired in: no run script, not registered in register_jobs.py, not called from any other module; needs a `run_llm_risk_diff.py` entry point and registration

### 🚀 PLANNED ENHANCEMENTS

- [ ] Revisions feed integration — enables full §4.2 scoring (set `REVISIONS_FEED_AVAILABLE=True`)
- [ ] Sleeve A: EV/EBITDA relative valuation scoring when data available
- [ ] Sleeve B: Index inclusion event screening (S&P/Nasdaq adds/removes)
- [ ] Performance attribution — realized P&L by sleeve in weekly report
- [ ] Backtester integration — validate sleeve screens against historical data

## Technical Debt

- [ ] `universe_loader.py`: async batch fundamentals (currently sequential asyncio.run per ticker)
- [ ] `gdelt_processor.py`: multi-day backfill needs its own orchestration script
- [ ] `risk_checker.py`: YAML fallback is a temporary measure; should always use k20_positions

## Known Issues

- GDELT GKG alias fuzzy matching at 0.93 threshold may miss 2-char ticker typos
- Trends poll has no persistent state for anchor-term calibration
- `EdgarDownloader.download_form4_filings()` downloads ALL form 4s for a date — large payload on busy days

## Testing Requirements

- [ ] Integration test: full morning chain with mock DB
- [ ] Integration test: /pos add → confirm_add → risk_checker roundtrip
- [ ] Performance test: universe_loader with 3000+ tickers

## Documentation Updates

- [ ] Add Telegram bot hook example for /pos handler
- [ ] Add deploy runbook (migration → register_jobs → enable jobs)
