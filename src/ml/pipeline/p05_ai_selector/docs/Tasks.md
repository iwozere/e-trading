# Tasks — P05 AI Selector

Reference spec: [`pipeline-specification.md`](pipeline-specification.md)

---

## Phase 0 — Shared Infrastructure: Russell 3000 Downloader

*Prerequisite for all P05 phases and the P15 weekly bundle. Build and test this independently first.*

- [ ] **0.1** Check whether `src/data/downloader/tests/` exists; create it (with empty `__init__.py`) if absent.
- [ ] **0.2** Download the current Russell 3000 constituent list from Slickcharts (`https://www.slickcharts.com/russell3000`) and save as `src/data/downloader/data/russell3000_static.csv` with columns: `ticker, name, sector, industry, exchange`. Commit this file to the repo. This is the permanent quarterly fallback — update it manually after each FTSE Russell rebalance (March / June / September / December).
- [ ] **0.3** Create `src/data/downloader/russell3000_downloader.py` implementing `Russell3000Downloader(BaseDataDownloader)`:
  - `load(force: bool = False) -> pd.DataFrame` — returns cached CSV.gz if fresh (< 90 days old); otherwise calls `_fetch_from_fmp()`, falls back to `_load_static_fallback()` on any error.
  - `is_stale() -> bool` — True when `DATA_CACHE_DIR/universe/russell3000.csv.gz` is absent or `last_updated` in `russell3000_meta.json` is > 90 days ago.
  - `_fetch_from_fmp() -> Optional[pd.DataFrame]` — GET `/v3/russell_constituent`; normalise columns to `ticker, name, sector, industry, exchange`; return `None` on HTTP error or empty response.
  - `_load_static_fallback() -> pd.DataFrame` — read `src/data/downloader/data/russell3000_static.csv`; raise `FileNotFoundError` if missing.
  - On successful load from any source: write `DATA_CACHE_DIR/universe/russell3000.csv.gz` and `russell3000_meta.json` (`last_updated`, `source_used`, `row_count`).
  - Logger: `setup_logger(__name__)`. All logging lazy-formatted.
- [ ] **0.4** Create `src/data/downloader/tests/test_russell3000_downloader.py` with tests:
  - `test_load_returns_dataframe` — fresh cache exists; `load()` returns a non-empty DataFrame with correct columns; no HTTP call made.
  - `test_stale_cache_triggers_refresh` — cache older than 90 days; `load()` calls `_fetch_from_fmp()`.
  - `test_fmp_failure_falls_back_to_static` — `_fetch_from_fmp()` returns `None`; `load()` falls back to static CSV.
  - `test_static_fallback_columns` — static CSV has required columns.
  - `test_is_stale_no_cache` — no cache file; `is_stale()` returns `True`.
- [ ] **0.5** Add `russell3000_refresh` job to `src/ml/pipeline/p15_hidden_deps/p15_weekly.py`:
  - Import `Russell3000Downloader` at the top of the file.
  - Add `_job_russell3000_refresh()` function that calls `dl.is_stale()` and either skips (returns `{"skipped": True}`) or calls `dl.load(force=True)` and returns `{"rows": len(df), "source": dl.last_source_used}`.
  - Register it in `main()` after `fred_quarterly` and before `fred_combined`: `results["russell3000_refresh"] = _run_job("russell3000_refresh", _job_russell3000_refresh)`.
- [ ] **0.6** Smoke-test Phase 0: run `python src/data/downloader/russell3000_downloader.py` directly (add a `__main__` block that prints row count and first 5 rows). Verify `DATA_CACHE_DIR/universe/russell3000.csv.gz` is created.

---

## Phase 1 — Universe & Signal Data Layer

*Depends on Phase 0 being complete and `russell3000.csv.gz` present in cache.*

- [ ] **1.1** Create the full P05 directory skeleton (all empty `__init__.py` files):
  ```
  src/ml/pipeline/p05_ai_selector/__init__.py
  src/ml/pipeline/p05_ai_selector/stages/__init__.py
  src/ml/pipeline/p05_ai_selector/signals/__init__.py
  src/ml/pipeline/p05_ai_selector/tests/__init__.py
  ```
- [ ] **1.2** Create `src/ml/pipeline/p05_ai_selector/config.py` with a `P05Config` dataclass (or plain constants) containing:
  - `CRYPTO_TICKERS: list[str]` — the 20 confirmed coins (BTC-USD, ETH-USD, BNB-USD, SOL-USD, ADA-USD, AVAX-USD, DOT-USD, LINK-USD, MATIC-USD, UNI-USD, XRP-USD, LTC-USD, ATOM-USD, NEAR-USD, ICP-USD, FIL-USD, APT-USD, ARB-USD, OP-USD, DOGE-USD).
  - Stage 1 filter thresholds: `MIN_PRICE = 2.0`, `MIN_AVG_DAILY_VOLUME_USD = 5_000_000`, `STAGE1_LOOKBACK_DAYS = 60`.
  - Stage 1 → 2 soft cap: `STAGE1_TOP_N = 200`.
  - Stage 2 → 3 cap: `STAGE2_TOP_N = 25`.
  - Signal weights (all values from §6 of spec).
  - P18 output path pattern: `results/p18_institutional_flow_tracker/{date}/`.
  - LLM model: `LLM_MODEL = "claude-sonnet-4-6"`.
  - Paths: `RESULTS_BASE`, `STAGE1_CACHE_DIR`, `STAGE2_CACHE_DIR`, `EARNINGS_CACHE_DIR`.
- [ ] **1.3** Create `src/ml/pipeline/p05_ai_selector/stages/universe_loader.py` implementing `UniverseLoader`:
  - `load() -> list[str]` — calls `Russell3000Downloader().load()`, extracts `ticker` column, appends `CRYPTO_TICKERS` from config, deduplicates, returns sorted list.
  - No filtering at this stage (filtering is Stage 1's job).
  - Logs the total count at INFO level.
- [ ] **1.4** Create `src/ml/pipeline/p05_ai_selector/signals/p18_reader.py` implementing `P18Reader`:
  - `get_high_score_tickers(as_of_date: date) -> dict` — searches for the most recent P18 results directory on or before `as_of_date`. Loads `top_picks.csv` (or equivalent) and `consensus.csv` if present. Returns `{"high_score_count": int, "tickers": {ticker: score}, "consensus_tickers": set[str], "form4_buy_tickers": set[str], "13dg_tickers": set[str]}`.
  - If no P18 results found for the date: return the dict with all counts as 0 and empty sets (do not raise).
  - Logs whether P18 data was found and how many signals it carries.
- [ ] **1.5** Create `src/ml/pipeline/p05_ai_selector/signals/earnings_calendar.py` implementing `EarningsCalendar`:
  - `get_earnings_within_days(tickers: list[str], as_of_date: date, window_days: int = 7) -> dict[str, date]` — returns `{ticker: earnings_date}` for tickers with earnings within `window_days`.
  - Fetch from FMP `/v3/earning_calendar?from=YYYY-MM-DD&to=YYYY-MM-DD`; cache monthly to `DATA_CACHE_DIR/p05/earnings/YYYY-MM.csv.gz` (TTL 24h).
  - On FMP failure: log warning and return empty dict (do not crash the pipeline).
- [ ] **1.6** Create `src/ml/pipeline/p05_ai_selector/signals/options_flow.py` as a **no-op stub**:
  - Single function `get_unusual_activity(tickers: list[str], as_of_date: date) -> dict[str, bool]` that immediately returns `{}`.
  - Module docstring: "Phase 5 stub — requires Tradier API access. Currently returns empty."
- [ ] **1.7** Write tests in `src/ml/pipeline/p05_ai_selector/tests/`:
  - `test_universe_loader.py`: `test_load_returns_list`, `test_crypto_tickers_included`, `test_no_duplicates`.
  - `test_p18_reader.py`: `test_no_p18_data_returns_zeros`, `test_reads_top_picks_csv`, `test_date_fallback_finds_most_recent`.
  - `test_earnings_calendar.py`: `test_fmp_failure_returns_empty`, `test_filters_by_window`, `test_cache_hit_no_http_call`.

---

## Phase 2 — Signal Computing

*Depends on Phase 1. DataManager must be importable and `DATA_CACHE_DIR` configured.*

- [ ] **2.1** Create `src/ml/pipeline/p05_ai_selector/signals/technical.py` with pure functions (no class needed):
  - `compute_sma(prices: pd.Series, window: int) -> pd.Series`
  - `compute_rsi(prices: pd.Series, period: int = 14) -> float` — returns latest RSI value.
  - `compute_volume_surge_ratio(volumes: pd.Series, window: int = 20) -> float` — today's volume / rolling mean.
  - `compute_momentum_pct(prices: pd.Series, days: int = 5) -> float` — 5-day price return %.
  - `compute_atr_compression(high: pd.Series, low: pd.Series, close: pd.Series, short: int = 5, long: int = 20) -> bool` — True when short ATR < 0.7 × long ATR.
  - `compute_52w_proximity(prices: pd.Series) -> tuple[float, float]` — returns `(pct_from_high, pct_from_low)`.
  - `score_technicals(ohlcv: pd.DataFrame, weights: dict) -> tuple[float, dict]` — runs all signals, returns `(total_score, signal_breakdown_dict)`.
- [ ] **2.2** Create `src/ml/pipeline/p05_ai_selector/signals/fundamental.py` with:
  - `score_fundamentals(fundamentals: Optional[Fundamentals], sector_medians: dict, weights: dict) -> tuple[float, dict]` — applies P/E, margin, debt, growth, dividend signals; returns `(score, breakdown)`. If `fundamentals is None`: returns `(0.0, {})` immediately (no crash).
  - `build_sector_medians(all_fundamentals: dict[str, Fundamentals]) -> dict[str, dict]` — computes per-sector median P/E from the full candidate set; used so comparisons are relative, not absolute.
- [ ] **2.3** Create `src/ml/pipeline/p05_ai_selector/stages/stage1_prefilter.py` implementing `Stage1Prefilter`:
  - `run(tickers: list[str], as_of_date: date) -> pd.DataFrame` — main method.
  - For each ticker: fetch 60-day 1d OHLCV via `DataManager.get_ohlcv(ticker, "1d", start, end)`.
  - Apply hard filters: last close > `MIN_PRICE`; average daily volume × last close > `MIN_AVG_DAILY_VOLUME_USD`. Crypto: skip the volume-in-USD filter (use raw volume with a separate threshold).
  - Compute soft momentum score: SMA crossover (+15 or +10), RSI (+12 or +8), volume surge (+15), momentum_5d (+10), ATR compression (+5), 52w proximity (+8 each).
  - Return DataFrame sorted descending by score, capped at `STAGE1_TOP_N` rows. Columns: `ticker, asset_type, last_price, avg_vol_usd, momentum_score, signal_breakdown (json str)`.
  - Cache result to `DATA_CACHE_DIR/p05/stage1/{YYYY-MM-DD}.csv.gz`; load from cache if today's file already exists (idempotent re-runs).
  - Log: symbols_input, symbols_after_price_filter, symbols_after_volume_filter, symbols_output.
- [ ] **2.4** Create `src/ml/pipeline/p05_ai_selector/stages/stage2_scorer.py` implementing `Stage2Scorer`:
  - `run(stage1_df: pd.DataFrame, p18_data: dict, earnings_flags: dict[str, date], as_of_date: date) -> pd.DataFrame` — main method.
  - For each of the ~200 Stage 1 symbols: fetch fundamentals via `FundamentalsCache` → FMP → Yahoo (in order; missing = 0 pts).
  - Compute `build_sector_medians()` across all candidates before scoring.
  - Compute `score_fundamentals()` per ticker.
  - Apply P18 signal boosts: check each ticker against `p18_data["tickers"]`, `consensus_tickers`, `form4_buy_tickers`, `13dg_tickers` for the configured point values.
  - Tag `earnings_flag: bool` from `earnings_flags`.
  - Compute `total_score = momentum_score + fundamental_score + p18_score`.
  - Sort descending by `total_score`; on tie break by `volume_surge_ratio`. Return top `STAGE2_TOP_N` rows.
  - Output columns: `ticker, asset_type, last_price, market_cap_b, total_score, momentum_score, fundamental_score, p18_score, earnings_flag, earnings_date, fundamentals_available, signal_breakdown (json str)`.
  - Cache result to `DATA_CACHE_DIR/p05/stage2/{YYYY-MM-DD}.csv.gz`.
- [ ] **2.5** Write tests:
  - `test_technical_signals.py`: `test_rsi_oversold_scores`, `test_volume_surge_scores`, `test_sma_crossover_bullish`, `test_missing_data_returns_zero`.
  - `test_fundamental_signals.py`: `test_none_fundamentals_returns_zero`, `test_high_margin_scores`, `test_sector_median_pe_comparison`.
  - `test_stage1_prefilter.py`: `test_price_filter_removes_penny_stocks`, `test_volume_filter_applied`, `test_output_capped_at_top_n`, `test_cache_hit_skips_computation`.
  - `test_stage2_scorer.py`: `test_p18_boost_applied`, `test_fundamentals_missing_stays_in_funnel`, `test_output_capped_at_25`, `test_earnings_flag_set`.

---

## Phase 3 — LLM Integration & Output

*Depends on Phase 2. Requires `ANTHROPIC_API_KEY` in config.*

- [ ] **3.1** Create `src/ml/pipeline/p05_ai_selector/stages/stage3_llm_synthesizer.py` implementing `Stage3LLMSynthesizer`:
  - `run(stage2_df: pd.DataFrame) -> dict` — builds data packets, calls Claude, parses response.
  - `_build_data_packets(stage2_df: pd.DataFrame) -> list[dict]` — converts each row to the JSON schema defined in spec §7.1. Includes all technicals, fundamentals, P18 signals, earnings context.
  - `_call_claude(packets: list[dict]) -> dict` — constructs system prompt (spec §7.2), user message (serialised packets), defines tool_use schema (spec §7.3), calls `anthropic.Anthropic().messages.create(...)`. Returns raw tool result dict.
  - `_parse_response(raw: dict) -> dict` — validates structure; if `picks` missing or malformed, raises `ValueError` with detail. Extracts `notification_override` from the response.
  - Returns `{"picks": list[dict], "market_context": str, "notification_override": bool, "tokens_used": int}`.
  - On API error: log exception, re-raise (pipeline catches and marks run as failed gracefully).
- [ ] **3.2** Create `src/ml/pipeline/p05_ai_selector/stages/stage4_output.py` implementing `Stage4Output`:
  - `write_results(picks: list[dict], stage2_df: pd.DataFrame, metadata: dict, run_date: date) -> Path` — writes all output files to `results/p05_ai_selector/{YYYY-MM-DD}/`; returns the directory path.
    - `top_picks.csv`: rank, ticker, confidence, bias, thesis, time_horizon, first_profit_target_price, first_profit_target_action.
    - `full_ranking.csv`: all 25 rows from stage2_df plus the LLM rank/confidence where assigned.
    - `report.md`: human-readable Markdown report with market context, per-pick sections (thesis + full exit strategy formatted as prose), signal tables.
    - `metadata.json`: run_date, trigger_reason, p18_signals_count, notification_override, stage1_out, stage2_out, llm_tokens_used, llm_model, elapsed_seconds, timestamp.
  - `format_telegram(picks: list[dict], trigger_reason: str, run_date: date) -> str` — returns the Telegram message string (top 3 picks; condensed exit advice: first 2 thesis_breakers + first profit target). Max ~3800 chars (Telegram safe limit).
  - `format_email_html(picks: list[dict], market_context: str, trigger_reason: str, run_date: date) -> str` — returns full HTML string with summary table, per-pick sections including complete exit strategy, disclaimer footer.
  - `should_notify(p18_signals_count: int, notification_override: bool) -> tuple[bool, str]` — applies dual OR logic; returns `(True/False, trigger_reason_str)`.
- [ ] **3.3** Create `src/ml/pipeline/p05_ai_selector/pipeline.py` implementing `P05Pipeline`:
  - `run(user_id: int, as_of_date: Optional[date] = None, force_refresh: bool = False) -> dict` — 4-stage orchestrator:
    1. `UniverseLoader().load()` → tickers.
    2. `P18Reader().get_high_score_tickers(as_of_date)` → p18_data.
    3. `EarningsCalendar().get_earnings_within_days(tickers, as_of_date)` → earnings_flags.
    4. `Stage1Prefilter().run(tickers, as_of_date)` → stage1_df.
    5. `Stage2Scorer().run(stage1_df, p18_data, earnings_flags, as_of_date)` → stage2_df.
    6. `Stage3LLMSynthesizer().run(stage2_df)` → llm_result.
    7. `should_notify(...)` → `(notify, trigger_reason)`.
    8. `Stage4Output().write_results(...)` → results_dir.
    9. If `notify`: send Telegram and email via the notification system.
    10. Build and return result dict (all keys listed in spec §9 result dict table).
  - Timing: record `time.monotonic()` at each stage boundary; include per-stage elapsed in metadata.
  - Any stage raising an exception: log it, return `{"success": False, "error": str(e), ...}`.
- [ ] **3.4** Create `src/ml/pipeline/p05_ai_selector/run_p05_scan.py` — scheduler entry point:
  - `argparse` args: `--user-id` (int, required), `--as-of-date` (ISO date string, default today), `--force-refresh` (flag).
  - Calls `P05Pipeline().run(user_id, as_of_date, force_refresh)`.
  - Prints `__SCHEDULER_RESULT__:{json.dumps(result)}` to stdout.
  - Exit code 0 always (scheduler reads the JSON success field).
  - `PROJECT_ROOT = Path(__file__).resolve().parents[4]` for sys.path setup.
- [ ] **3.5** Write tests:
  - `test_stage3_llm.py`: `test_build_data_packets_structure`, `test_parse_response_valid`, `test_parse_response_missing_picks_raises`, `test_notification_override_extracted`. Mock the `anthropic` client — do not make real API calls in tests.
  - `test_stage4_output.py`: `test_write_results_creates_all_files`, `test_telegram_format_under_3800_chars`, `test_should_notify_p18_trigger`, `test_should_notify_override_trigger`, `test_should_notify_neither_returns_false`.

---

## Phase 4 — Integration, Testing & Documentation

- [ ] **4.1** Verify `condition_mode: "any"` is supported in the scheduler's notification rule evaluator (`src/scheduler/scheduler_service.py`). If not implemented: add OR-mode support to the rule evaluation logic before registering the SQL.
- [ ] **4.2** Run a full end-to-end dry run against a recent historical date (e.g. `--as-of-date 2026-06-10 --user-id 1`). Pass `--force-refresh` to bypass stage caches. Inspect all four output files in `results/p05_ai_selector/2026-06-10/`. Verify:
  - `top_picks.csv` has 5 rows.
  - `full_ranking.csv` has 25 rows.
  - `report.md` renders correctly (open in a Markdown viewer).
  - `metadata.json` has all expected keys.
  - Telegram message is under the character limit.
  - Email HTML renders without broken tags (open in a browser).
- [ ] **4.3** Check that the Stage 1 cold-start completes in under 60 minutes on first run (acceptable one-time cost). If it exceeds this, investigate DataManager batch fetch performance.
- [ ] **4.4** Register the scheduler job via SQL (spec §9). Confirm the job appears in the scheduler dashboard and the cron fires at 10:00 UTC on the next weekday.
- [ ] **4.5** Run one live scheduler-triggered execution. Verify the result dict is persisted to `job_schedule_runs`, notification rule evaluation fires correctly, and Telegram / email are received.
- [ ] **4.6** Create `src/ml/pipeline/p05_ai_selector/README.md` — overview, quick start, integration notes, link to spec.
- [ ] **4.7** Create `src/ml/pipeline/p05_ai_selector/docs/Requirements.md` — Python dependencies (`anthropic`, `pandas`, `requests`, `numpy`), cross-module dependencies (DataManager, FundamentalsCache, FMP downloader, Russell3000Downloader, notification system), external services (Anthropic API, FMP free tier).
- [ ] **4.8** Create `src/ml/pipeline/p05_ai_selector/docs/Design.md` — architecture diagram (the 4-stage funnel), data flow, key design decisions (single LLM call, dual-trigger notifications, deterministic scoring + LLM narrative separation), integration patterns.
- [ ] **4.9** Update this `Tasks.md` to mark completed items and add any new issues discovered during integration testing.

---

## Phase 5 — Enhancements (Deferred)

*Not scheduled. Pick up after Phase 4 is stable in production.*

- [ ] **5.1** `signals/options_flow.py` — implement real Tradier integration for unusual put/call volume once API access is secured.
- [ ] **5.2** Simple ML pre-filter — train an XGBoost or logistic regression model on 2-year history. Label: binary (top-quartile 5-day forward return). Features: Stage 2 signal scores. Replace or supplement deterministic Stage 2 ranking for equities. Needs a separate training pipeline and model artifact store.
- [ ] **5.3** Sector-relative scoring — instead of absolute thresholds (e.g. margin > 15%), compare each candidate vs. its sector peers within the Stage 2 candidate pool.
- [ ] **5.4** Short-candidate mode — a config flag that inverts the bias of SMA/RSI signals and filters for bearish setups. Useful in trending-down markets.
- [ ] **5.5** Backtesting module — track each weekly cohort of P05 top-5 picks against their actual 5-day, 20-day, and 60-day forward returns. Store in `results/p05_ai_selector/backtest/`. Use to calibrate signal weights and LLM confidence thresholds over time.

---

## Known Issues / Risks

| Risk | Mitigation |
|------|-----------|
| Stage 1 cold-start (~20–40 min on first run) | One-time cost; subsequent runs are fast. Document in README. |
| FMP earnings calendar may be incomplete for small-caps | `EarningsCalendar` returns empty on failure; LLM is not told about earnings it cannot find. |
| FMP fundamentals sparse for the smallest Russell 3000 stocks | Silent 0-score; ticker stays in funnel; LLM is told `"available": false`. |
| P18 may not have run yet on a given day (failure / off-hours) | `P18Reader` returns zero signals gracefully; P05 still runs and produces results. |
| Anthropic API rate limits or outage | Stage 3 raises; pipeline catches and returns `success: False`; no partial output written. |
| `condition_mode: "any"` may not be implemented in scheduler | Task 4.1 must be completed before scheduler registration. |
