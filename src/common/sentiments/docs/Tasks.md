# Tasks

## Implementation Status

### ✅ COMPLETED FEATURES
- [x] Async StockTwits adapter with rate limiting
- [x] Async Reddit adapter (direct OAuth2 API)
- [x] HuggingFace sentiment model integration
- [x] Batch processing with concurrency control
- [x] Sentiment aggregation and normalization
- [x] Error handling and graceful degradation
- [x] **Rev 2 Phase 1** (2026-08-20): `signal_class`/`supports_cashtag` on `BaseSentimentAdapter`;
  `entity/` package (whole-word ticker resolver + curated `tickers.yml`, ~120 tech names); async
  Hacker News adapter (`tech_discourse` signal class, shared-corpus fetch, HTML/code-block
  cleaning, Postgres `ss_hn_corpus` cache); `collect_sentiment_batch` aggregation now routes
  retail vs. `tech_discourse` separately and never blends them; `tech_sentiment_24h` propagated
  through `TransientMetrics`/`DeepScanMetrics` end to end. See `sentiment-spec-rev2.md`.
- [x] **Rev 2 Phase 2** (2026-08-20): async Bluesky adapter (`retail` signal class) -- app-password
  auth via the `atproto` SDK, cashtag+company-name query union (ambiguous tickers cashtag-only),
  cursor pagination with a time-window fallback on 403 (401 fails loudly instead), English-only
  filtering, Bluesky-specific bot heuristics (post volume + new-account activity). Gated off
  (`providers.bluesky: false`) pending `BLUESKY_HANDLE`/`BLUESKY_APP_PASSWORD`. 22 tests, all
  mocked at the `atproto` client boundary -- no live network calls or real credentials needed to
  exercise it.
- [x] **Rev 2 Phase 3** (2026-08-20): per-source lexicons (`config/sentiments.json`'s
  `lexicons.tech_discourse` bucket, selected via `HeuristicSentimentAnalyzer(signal_class=...)`,
  every adapter now passes its own `self.signal_class`); per-source HF model routing
  (`"huggingface"` retail model + a second `"huggingface_tech"` adapter instance, both registered
  in `adapter_manager.py`) with an explicit `long_text_strategy` (`truncate` | `chunk_mean` |
  `chunk_max`) for HN comments that exceed 512 tokens; salted-hash author IDs now applied
  **uniformly** across StockTwits/Reddit/Twitter/Discord/Bluesky (previously Bluesky-only) --
  `hash_author_id`/`is_bot_username` in `bot_detector.py`, with each adapter recording a
  `meta.is_bot` signal from the native username *before* hashing it away; `virality_index`
  redefined per spec §2.5.5 into orthogonal unsigned-reach (`Σengagement / sqrt(unique_authors+1)`)
  and signed-sentiment quantities, with `normalized_engagement` as a per-batch percentile rank
  (see `_percentile_ranks`) rather than a raw count; new `processing/calibration.py` +
  `ss_sentiment_calibration` table pool a trailing 30-day per-provider score distribution and
  z-score-calibrate each raw score before blending (`data_quality.calibration` records
  `"ok"`/`"insufficient_history"`), injected into `collect_sentiment_batch` via a
  `calibration_lookup` callback mirroring `history_lookup`'s DB-agnostic pattern.
  Fixed alongside this phase (found while wiring HF routing): the "huggingface" adapter was never
  actually registered by `_initialize_adapters` (its provider-loop special-case was keyed on a
  `config["providers"]["huggingface"]` entry that never existed), so the entire HF-enhancement
  path was silently dead whenever `hf_enabled=True` -- now registered explicitly, gated on
  `hf_enabled`, outside the generic provider loop. 44 new tests (calibration math against a
  synthetic known-mean/std distribution, per-source lexicon selection, hash/bot-username helpers,
  percentile ranking, reach-vs-sentiment orthogonality).
  Known gap, deliberately out of scope: `mentions_growth_7d` stays a single blended-retail figure,
  not source-aware -- `_get_historical_mentions_async` in `daily_deep_scan.py` is a pre-existing
  stub that always returns `None` (predates this work), so there's no per-provider mention history
  to make source-aware yet; doing so needs its own schema (a `ss_sentiment_provider_history` table)
  and is deferred rather than bolted on here.

### 🔄 IN PROGRESS
- [ ] Rev 2 Phase 4: `coverage-report` CLI, structured-log observability metrics, §2.1.1 Stocktwits
  API-terms verification, 90-day `sentiment_raw_payload` retention job, docs pass
- [ ] Performance optimization for large batches
- [ ] Enhanced bot detection algorithms
- [ ] Sentiment trend analysis over time

### 🚀 PLANNED ENHANCEMENTS
- [ ] Twitter API integration (when available)
- [ ] Discord sentiment monitoring
- [ ] Real-time streaming sentiment updates
- [ ] Sentiment-based alert triggers
- [ ] Historical sentiment data storage

## Technical Debt
- [ ] Add comprehensive unit tests for all adapters
- [ ] Implement proper caching layer
- [ ] Add metrics collection and monitoring
- [ ] Improve error recovery mechanisms

## Known Issues
- The Pushshift-based Reddit adapter (`async_pushshift_adapter`) was removed 2026-08-18: Pushshift
  has been restricted to verified Reddit moderators since May 2023 (no public/developer access).
  The direct-API `AsyncRedditAdapter` remains but requires manually-approved app credentials.
- HuggingFace model loading can be slow on first run
- Rate limiting may need adjustment based on usage patterns

## Testing Requirements
- [ ] Unit tests for each adapter
- [ ] Integration tests with mock API responses
- [ ] Performance testing with large ticker batches
- [ ] Error handling tests for API failures

## Documentation Updates
- [x] API documentation for public methods
- [x] Usage examples and configuration guide
- [ ] Performance tuning guide
- [ ] Troubleshooting documentation