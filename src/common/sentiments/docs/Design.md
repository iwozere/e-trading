# Design

## Purpose
The sentiment analysis module provides real-time sentiment scoring for financial instruments by aggregating data from multiple social media sources. It serves the short squeeze detection pipeline and other trading strategies that require sentiment-based signals.

Since Rev 2 ([`sentiment-spec-rev2.md`](sentiment-spec-rev2.md)), the module produces **two
structurally distinct signal classes** rather than one blended score: `retail` (hype/positioning
chatter -- StockTwits, Bluesky, Reddit, Twitter, Discord) and `tech_discourse` (engineering-
reputation discussion -- Hacker News). They are never averaged together; `tech_sentiment_24h` is
reported as an independent feature alongside `sentiment_24h`, not folded into it.

## Architecture
The system follows an async adapter pattern that allows concurrent data collection from multiple providers while maintaining rate limit compliance and error resilience.

### High-Level Architecture
- **Async Collectors**: Individual adapters for each data source
- **Signal-class routing**: `collect_sentiment_async.py` splits provider summaries into
  `retail`/`tech_discourse` buckets by `adapter.signal_class` (never by adapter name) before any
  aggregation happens
- **Entity resolver**: whole-word ticker↔company-name matching (`entity/resolver.py` +
  `entity/tickers.yml`), shared by Hacker News (name-only, no cashtags) and Bluesky
  (cashtag + company-name union)
- **Per-source calibration**: raw scores are z-scored against each provider's own trailing
  30-day distribution before blending (`processing/calibration.py` + `ss_sentiment_calibration`),
  since raw scores aren't comparable across platforms
- **Aggregation Engine**: Combines multiple retail sentiment signals (weighted, renormalized
  across whichever providers actually responded)
- **HuggingFace Integration**: Optional ML-based sentiment analysis, per-source model routing
  (retail-tuned vs. tech_discourse-tuned model, each its own adapter instance)
- **Batch Processing**: Efficient concurrent processing of multiple tickers

### Component Design
- **AsyncStocktwitsAdapter**: Social sentiment from StockTwits platform (`retail`, unauthenticated
  public access -- see Requirements.md for a 2026-08-20 access-terms verification note)
- **AsyncRedditAdapter**: Reddit sentiment via direct Reddit OAuth2 API (`retail`, requires app credentials)
- **AsyncBlueskyAdapter**: Bluesky sentiment via the `atproto` SDK (`retail`, app-password auth;
  gated off pending credentials) -- cashtag+company-name query union, time-window pagination
  fallback on 403
- **AsyncHackerNewsAdapter**: Hacker News discussion (`tech_discourse`, no auth) -- shared-corpus
  fetch strategy (fetch once per batch, entity-match every ticker against it in-process) keeps
  cost O(corpus size) instead of O(tickers); HTML/code-block cleaning before scoring
- **AsyncHFSentiment**: ML-based sentiment using HuggingFace transformers, one instance per
  signal class (`"huggingface"` retail model, `"huggingface_tech"` tech_discourse model)
- **SentimentFeatures**: Standardized output dataclass, `tech_*` fields explicitly `None` (never
  a fabricated neutral) when a ticker isn't in the entity map at all

## Data Flow
1. **Input**: List of ticker symbols
2. **Collection**: Concurrent API calls to multiple providers; Hacker News's shared corpus is
   fetched once per batch, not once per ticker
3. **Processing**: Heuristic (per-source lexicon) and ML-based (per-source model) sentiment analysis
4. **Calibration**: raw per-provider scores z-scored against trailing history (falls back to raw
   scores until a provider has accumulated enough history)
5. **Aggregation**: Weighted combination of retail signals only; tech_discourse reported separately
6. **Output**: Normalized sentiment features per ticker, split into retail (`sentiment_24h`) and
   tech_discourse (`tech_sentiment_24h`) fields

## Design Decisions

### Technology Choices
- **Async/Await Pattern**: Enables high-concurrency data collection
- **aiohttp**: Non-blocking HTTP client for API calls
- **HuggingFace Transformers**: State-of-the-art sentiment models
- **ThreadPoolExecutor**: Isolates blocking ML inference from async loop

### Architecture Patterns
- **Adapter Pattern**: Standardized interface for different data sources
- **Factory Pattern**: Configurable provider instantiation
- **Semaphore Pattern**: Rate limiting and concurrency control
- **Circuit Breaker**: Graceful degradation on provider failures

### Performance Considerations
- **Concurrent Processing**: Multiple tickers processed simultaneously
- **Rate Limiting**: Respects API provider limits
- **Caching**: Avoids redundant API calls within time windows
- **Lazy Loading**: HF models loaded only when needed

### Security Decisions
- **Input Validation**: All ticker symbols sanitized
- **Error Isolation**: Provider failures don't affect other providers
- **Rate Limit Compliance**: Prevents API abuse
- **No Credential Storage**: Environment-based configuration
- **Salted-hash author IDs**: every adapter (StockTwits, Reddit, Twitter, Discord, Bluesky) hashes
  the native author ID (`SHA256(salt + native_id)`, `bot_detector.hash_author_id`) at the
  message-normalization boundary and never returns the raw username/handle/DID. A lightweight
  username-shape check (`is_bot_username`) runs *before* hashing so a bot signal (`meta.is_bot`)
  survives into aggregation even though the identifier itself doesn't
- **90-day raw_payload retention**: `ss_deep_metrics.raw_payload` is purged (nulled, not deleted)
  after 90 days by `scripts/run_sentiment_retention.py`; scalar metrics for the row are retained
  for historical backtesting