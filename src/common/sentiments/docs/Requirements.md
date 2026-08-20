# Requirements

## Python Dependencies
- `aiohttp` >= 3.8.0
- `asyncio` (built-in)
- `transformers` >= 4.20.0 (optional, for HuggingFace sentiment)
- `torch` >= 1.12.0 (optional, for HuggingFace sentiment)
- `atproto` >= 0.0.71 (Bluesky adapter; ships no `py.typed` marker, see `pyproject.toml`'s mypy
  overrides)
- `PyYAML` (entity/tickers.yml)

## External Dependencies
- `src.notification` - For logging infrastructure
- `src.data` - For data provider patterns
- `src.data.db` - `ShortSqueezeService`/`ss_deep_metrics`/`ss_hn_corpus`/`ss_sentiment_calibration`
  for the Hacker News corpus cache, calibration history, and raw_payload persistence/retention

## External Services
- **StockTwits API**: Public (unauthenticated) API for social sentiment data --
  `GET /streams/symbol/{ticker}.json`, no registered developer app or API key.
  **Verified 2026-08-20 (spec §2.1.1):** the endpoint is live and returns real data, but is
  behind Cloudflare bot protection -- a plain HTTP client without a browser-like `User-Agent`
  gets a Cloudflare challenge page (HTTP 403) instead of JSON; `AsyncStocktwitsAdapter` already
  sends one (`self.user_agents[0]`) and works correctly. Separately, StockTwits' *formal*
  developer/API registration program is currently under review and not accepting new
  registrations (per public reporting as of 2026-08) -- there is no path to get an officially
  registered/authenticated key today if this unauthenticated access is ever tightened further or
  revoked. No official rate limits are published for this path; the adapter's own conservative
  self-imposed concurrency/delay and retry/backoff are the only real protection. **Implication:**
  this access is inherently fragile (scraping-adjacent, not a contractual guarantee) -- Bluesky is
  the intended hedge, not a redundant nice-to-have (spec §0.1).
- **Reddit API**: Direct OAuth2 access via `AsyncRedditAdapter` (requires app credentials; the
  legacy Pushshift-based adapter was removed — Pushshift has been restricted to verified Reddit
  moderators since May 2023, with no public or developer access)
- **Bluesky (`app.bsky.feed.searchPosts`)**: `retail` signal class via the official `atproto` SDK,
  app-password auth (`BLUESKY_HANDLE`/`BLUESKY_APP_PASSWORD`). Gated off
  (`providers.bluesky: false`) until credentials are configured.
- **Hacker News (Firebase API)**: `tech_discourse` signal class, `https://hacker-news.firebaseio.com/v0/`
  -- no auth, no key, stable for a decade per the spec's own risk assessment.
- **HuggingFace Models**: Pre-trained sentiment analysis models, one per signal class (retail:
  `cardiffnlp/twitter-roberta-base-sentiment`, tech_discourse:
  `distilbert-base-uncased-finetuned-sst-2-english` by default, both configurable)

## System Requirements
- Memory requirements: 512MB minimum, 2GB recommended for HF models
- CPU requirements: Multi-core recommended for async processing
- Network: Stable internet connection for API calls

## Security Requirements
- Rate limiting compliance with external APIs
- No API keys stored in code (environment variables)
- Input validation for all external data

## Performance Requirements
- Response time targets: < 5 seconds for batch sentiment analysis
- Throughput requirements: 100+ tickers per minute
- Concurrent request limits: Configurable per provider