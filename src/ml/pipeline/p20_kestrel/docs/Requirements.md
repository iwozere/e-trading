# Requirements

## Python Dependencies

- `anthropic` >= 0.25 — LLM API client
- `pandas` >= 2.0 — Data processing
- `sqlalchemy` >= 2.0 — Database ORM
- `requests` >= 2.31 — HTTP calls (StockTwits, Reddit, ApeWisdom, FMP, Finnhub)
- `praw` >= 7.7 — Reddit OAuth (optional, falls back to requests)
- `pyyaml` >= 6.0 — YAML position file fallback (optional)

## External Dependencies

- `src.data.db.core.database` — `session_scope()` context manager
- `src.data.db.core.json_types` — `JsonType` for JSONB columns
- `src.data.market_data.data_manager` — `DataManager`, `get_ohlcv_batch()`
- `src.data.fundamentals.get_fundamentals_unified` — async fundamentals fetch
- `src.data.edgar.edgar_downloader` — `EdgarDownloader`
- `src.notification.service.client` — `NotificationServiceClient.send_to_admins()`
- `src.notification.logger` — `setup_logger()`
- `src.data.db.models.model_jobs` — `Schedule` model for job registration

## External Services

- **Anthropic API** — LLM calls (claude-haiku-4-5, claude-sonnet-4-6)
- **AlphaVantage** — News sentiment (20 calls/day quota)
- **StockTwits** — `/api/2/streams/symbol/{ticker}.json` (rate: 2.1s delay; via a curl_cffi Chrome-impersonating session — Cloudflare fingerprints TLS and 403s plain `requests` from the Pi; 600s budget per run)
- **Reddit** — OAuth2, r/stocks r/investing r/wallstreetbets
- **ApeWisdom** — `https://apewisdom.io/api/v1.0/filter/all-stocks/page/{n}` (all ~10 pages, ~1,000 tickers)
- **Wikipedia pageviews** — Wikimedia REST API (official, keyless; User-Agent with contact required); ticker → article via Wikidata SPARQL + MediaWiki search, cached in `DATA_CACHE_DIR/wikipedia/`
- **EDGAR** — GKG files, Form 4, 8-K, 13D/G, 10-K/Q via EdgarDownloader
- **Nasdaq** — Tickers CSV at path configured by `NASDAQ_TICKERS_CSV`
- **FMP** (`/stable/analyst-estimates`, `/stable/grades`) and **Finnhub**
  (`/stock/recommendation`) — Sleeve A revisions feed ingest (gap 10.1).
  `period=quarter` on FMP's analyst-estimates and Finnhub's `eps-estimate`/
  `revenue-estimate` are not available on the account's current plan tier
  (402/403) — only `period=annual` and the endpoints above are used.

## Environment Variables

- `SCHEDULER_SYSTEM_USER_ID` — user_id for job_schedules rows (default: 1)
- `REDDIT_API_KEY`, `REDDIT_API_SECRET`, `REDDIT_USER_AGENT` — optional; unset or the `.env.example` placeholder `xxx` means Reddit is skipped (ApeWisdom covers Reddit mentions)
- `FMP_API_KEY`, `FINNHUB_API_KEY` — required by `revisions_ingest.py`; job is
  skipped (logged) if either is unset

Note: `REVISIONS_FEED_AVAILABLE` is **not** an environment variable — it's a
plain `bool` constant in `config.py` (default `False`), flipped by editing
the file, not via env var.

## System Requirements

- PostgreSQL 14+ with `k20_*` tables created by Alembic migration `002_kestrel`
- Network access to GDELT GKG cache at `R:/data-cache/gdelt/gkg/`
- Results directory writable at project root `results/p20_kestrel/`

## Security Requirements

- Anthropic and Reddit credentials via environment variables only
- No credentials stored in code or config files

## Performance Requirements

- Morning chain completes < 30min (06:00–07:00 UTC)
- EOD ingest chain completes < 2h (20:00–22:00 UTC)
- LLM budget: ≤ USD 80/month hard cap (configurable)
