Technical Specification (TS)

1) Goal & Deliverables

Goal: Build a modular Python service for downloading and caching market data (candles for crypto and equities) and fundamentals (equities) from multiple providers, with a single access façade DataManager, file cache and Redis cache, and adapters for backtesting and live trading (pandas/Backtrader).

Deliverables:

Installable library (pip package) + optional REST API (FastAPI).

Unified interface: DataManager.get_ohlcv(...), DataManager.stream_candles(...), DataManager.get_fundamentals_snapshot(...).

Normalized data format (OHLCV) and unified timeframes.

File cache (yearly segments) and Redis cache for metadata/hot windows.

Adapters: pandas DataFrame feed, Backtrader DataFeed (history + live).

Full test suite, docs, Docker image.



---

2) Scope & Data Sources

Crypto: Binance (spot/USDT pairs minimum), Coingecko (daily/hourly where available).
Equities: Yahoo Finance (yfinance), Financial Modeling Prep (FMP), Alpha Vantage, IBKR (where accessible).
Fundamentals (equities): FMP (priority), Alpha Vantage, yfinance, IBKR.

> Note: API availability and limits are controlled by config. Sources can be enabled/disabled.




---

3) Terms & Definitions

OHLCV: open, high, low, close, volume. Optional extra fields: vwap, turnover, trades.

Timeframes (TF): 1m, 5m, 15m, 1h, 4h, 1d. Extensible.

Segment: yearly candle file for (asset_type, symbol, timeframe, year).

Fundamentals snapshot: immutable set of fundamental metrics at as_of date.



---

4) Functional Requirements

4.1 Candle data

1. DataManager.get_ohlcv(symbol, timeframe, start, end, *, asset_type=None, adjusted=True, source_priority=None, force_refresh=False) -> pd.DataFrame

Auto-detect asset type (crypto/equity) via SymbolRegistry; fallback heuristics.

Merge segments from file cache and fetch missing ranges online.

Normalize to a single schema and UTC timezone (tz-aware DatetimeIndex[UTC]).

Filter final range [start, end], drop duplicates.

Equities: support adjusted=True (splits/dividends) where provider supports it; persist close_adj.



2. Caching:

File cache: base_dir/candles/{asset}/{symbol}/{timeframe}/{year}.parquet.gz (Parquet + gzip by default; CSV.gz as fallback).

Previous years (< current): considered frozen — not auto-refreshed.

Current year: marked hot. If last_update > 30 days, perform incremental refresh.

Redis cache: store coverage metadata, eTag/hash of segments, "hot windows" (e.g., last N candles) and staleness flags.

Redis keys: candles:{asset}:{symbol}:{tf}:meta, candles:{asset}:{symbol}:{tf}:{year}:etag, candles:{...}:hot.



3. Incremental updates:

Compute range "holes" by year. For the current year — refresh from last timestamp in the file + a safety rewind (e.g., 2×TF) to safely overwrite the edge.



4. Streaming (live) mode:

DataManager.stream_candles(symbol, timeframe, *, poll_interval, on_tick, run_until=None, backfill=True)

Guarantee monotonic time, merge with historical cache; guard against late corrections of the last candle.




4.2 Fundamentals (snapshots)

1. DataManager.get_fundamentals_snapshot(symbols, fields, as_of=None, provider_preference=None, force_refresh=False) -> pd.DataFrame

If as_of is not set — use the latest known snapshot(s).

File cache format: store snapshots as JSON files under base_dir/fundamentals/ with file names: provider_symbol_timestamp.json, where:

provider ∈ {fmp, alphavantage, yfinance, ibkr, ...}

symbol is raw symbol (e.g., AAPL), normalized to [A-Z0-9._-]

timestamp is ISO-like YYYYMMDDTHHMMSSZ (UTC)


Maintain an index (index.json) in each provider directory for quick lookup by symbol and freshness.

Cache-first policy: for all stock providers (yfinance, IBKR, FMP, Alpha Vantage, etc.), before hitting the API:

check file cache for the newest snapshot for each (provider, symbol);

if it is not older than 7 days, use the cached JSON;

if older or missing, fetch from the provider, then persist a new provider_symbol_timestamp.json.


Multi-provider combination: if multiple fresh snapshots exist (or are fetched) for the requested symbols, combine them into a single DataFrame by merging fields from all providers using normalized aliases:

unified fields: market_cap, pe_ttm, pb, ps, eps_ttm, div_yield, shares_out, sector, industry, currency, etc.

conflict resolution strategy (configurable):

1. priority order per field (e.g., fmp > ibkr > alphavantage > yfinance), or


2. most recent as_of, or


3. aggregation rule (e.g., prefer non-null; if all present & differ, keep provider‑suffixed columns like pe_ttm@fmp).






2. Update strategy:

Snapshots are append-only — never overwritten. New snapshots created on schedule or on-demand.

Schedules are configurable: daily/weekly/by earnings publications.




4.3 Symbol registry & routing

Unified SymbolRegistry: { symbol -> {asset_type, exchange, base_quote, provider_map, currency, delisted?} }.

Resolver decides provider order by priorities and availability (limits/quotas/TF coverage). Configurable source_priority.


4.4 Formats & normalization

Candle schema: timestamp, open, high, low, close, volume, vwap?, turnover?, trades?, provider.

dtypes: float64; timestamp — UTC tz-aware.

Ensure strictly increasing time; deduplicate; guard against zero/NaN candles.

Resampling (optional): resample_candles(df, to_tf, how="ohlcv").

Fundamentals normalization maps provider field names to unified aliases.



---

5) Non-Functional Requirements

Performance:

5 years of 1m history for liquid symbols is assembled from segments without reprocessing everything (read only needed years).

Parallel fetch by years/symbols (async/httpx with per‑provider throttling).


Reliability:

Retries with exponential backoff; detect 429/5xx; jitter.

Fallback to alternative provider on failures.


Consistency:

Edge candles of the current year are rewritten on each refresh (safety rewind).


Audit/logging:

Structured logs (json): provider, symbol, tf, years, bytes, latency, retries.

Prometheus metrics: fetch_latency_ms, cache_hit_ratio, api_calls_total, rate_limit_throttles.


Security: keep API keys in .env/Vault; never log secrets.

Portability: Linux/Windows, Python 3.11+.



---

6) Architecture & Modules

src/
  datacore/
    __init__.py
    config/                    # pydantic-based config
    managers/
      data_manager.py          # façade
    registry/
      symbols.py               # SymbolRegistry
    providers/
      base.py                  # abstract provider interfaces
      binance.py
      coingecko.py
      yfinance.py
      fmp.py
      alphavantage.py
      ibkr.py
    cache/
      file_cache.py            # parquet.gz segments + index
      redis_cache.py           # metadata / hot windows
    io/
      normalize.py             # schemas, aliases, resampling
      timeframes.py            # TF maps per provider
    feeds/
      pandas_feed.py           # convenient wrappers
      backtrader_feed.py       # bt.feeds.PandasData / live bridge
    scheduler/
      jobs.py                  # periodic jobs (apscheduler)
    api/
      rest.py                  # FastAPI (optional)

6.1 Candle provider interface

class CandleProvider(Protocol):
    name: str
    supported_tfs: set[str]
    asset_types: set[str]  # {"crypto", "equity"}

    async def fetch(
        self, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp,
        *, adjusted: bool = True
    ) -> pd.DataFrame:  # columns: timestamp, open, high, low, close, volume
        ...

6.2 Fundamentals provider interface

class FundamentalsProvider(Protocol):
    name: str
    async def snapshot(self, symbols: list[str], as_of: date | None, fields: list[str] | None) -> pd.DataFrame:
        ...

6.3 DataManager façade: get_ohlcv algorithm

1. Detect asset_type and provider priorities.


2. Split the range by years; read coverage from Redis/index.


3. For missing/stale fragments — call the first available provider with retries.


4. Normalize and write segments year.parquet.gz (atomic temp → rename).


5. Concatenate segments, sort, deduplicate, return DataFrame.



Incremental pseudocode:

# edge backfill for current year
last_ts = file_index.last_ts(year=now.year)
fetch_start = max(start, last_ts - 2 * tf_delta)  # safety rewind
raw = await provider.fetch(symbol, tf, fetch_start, end)
merged = merge_and_overwrite(file_segment, raw)
save_segment(year, merged)

6.4 DataManager: fundamentals flow with JSON cache

# for each provider in preference order and each symbol
snap = cache.find_latest_json(provider, symbol)
if snap and snap.as_of >= now - timedelta(days=7) and not force_refresh:
    use snap
else:
    snap = await provider.fetch_snapshot(symbol)
    cache.write_json(provider, symbol, snap)  # provider_symbol_timestamp.json

combined = combine_snapshots(all_snaps, strategy=config.combine.strategy)
return normalize_fields(combined)


---

7) Cache & Storage

7.1 File cache (Parquet + gzip for candles)

Tree: candles/{asset}/{symbol}/{tf}/{year}.parquet.gz.

Separate index.json metadata: {year: {first_ts, last_ts, rows, etag}}.


7.2 File cache (JSON for fundamentals)

Tree: fundamentals/{provider}/ containing files provider_symbol_timestamp.json.

index.json per provider with {symbol: [{timestamp, path, as_of}], ...} for quick lookup.

JSON schema includes provider, symbol, as_of, fields map, and raw payload if needed.


7.3 Redis cache

candles:{asset}:{symbol}:{tf}:meta -> {years: [...], hot_ttl, last_update}

candles:{asset}:{symbol}:{tf}:{year}:etag -> sha256

fund:latest:{provider}:{symbol} -> timestamp

TTL for hot windows: 24–72h. Metadata without TTL.



---

8) Data Quality & Normalization

Convert time to UTC; enforce TF alignment (minute starts at :00).

Remove non-numeric/NaN; optional outlier filter.

Gap detector for missing candles; default policy: do not fill; optional forward-fill for close.

Equities: store close & close_adj, dividends, splits when available; adjusted flag in segment metadata.

Fundamentals: provider field mapping → unified aliases; track as_of per record.



---

9) Configuration

Format: YAML + ENV. Example:


cache:
  base_dir: ./data_cache
  parquet_compression: gzip
  hot_edge_multiplier: 2
  refresh_days_current_year: 30
redis:
  url: redis://localhost:6379/2
providers:
  priority:
    crypto: [binance, coingecko]
    equity: [fmp, ibkr, alphavantage, yfinance]
  binance:
    api_key: ${BINANCE_KEY}
    api_secret: ${BINANCE_SECRET}
  fmp:
    api_key: ${FMP_KEY}
  alphavantage:
    api_key: ${ALPHAVANTAGE_KEY}
combine:
  strategy: priority  # priority | most_recent | suffix_conflicts
  field_priority:
    pe_ttm: [fmp, ibkr, alphavantage, yfinance]
    market_cap: [fmp, ibkr, yfinance, alphavantage]


---

10) REST API (optional, FastAPI)

GET /candles?symbol=BTCUSDT&tf=15m&start=2020-01-01&end=2020-12-31 → JSON/Parquet stream.

GET /fundamentals/snapshot?symbols=AAPL,MSFT&as_of=2025-06-30&fields=pe_ttm,market_cap

POST /refresh/candles (bulk warm-up).

Auth: API key (header), per-endpoint rate limiting.



---

11) Backtesting & Live integrations

Pandas feed: already returns standardized DataFrames.

Backtrader: BtParquetFeed / BtPandasLive:

History: load via DataManager.get_ohlcv → feed.

Live: stream_candles → queue/callback, update the last bar and emit ondata.


Extras: adapters for Zipline/Backtesting.py if needed.



---

12) Testing & Acceptance

12.1 Unit (pytest)

Providers: mocked responses, TF mapping, 429/5xx handling, retries.

File cache (candles): write/read segment, atomicity, indexing.

Incremental refresh: edge overwrite for current year, no duplicates.

Normalization: UTC, sorting, schema, resampling.

Fundamentals JSON cache: write/read provider_symbol_timestamp.json, 7‑day freshness logic, multi‑provider combine rules.


12.2 Integration

Provider cascade (fallback), mixed year ranges.

Redis: correct keys, coverage metadata, cache hit‑ratio > 0.7 on repeats.


12.3 Load

100 symbols × 5 years × 1m assembled under defined SLO (configurable).

Concurrency without exceeding provider limits.


12.4 Acceptance criteria

API & schema compliance.

Cache policy (current year refresh if last update > 30 days).

Zero loss/duplication of edge candles.

Docs and 90% coverage of critical paths.



---

13) Scheduler

Apscheduler/cron jobs:

Nightly warm-up of hot symbols (current year).

Weekly fundamentals snapshots.

Redis hot window/metrics cleanup.




---

14) Monitoring & Alerts

Prometheus exporter + Grafana dashboard: latency, errors, missing candles.

Telegram/Email alerts on high error rates, 429 storms, segment desync.



---

15) Deployment

Dockerfile (slim + poetry/pip), docker‑compose (web + worker + redis).

Environment variables for API keys.

Volume for file cache.



---

16) Risks & Assumptions

Provider limitations (licenses, volumes, retention, accuracy).

Intraday for equities on Yahoo is limited/unstable; for quality intraday consider paid sources.

Corporate actions: adjusted accuracy depends on provider.



---

17) Usage examples (Python API)

from datacore.managers.data_manager import DataManager
from datetime import datetime, timezone

mgr = DataManager.from_yaml("config.yml")

# History
df = mgr.get_ohlcv("BTCUSDT", "15m", start=datetime(2020,1,1,tzinfo=timezone.utc), end=datetime(2021,1,1,tzinfo=timezone.utc))

# Live stream
async def on_tick(bar):
    print(bar)
await mgr.stream_candles("AAPL", "1m", poll_interval=5.0, on_tick=on_tick)

# Fundamentals snapshot (combined from cache/providers)
snap = mgr.get_fundamentals_snapshot(["AAPL","MSFT"], fields=["market_cap","pe_ttm"], as_of=None)


---

18) Minimal class interfaces

class DataManager:
    @classmethod
    def from_yaml(cls, path: str) -> "DataManager": ...

    def get_ohlcv(self, symbol: str, timeframe: str, start, end, *, asset_type=None, adjusted=True, source_priority=None, force_refresh=False) -> pd.DataFrame: ...

    async def stream_candles(self, symbol: str, timeframe: str, *, poll_interval: float = 2.0, on_tick=None, backfill: bool = True): ...

    def get_fundamentals_snapshot(self, symbols: list[str], fields=None, as_of=None, provider_preference=None, force_refresh=False) -> pd.DataFrame: ...


---

19) Documentation

README: overview + quick start.

docs/: architecture, examples, provider reference, data schemas.

Diagrams: request/cache sequence, cache folder tree.



---

20) Possible extensions

Extra providers (Polygon, Finnhub, Tiingo, Bybit).

Corporate calendar (earnings, splits) and event-driven refresh.

Built-in alias unifier (AAPL.US / AAPL / NASDAQ:AAPL).



---


JSON fundamentals cache helper (find_latest_json, write_json, index maintenance)

A simple combine_snapshots(...) with pluggable strategies.


