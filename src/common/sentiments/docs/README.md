# Sentiment Check — Requirements & Detailed Design

Below are two clearly separated artifacts you asked for:

1. **Requirements** — concise, testable requirements for the Sentiment Check module and how it must behave inside your pipeline.
2. **Detailed Design** — implementation-level design: interfaces, data model, async flow, algorithms, config, error handling, testing, monitoring and integration points with the five scripts you listed.

---

# 1. Requirements (what the Sentiment Check module must deliver)

## 1.1 Purpose

Provide robust, production-ready sentiment metrics per ticker (transient/short-term) to feed the `run_daily_deep_scan.py` scoring engine and the ad-hoc candidate logic.

## 1.2 Scope

* Input: a list of tickers (batch), time window (default 24h), optional filters (min_mentions, language).
* Output: for each ticker produce a **sentiment feature set** stored in `deep_metrics.sentiment_24h` (and raw payload saved):

  * `mentions_24h` (int)
  * `unique_authors_24h` (int)
  * `mentions_growth_7d` (float)
  * `positive_ratio_24h` (0..1)
  * `sentiment_score_24h` (-1..+1) **and** `sentiment_normalized` (0..1)
  * `virality_index` (float)
  * `bot_pct` (0..1)
  * `data_quality` flags / provider availability
  * `sentiment_raw_payload` (JSON) — persisted for audit/retraining

## 1.3 Non-functional requirements

* **Latency**: should collect sentiment for a batch of up to 50 tickers within **≤ 120 seconds** under normal network conditions (configurable concurrency).
* **Throughput**: support typical deep-scan `batch_size` (default 10); can scale to 50 per batch with higher concurrency.
* **Resilience**: tolerate partial provider outages (mark `data_quality` and return best-effort results).
* **Resource use**: HF inference must be optional and off by default; heavy inference only for tickers with `mentions_24h ≥ X` (configurable, default X=20).
* **Security**: no secrets in logs; API keys only in env vars/secret store.
* **Configurability**: all thresholds, providers, concurrency, and HF model path via `config.yml` / `config.json`.

## 1.4 Acceptance criteria

* For a representative batch (10 tickers), the module returns complete feature sets for ≥ 95% tickers when public providers are available.
* `sentiment_raw_payload` for each ticker is stored and parseable JSON.
* When `mentions_24h < min_mentions` (default 5), `sentiment_score_24h` is marked as low-confidence and normalized to 0.5 (neutral) but still stored.
* Module must expose both **sync-compatible** and **async** interfaces so it can be used by `run_daily_deep_scan.py` (now async) or other scripts.

---

# 2. Detailed Design (how to implement it)

> Design goals: modular, async-first, pluggable providers, light-by-default, HF-enhanced optional, auditable.

---

## 2.1 High-level architecture

* **Adapters layer** (async): one adapter per provider:

  * `async_stocktwits_adapter` (public Stocktwits)
  * `async_pushshift_adapter` (Pushshift/Reddit)
  * (optional later) `async_twitter_adapter`, `finnhub_sentiment_adapter`, `google_trends_adapter`
* **Collector** (async): orchestrates parallel adapter calls per batch, does rate-limit/backoff, returns normalized message lists and lightweight summaries.
* **Processor / Scorer** (sync or async): takes messages/summary and computes features (keywords heuristics + HF model if enabled).
* **HF Inference** (async-friendly wrapper): runs HF pipeline in threadpool/external service.
* **Storage**: writes features + raw_payload to `deep_metrics` (or separate `sentiment_payloads`) via DB layer. For async runner use `aiosqlite` or `asyncpg`.
* **Interface**: public API `collect_sentiment_batch(tickers, lookback_hours, config) -> Dict[ticker] -> SentimentFeatures` usable by `run_daily_deep_scan` (async).

---

## 2.2 Data model (per ticker output)

Example JSON / DB columns to store in `deep_metrics`:

```yaml
sentiment_24h: float   # normalized 0..1 (use in squeeze_score)
sentiment_score_24h: float  # -1..1 raw
mentions_24h: int
unique_authors_24h: int
mentions_growth_7d: float
positive_ratio_24h: float  # 0..1
virality_index: float
bot_pct: float  # 0..1
data_quality: { stocktwits: ok|missing|partial, reddit: ok|missing, hf: ok|disabled }
sentiment_raw_payload: JSONB
```

* `squeeze_score` uses `sentiment_24h` (0..1). Keep `sentiment_score_24h` for audit.

---

## 2.3 Config schema (add to existing YAML)

```yaml
sentiment:
  providers:
    stocktwits: true
    reddit_pushshift: true
    finnhub_news: false
    google_trends: false
  lookback_hours: 24
  min_mentions_for_hf: 20
  min_mentions_for_confident_signal: 5
  hf:
    enabled: false
    model_name: cardiffnlp/twitter-roberta-base-sentiment
    device: -1          # -1 CPU, >=0 GPU index
    max_workers: 1
  batching:
    concurrency: 8
    rate_limit_delay_sec: 0.3
  thresholds:
    positive_ratio_high: 0.6
    positive_ratio_medium: 0.5
    virality_percentile: 90
  caching:
    ttl_seconds: 900  # 15m
```

---

## 2.4 Interfaces (function signatures)

### Async (primary)

```python
# returns dict[ticker] -> SentimentFeatures (dataclass / dict)
async def collect_sentiment_batch(
    tickers: List[str],
    lookback_hours: int = 24,
    concurrency: int = None,
    config: Optional[Dict] = None
) -> Dict[str, Optional[SentimentFeatures]]:
    ...
```

### Sync wrapper (for scripts that still call sync)

```python
def collect_sentiment_batch_sync(*args, **kwargs):
    return asyncio.run(collect_sentiment_batch(*args, **kwargs))
```

### SentimentFeatures dataclass (Python)

```python
@dataclass
class SentimentFeatures:
    ticker: str
    mentions_24h: int
    unique_authors_24h: int
    mentions_growth_7d: float
    positive_ratio_24h: float
    sentiment_score_24h: float
    sentiment_normalized: float
    virality_index: float
    bot_pct: float
    data_quality: dict
    raw_payload: dict
```

---

## 2.5 Algorithms & heuristics

### 2.5.1 Message collection

* For each ticker, call provider summaries (cheap) first:

  * `stocktwits.fetch_summary`, `pushshift.fetch_mentions_summary`.
* If `mentions >= min_mentions_for_hf`, fetch message bodies for HF:

  * `stocktwits.fetch_messages(limit=200)`, `pushshift.fetch_submissions+comments`.
* Each provider message normalized to: `{id, author_id, created_utc, body, likes, replies, retweets}`.

### 2.5.2 Preprocessing

* Normalize text (lowercase, unicode normalizer).
* Replace ticker mentions (`$GME`, `GME`, `gamestop`) to canonical ticker.
* Deduplicate identical messages (same body + same author).
* Bot heuristics:

  * Author posting rate > threshold (e.g., > 20 messages/day) → mark as likely bot.
  * Accounts <= 2 days old AND > 5 posts → suspicious.
* Engagement extraction: weighted engagement = likes + replies*2 + retweets*1.5 (configurable).

### 2.5.3 Heuristic sentiment baseline

* Keyword lists (short curated list, configurable):

  * positive: `['moon','🚀','diamond','buy','long','hold','to the moon']`
  * negative: `['short','sell','dump','bankrupt','bagholder','paper hands']`
* For each message compute `heuristic_polarity = +1/0/-1` for presence of positive/negative tokens. Emojis included.

### 2.5.4 HF model enhancement (optional)

* Use `AsyncHFSentiment.predict_batch(texts)`; output per-message label mapping to polarity:

  * If model returns `LABEL_0/LABEL_1/LABEL_2` map to neg/neu/pos. Make mapping configurable per `model_name`.
* Compute `hf_sentiment_score = (pos_count - neg_count)/total`.
* Final message polarity = weighted average: `0.4*heuristic + 0.6*hf` (configurable).

### 2.5.5 Aggregation -> features

* `mentions_24h`: count of non-duplicate messages.
* `unique_authors_24h`: distinct author ids.
* `positive_ratio_24h`: (# messages with polarity > 0) / mentions.
* `sentiment_score_24h`: average message polarity weighted by engagement and author credibility:

  * message_weight = sqrt(engagement + 1) * author_trust (author_trust = 0.5..1.0, bots get 0.2)
  * `sentiment_score = Σ(polarity * message_weight) / Σ(message_weight)`
* `sentiment_normalized = (sentiment_score + 1)/2`
* `mentions_growth_7d = mentions_24h / (avg_mentions_prev_7d + eps)` where `prev_7d` pulled from DB or cached counts.
* `virality_index = Σ(engagement * polarity) / sqrt(unique_authors)` — helps detect single-viral-post patterns.
* `bot_pct = number_of_messages_from_suspected_bots / mentions_24h`.
* `data_quality`: e.g., `{'stocktwits': 'ok', 'reddit': 'missing', 'hf': 'disabled'}`.

---

## 2.6 Integration points with your pipeline

### run_weekly_screener.py

* No changes; screener persists `screener_snapshot`. Sentiment module uses outputs of screener (for deep-scan candidates) and historical mentions for `mentions_growth_7d`.

### run_finra_collector.py

* No changes. Deep-scan merges FINRA SI with sentiment features.

### run_volume_detector.py

* If `volume_detector` finds ad-hoc candidates, it should call Sentiment Module for a quick check before promoting to ad-hoc list (example: call `collect_sentiment_batch_sync([ticker], lookback_hours=3)` to get near-real-time sentiment).

### run_daily_deep_scan.py (primary integration)

* For each batch of candidates (batch_size from config), call:

```python
sentiment_map = await collect_sentiment_batch(batch_tickers, lookback_hours=config.sentiment.lookback_hours, concurrency=config.sentiment.batching.concurrency)
```

* Attach `sentiment_map[ticker]` to candidate's transient metrics (e.g., `candidate.transient_metrics.sentiment_24h = sentiment_map[ticker].sentiment_normalized`).
* Use `sentiment_map[ticker].raw_payload` to persist in `deep_metrics.raw_payload`.

### manage_adhoc_candidates.py

* When an ad-hoc ticker is added manually, run an immediate quick sentiment check and store initial sentiment snapshot.

---

## 2.7 DB schema additions

Add these fields to `deep_metrics` table (or a dedicated `sentiment_payloads` table with FK to deep_metrics):

```sql
ALTER TABLE deep_metrics
  ADD COLUMN sentiment_score_24h REAL,
  ADD COLUMN sentiment_24h REAL,
  ADD COLUMN mentions_24h INTEGER,
  ADD COLUMN unique_authors_24h INTEGER,
  ADD COLUMN mentions_growth_7d REAL,
  ADD COLUMN positive_ratio_24h REAL,
  ADD COLUMN virality_index REAL,
  ADD COLUMN bot_pct REAL,
  ADD COLUMN sentiment_raw_payload JSONB,
  ADD COLUMN sentiment_data_quality JSONB;
```

If using SQLite, use TEXT to store JSON strings.

---

## 2.8 Error handling & fallbacks

* If all providers fail for a ticker:

  * Set `data_quality` = all_missing and `sentiment_24h` = 0.5 (neutral), `sentiment_score_24h` = 0.
  * Log a WARN and continue.
* If HF is enabled but inference fails:

  * Use heuristic-only polarity, mark `hf: failed` in `data_quality`.
* On partial provider responses:

  * Use available providers with weights (configurable). Example default weights: `reddit:0.6, stocktwits:0.4`.
* Rate-limit handling:

  * Adapters implement semaphore-based concurrency + sleep/jitter on 429.
  * Collectors use exponential backoff (configurable attempts).

---

## 2.9 Testing strategy

### Unit tests

* Adapter mocks: verify `fetch_summary` and `fetch_messages` handle empty responses, partial responses, error codes.
* Processor tests: feed crafted messages and assert computed `sentiment_score` / `virality_index` / `positive_ratio` match expected.
* HF wrapper: mock HF pipeline to return deterministic labels and test aggregation.

### Integration tests

* End-to-end test: run `collect_sentiment_batch` for a small set (`["GME","AMC"]`) with mocked adapter responses to validate DB write, returned features, and integration with `run_daily_deep_scan` (dry-run).

### Performance tests

* Measure time to process batch sizes [10, 25, 50] with concurrency config; assert targets met (e.g., ≤120s for 50 tickers under normal network).

### Backtest & validation

* Replay historical Reddit/Stocktwits dumps for known squeeze episodes; measure:

  * Lead time: how many days before price move did `mentions_growth_7d` spike?
  * Precision/recall for sentiment-based alerts.

---

## 2.10 Observability & metrics

Emit metrics (Prometheus or logs):

* `sentiment.batch.duration_seconds`
* `sentiment.tickers_processed`
* `sentiment.provider_calls_{stocktwits|reddit}`
* `sentiment.hf.inference_count`
* `sentiment.errors` (per provider)
* `sentiment.data_quality_skipped_count`

Log per-run summary:

* number of tickers with `mentions >= min`, number of HF inferences run, number of tickers with missing providers.

---

## 2.11 Security & privacy

* Keep API keys in env vars (e.g., `STOCKTWITS_API_KEY` if private).
* Do not log message bodies or author IDs to general logs — only persist raw payload to DB for audit (with access controls).
* If you store user-identifiable data (author handles), treat DB as restricted.

---

## 2.12 Deployment notes

* Add required dependencies to `requirements.txt` / poetry:

  * `aiohttp`, `transformers`, `torch` (or `onnxruntime` if using ONNX), `tqdm` (optional), DB async client (`aiosqlite` / `asyncpg`).
* For HF inference in production, prefer:

  * small distil models, or
  * serve model in a separate inference microservice (FastAPI) and call from sentiment module (keeps deep-scan memory small).
* Docker:

  * Add optional GPU support if using CUDA inference.

---

## 2.13 Example flow (summary)

1. `run_daily_deep_scan` prepares a batch of tickers.
2. `collect_sentiment_batch` is awaited; it:

   * calls `async_stocktwits.fetch_summary` and `async_pushshift.fetch_mentions_summary` for each ticker concurrently,
   * if `mentions >= min_for_hf` fetches messages and calls HF inference via `AsyncHFSentiment`,
   * aggregates messages into features and returns `SentimentFeatures` per ticker.
3. `run_daily_deep_scan` attaches the features to candidate transient metrics and persists to DB with `raw_payload`.
4. Scoring engine uses `sentiment_24h` (0..1) in `squeeze_score`.
5. Alerts may be triggered depending on thresholds.

---

# Appendix — Implementation checklist (practical)

1. Create async adapters (if not already): `async_stocktwits`, `async_pushshift`.
2. Implement `SentimentFeatures` dataclass and helper normalizers.
3. Implement `collect_sentiment_batch` async function with concurrency config and HF optionality.
4. Add DB schema changes and migration script.
5. Integrate calls into `run_daily_deep_scan` before per-candidate scoring in batch loop.
6. Add config parameters to `config.yml`.
7. Add unit tests, integration tests, performance tests.
8. Add Prometheus metrics or structured log events.
9. Deploy to staging, run smoke tests on a few tickers (GME, AMC).
10. Tune thresholds and min_mentions based on actual data and backtests.
