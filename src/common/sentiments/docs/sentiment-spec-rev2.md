# Sentiment Check — Requirements & Detailed Design (Rev 2)

**Supersedes:** Rev 1
**Change summary:** Reddit/Pushshift path removed (no longer accessible). Two new adapters — Hacker News and Bluesky — with a revised aggregation model that treats them as *different kinds of signal* rather than interchangeable message sources.

---

## 0. What changed and why

| Rev 1 | Rev 2 | Reason |
|---|---|---|
| `async_pushshift_adapter` | **Removed** | Pushshift restricted to verified Reddit moderators since May 2023. No public or developer access. |
| Reddit via Data API | **Removed** | Self-service API registration closed under the Responsible Builder Policy; access requires manual approval, commonly denied. |
| — | `async_bluesky_adapter` | Substitute for the retail-chatter signal Reddit provided. Cashtag-searchable, open protocol, no approval process. |
| — | `async_hackernews_adapter` | New **`tech_discourse`** signal class. Not a retail-hype source — see §0.1. |
| Single blended `sentiment_score_24h` | Two signal classes: `retail_sentiment` + `tech_discourse` | Averaging structurally different sources destroys information. See §2.5.6. |
| Global keyword lexicon | Per-source lexicons | WSB slang has near-zero frequency on HN. A shared lexicon silently returns neutral for every HN message. |
| Single HF model | Per-source model routing | `twitter-roberta` is tuned for short social text and degrades on HN's long-form technical prose. |
| Raw author IDs stored | Salted-hash author IDs | Preserves `unique_authors` and bot detection without retaining handles. |

### 0.1 Set expectations on coverage

This is the most important thing to understand before building:

- **Bluesky** carries real finance chatter — part of FinTwit migrated there and cashtags are used — but at meaningfully lower volume than Reddit or Stocktwits ever provided. Expect `mentions_24h` in the single or low double digits for mid-cap tickers, and often **zero** for small caps. Your `min_mentions_for_confident_signal: 5` threshold will fire constantly.
- **Hacker News** has effectively **zero** ticker coverage. It will produce usable signal for perhaps 50–150 large tech names via *company-name* resolution (NVDA, TSLA, PLTR, COIN, MSFT…) and nothing at all for the small-cap squeeze candidates a FINRA short-interest pipeline typically surfaces.

**Implication:** for the squeeze-detection use case these two sources together are thinner than what you had. Build the adapters — they're free and the plumbing is sound — but instrument coverage from day one (§2.10) and decide based on measured `ticker_coverage_pct` whether the sentiment term still deserves its weight in `squeeze_score`. Do not tune thresholds until you have that number.

---

# 1. Requirements

## 1.1 Purpose

Unchanged: provide sentiment metrics per ticker to feed `run_daily_deep_scan.py` scoring and ad-hoc candidate logic.

## 1.2 Scope — revised output schema

Output per ticker, stored in `deep_metrics`:

**Retail sentiment (Bluesky, Stocktwits)**
* `mentions_24h` (int)
* `unique_authors_24h` (int)
* `mentions_growth_7d` (float)
* `positive_ratio_24h` (0..1)
* `sentiment_score_24h` (-1..+1), `sentiment_normalized` (0..1)
* `virality_index` (float) — **redefined, see §2.5.5**
* `bot_pct` (0..1)

**Tech discourse (Hacker News) — new, separate**
* `tech_mentions_24h` (int)
* `tech_sentiment_score_24h` (-1..+1), `tech_sentiment_normalized` (0..1)
* `tech_discussion_depth` (float) — mean comments per matched story
* `tech_coverage_available` (bool) — is this ticker in the entity map at all

**Meta**
* `data_quality` flags per provider
* `ticker_coverage` — which providers returned any data
* `sentiment_raw_payload` (JSON)

## 1.3 Non-functional requirements

* **Latency**: ≤ 120s for a batch of 50 tickers. *Note:* HN's fan-out pattern (§2.5.1) makes this harder than Rev 1 assumed. Mitigate with the shared-corpus strategy, not per-ticker fetching.
* **Resilience**: partial provider outage → best-effort result with `data_quality` marked.
* **Resource use**: HF inference optional, off by default, gated on `mentions_24h ≥ min_mentions_for_hf`.
* **Security**: no secrets in logs. Bluesky app password in env/secret store — **never the account password**.
* **Configurability**: all thresholds, providers, concurrency, models via `config.yml`.

## 1.4 Acceptance criteria — revised

* For a batch of 10 **large-cap tech** tickers, ≥ 95% return complete feature sets.
* For a batch of 10 **small-cap squeeze candidates**, the module returns valid low-confidence results with correct `data_quality` flags. *Zero mentions is a correct answer, not a failure* — assert that the module degrades cleanly rather than asserting coverage.
* `sentiment_raw_payload` stored and parseable.
* When `mentions_24h < min_mentions`, marked low-confidence, normalized to 0.5, still stored.
* Sync and async interfaces both exposed.
* **New:** a `coverage-report` command reports `ticker_coverage_pct` per provider across the full screener universe. Run this before tuning anything.

---

# 2. Detailed Design

## 2.1 High-level architecture

```
sentiment/
├── adapters/
│   ├── base.py              # AsyncAdapter ABC, RawMessage
│   ├── stocktwits.py        # existing — verify status, see §2.1.1
│   ├── bluesky.py           # NEW
│   └── hackernews.py        # NEW
├── entity/
│   ├── resolver.py          # NEW — ticker ↔ company/product name mapping
│   └── tickers.yml          # NEW — curated entity map
├── collector.py             # orchestration, concurrency, backoff
├── processor/
│   ├── lexicon.py           # per-source keyword lexicons
│   ├── hf.py                # AsyncHFSentiment, per-source model routing
│   ├── bots.py              # bot heuristics
│   └── aggregate.py         # features, calibration, blending
├── storage.py
└── config.py
```

**New invariant:** adapters never see ticker semantics beyond a search term. All ticker↔entity logic lives in `entity/`. This is what lets HN — which needs name-based resolution — use the same adapter contract as cashtag sources.

### 2.1.1 Stocktwits

Rev 1 assumed public Stocktwits access. **Verify current API terms and rate limits before relying on it**; access conditions have tightened across the industry and an unverified assumption here leaves you with Bluesky as the sole retail source. Record findings in the README.

---

## 2.2 Adapter contract

```python
@dataclass(frozen=True)
class RawMessage:
    provider: str            # "bluesky" | "hackernews" | "stocktwits"
    message_id: str          # unique within provider
    author_hash: str         # salted SHA-256 of native author ID — see §2.11
    created_utc: datetime    # tz-aware
    body: str                # cleaned plain text
    engagement: dict         # provider-native counts, normalized in §2.5.5
    lang: str | None
    url: str | None
    parent_id: str | None
    meta: dict

class AsyncAdapter(ABC):
    name: str
    signal_class: Literal["retail", "tech_discourse"]   # NEW — drives routing
    supports_cashtag: bool                              # NEW — HN is False

    async def fetch_summary(self, term: str, lookback_hours: int) -> dict: ...
    async def fetch_messages(self, term: str, lookback_hours: int,
                             limit: int) -> list[RawMessage]: ...
```

`signal_class` is what keeps HN out of the retail blend. `aggregate.py` routes on it; no adapter-name special-casing anywhere downstream.

---

## 2.3 Adapter: Bluesky

**Role:** primary retail-chatter source, replacing Reddit.

### Authentication

Authenticate. Do not rely on unauthenticated public access — `public.api.bsky.app` has returned 403 for search since mid-2026, and unauthenticated `searchPosts` returns 403 on any cursor-based pagination request even when the first page succeeds, producing an error that resembles rate limiting but isn't.

Create a free account, generate an **app password** (Settings → App Passwords), store as `BLUESKY_HANDLE` / `BLUESKY_APP_PASSWORD`. Use the official `atproto` SDK for session handling and refresh.

> **Verify against `https://docs.bsky.app` before implementing.** This is the highest-churn item in the spec.

### Endpoints

| Lexicon | Use |
|---|---|
| `app.bsky.feed.searchPosts` | Cashtag/keyword search. Params: `q`, `limit` (max 100), `sort` (`top`\|`latest`), `since`, `until`, `cursor`, `lang` |
| `app.bsky.feed.getPostThread` | Reply threads for high-engagement posts |

### Query construction

Search **both** `$TICKER` and the company name, then union and dedupe on `uri`. Cashtag-only search badly undercounts on Bluesky, where convention is less established than it was on Twitter.

Guard against ticker/word collisions — `$ALL`, `$IT`, `$ON`, `$NOW`, `$GOOD`, `$LOVE`, `$CAR` are real tickers and common words. Maintain an `ambiguous_tickers` list in `entity/tickers.yml`; for these, **require** the `$` prefix and drop bare-name matches.

### Mapping

- `message_id` = post AT-URI
- `author_hash` = hash of author DID (§2.11)
- `body` = `record.text`
- `created_utc` = `record.createdAt` (ISO 8601 UTC)
- `engagement` = `{likes, reposts, replies}`
- `lang` = from `record.langs`; **pass `lang=en` and additionally drop posts whose `langs` excludes English.** English-tuned models score foreign text confidently and wrongly.

### Pagination

Prefer `cursor` when authenticated. Implement a time-window fallback: `sort=latest`, set `until` to the `createdAt` of the previous page's last post, dedupe on `uri`. **Guard it** — if a page yields no new URIs, stop, or you will loop on a timestamp boundary.

---

## 2.4 Adapter: Hacker News

**Role:** `tech_discourse` signal. Not retail hype. Read §0.1 before building.

### Why it's structurally different

HN discussion of a company is *analytical* — outage post-mortems, product critique, hiring threads, technical comparison. Its sentiment correlates with engineering reputation, not with retail buying pressure. That is a genuinely useful covariate for tech names, and a lead indicator for some kinds of news, but it is **not** a squeeze predictor and must not be blended as one.

### API — no auth, no key

Base: `https://hacker-news.firebaseio.com/v0/`

| Endpoint | Returns |
|---|---|
| `newstories.json` | Up to 500 newest story IDs |
| `topstories.json` | Up to 500 front-page story IDs |
| `item/{id}.json` | Single item — story, comment, job, poll |
| `maxitem.json` | Largest current item ID |

Officially no rate limit; impose one anyway (default 10 req/s). `item.time` is Unix seconds. `item.kids` is an array of **child IDs only** — a 200-comment thread is 200 requests.

### Critical: shared-corpus fetch strategy

Do **not** fetch per ticker. With 50 tickers that's 50× the fan-out and blows the 120s budget.

Instead, once per batch run:

1. Fetch the last N hours of stories from `newstories.json` + `topstories.json` (union, dedupe).
2. Fetch each story's comment tree once, bounded by `max_depth` and `max_items_per_thread`.
3. Cache the entire corpus in SQLite keyed on item ID.
4. Run **every** ticker's entity match against that single in-memory corpus.

Cost becomes O(corpus), independent of batch size. Cache aggressively — re-running a scan must not re-fetch known items.

### Text cleaning — highest-risk code path

HN `text` is **HTML**: `<p>`, `<i>`, `<a href>`, `<pre><code>`, and entities like `&#x27;`, `&gt;`, `&quot;`. Strip tags and unescape entities before scoring. Preserve paragraph breaks as `\n\n`.

**Strip `<pre><code>` blocks entirely before sentiment scoring.** Source code scores as strongly negative on most classifiers (`fail`, `error`, `abort`, `kill`, `dead`) and HN comments are full of it. This single omission will systematically bias every tech ticker negative, and the bias is invisible in aggregate output.

Skip items with `deleted: true` or `dead: true`.

### Entity resolution — `entity/resolver.py`

HN never says `$NVDA`. Match on curated aliases:

```yaml
NVDA:
  names: ["nvidia"]
  products: ["cuda", "h100", "b200", "geforce", "rtx"]
  ambiguous: false
COIN:
  names: ["coinbase"]
  products: ["base l2"]
  ambiguous: true      # "coin" alone is far too generic — require full name
```

Rules:
- Case-insensitive whole-word matching. Never substring — `ON` inside "on" or `IT` inside "it" will match everything.
- A message matching ≥2 distinct tickers counts toward both, flagged `multi_entity: true` in meta. Comparison threads are legitimate signal for both names.
- `tech_coverage_available = ticker in entity_map`. Tickers absent from the map return `tech_*` fields as `None`, **not** 0.5. Distinguish "no data" from "neutral" — conflating them poisons any downstream average.

Seed the map with the 100–150 largest listed tech companies. Accept that coverage stops there.

### Engagement mapping

HN comments have **no per-comment score** in the public API. Only stories carry `score`. Map:
- Story: `engagement = {score, descendants}`
- Comment: `engagement = {parent_story_score, reply_count: len(kids)}`

`sqrt(engagement+1)` from Rev 1 does not transfer — a comment's weight must derive from its thread's prominence, not from a like count that doesn't exist.

---

## 2.5 Algorithms

### 2.5.1 Collection

Per batch:
1. **Once:** build HN corpus (§2.4). Independent of ticker list.
2. **Per ticker, concurrent:** Bluesky `searchPosts`, Stocktwits summary.
3. Gate HF inference on `mentions_24h ≥ min_mentions_for_hf`.
4. Entity-match the HN corpus against all tickers in-process.

### 2.5.2 Preprocessing

Unicode-normalize, lowercase for matching (preserve original case for HF — casing carries signal for transformer models). Canonicalize ticker mentions. Dedupe on `(body_hash, author_hash)`.

Bot heuristics — **per source, not global:**
- Bluesky: >20 posts/day on one ticker → likely bot; account age <2 days AND >5 posts → suspicious.
- HN: **skip bot detection.** HN is heavily human-moderated with negligible automated posting; applying Bluesky's thresholds there will misflag prolific legitimate commenters. Set `bot_pct = None` for HN, not 0.0.

### 2.5.3 Per-source lexicons — `processor/lexicon.py`

Rev 1's single global lexicon is the design's biggest silent failure mode. Split it:

```yaml
lexicons:
  retail:          # Bluesky, Stocktwits
    positive: ['moon','🚀','diamond','buy','long','hold','squeeze','breakout']
    negative: ['short','sell','dump','bankrupt','bagholder','paper hands','rug']
  tech_discourse:  # Hacker News
    positive: ['impressive','solid','elegant','ships','reliable','well-designed']
    negative: ['broken','regression','outage','vendor lock-in','enshittification',
               'layoffs','abandoned','deprecated']
```

Note these encode different constructs. Retail measures *hype*; tech_discourse measures *engineering reputation*. Do not compare their scores directly.

Beware domain-neutral vocabulary: on HN, `bug`, `crash`, `fails`, `broken` frequently appear in neutral technical description. This is exactly why the HF model matters more here than the lexicon.

### 2.5.4 HF model routing

Per-source model selection:

```yaml
hf:
  models:
    retail: cardiffnlp/twitter-roberta-base-sentiment-latest
    tech_discourse: distilbert-base-uncased-finetuned-sst-2-english
```

`twitter-roberta` is tuned for short informal posts and degrades on HN's long-form prose. Verify both identifiers resolve on HuggingFace before depending on them.

**Long text:** HN comments routinely exceed 512 tokens. Do not silently truncate. Implement explicit `long_text_strategy: truncate | chunk_mean | chunk_max` and record the choice in `raw_payload`.

Label mapping stays configurable per model — `LABEL_0/1/2` conventions differ between checkpoints and a wrong mapping silently inverts your signal.

### 2.5.5 Feature computation

`virality_index` — **redefined.** Rev 1's `Σ(engagement * polarity) / sqrt(unique_authors)` conflates reach with direction: a viral *negative* post and a quiet positive one can produce the same value. Split:

```
virality_index    = Σ(engagement) / sqrt(unique_authors + 1)     # reach only, unsigned
sentiment_score   = Σ(polarity * weight) / Σ(weight)             # direction only
```

Keep them orthogonal; let `squeeze_score` combine them if it wants to.

Message weight, source-aware:
```
weight = sqrt(normalized_engagement + 1) * author_trust
author_trust: 0.5..1.0, suspected bots 0.2, HN uniformly 1.0
```

`normalized_engagement` must be a per-source percentile rank, not a raw count. An HN story score of 300 and a Bluesky like count of 300 are not comparable quantities.

`mentions_growth_7d = mentions_24h / (avg_mentions_prev_7d + eps)` — **per source.** Your historical Reddit baselines are unrecoverable; both new sources start with no history. Expect this feature to be unusable for the first 7 days and return `None` (not 1.0) until sufficient history exists.

### 2.5.6 Cross-source aggregation — the core change

Raw scores from different platforms are not on a common scale. Bluesky finance posts skew promotional-positive; HN skews critical-negative. Blending raw values imports that platform bias directly into `squeeze_score`.

**Calibrate before blending.** Per source, maintain a trailing 30-day distribution of `sentiment_score` across all tickers, and convert each ticker's raw score to a z-score against it:

```
calibrated = (raw_score - source_mean_30d) / (source_std_30d + eps)
```

This answers the question that actually matters — *is this ticker unusually positive **for this platform**?* — rather than the meaningless cross-platform comparison. Fall back to raw scores with `data_quality.calibration: "insufficient_history"` until 30 days accumulate.

**Then blend within signal class only:**

```yaml
blend_weights:
  retail:
    bluesky: 0.6
    stocktwits: 0.4
  # tech_discourse is NOT blended into retail — reported separately
```

Renormalize weights across providers that actually returned data, so a Stocktwits outage doesn't silently halve the score.

`sentiment_24h` (consumed by `squeeze_score`) derives from the **retail** class only. `tech_sentiment_24h` is exposed as an independent feature. If you later find it predictive, add it to `squeeze_score` with its own weight — but let the backtest decide that, not the aggregation layer.

---

## 2.6 Config schema

```yaml
sentiment:
  providers:
    stocktwits: true
    bluesky: true
    hackernews: true
  lookback_hours: 24
  min_mentions_for_hf: 20
  min_mentions_for_confident_signal: 5

  bluesky:
    handle: ${BLUESKY_HANDLE}
    app_password: ${BLUESKY_APP_PASSWORD}
    search_terms: [cashtag, company_name]
    lang_filter: en
    max_posts_per_ticker: 200
    rate_limit_rps: 5

  hackernews:
    corpus_sources: [newstories, topstories]
    corpus_lookback_hours: 48
    max_depth: 4
    max_items_per_thread: 300
    strip_code_blocks: true
    rate_limit_rps: 10
    entity_map_path: entity/tickers.yml

  hf:
    enabled: false
    models:
      retail: cardiffnlp/twitter-roberta-base-sentiment-latest
      tech_discourse: distilbert-base-uncased-finetuned-sst-2-english
    long_text_strategy: chunk_mean
    device: -1
    max_workers: 1

  calibration:
    enabled: true
    window_days: 30
    min_observations: 200

  blend_weights:
    retail: { bluesky: 0.6, stocktwits: 0.4 }

  batching:
    concurrency: 8
    rate_limit_delay_sec: 0.3
  caching:
    ttl_seconds: 900
    hn_corpus_ttl_seconds: 1800
```

---

## 2.7 DB schema additions

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
  -- new: tech discourse class
  ADD COLUMN tech_mentions_24h INTEGER,
  ADD COLUMN tech_sentiment_score_24h REAL,
  ADD COLUMN tech_sentiment_24h REAL,
  ADD COLUMN tech_discussion_depth REAL,
  ADD COLUMN tech_coverage_available INTEGER,
  ADD COLUMN sentiment_raw_payload JSONB,
  ADD COLUMN sentiment_data_quality JSONB;

-- new: HN corpus cache, shared across tickers
CREATE TABLE hn_corpus (
    item_id     INTEGER PRIMARY KEY,
    item_type   TEXT NOT NULL,
    parent_id   INTEGER,
    story_id    INTEGER,
    author_hash TEXT,
    created_utc TEXT NOT NULL,
    text_clean  TEXT,
    score       INTEGER,
    fetched_at  TEXT NOT NULL
);
CREATE INDEX idx_hn_created ON hn_corpus(created_utc);

-- new: per-source calibration history
CREATE TABLE sentiment_calibration (
    provider    TEXT NOT NULL,
    day         TEXT NOT NULL,
    mean_score  REAL NOT NULL,
    std_score   REAL NOT NULL,
    n_obs       INTEGER NOT NULL,
    PRIMARY KEY (provider, day)
);
```

SQLite: JSONB → TEXT.

**Migration note:** existing rows have Reddit-derived `mentions_growth_7d` values computed against a source that no longer exists. Null them rather than letting them mix with new-source baselines.

---

## 2.8 Error handling

* All providers fail → `data_quality: all_missing`, `sentiment_24h = 0.5`, `sentiment_score_24h = 0`, WARN, continue.
* HF fails → lexicon-only polarity, `hf: failed`.
* Partial responses → renormalize blend weights across responding providers (§2.5.6).
* **Bluesky auth failure** → distinguish 401 (bad credentials, fail loudly at startup) from 403 on pagination (known behavior, fall back to the time-window strategy). Do not treat 403 as rate limiting.
* **HN corpus fetch fails** → all `tech_*` fields `None`, `data_quality.hackernews: missing`. Never fall back to 0.5 — that would inject a fabricated neutral reading.
* Rate limits → semaphore concurrency + exponential backoff with jitter on 429/5xx.

---

## 2.9 Testing

**Unit**
- Adapter mocks per provider: empty, partial, malformed, 429, 403.
- **HTML cleaning suite for HN** — `&#x27;`, `&gt;`, nested tags, `<pre><code>` stripping, empty text. Highest-value tests in the project.
- **Entity resolver:** whole-word matching, ambiguous-ticker rejection (`$ALL`, `$IT`, `$ON`), multi-entity messages, case insensitivity. Assert `ON` does not match the word "on".
- Calibration math against a synthetic distribution with known mean/std.
- Blend-weight renormalization when a provider is absent.

**Integration**
- End-to-end for `["NVDA", "GME"]` with mocked adapters — NVDA exercises the HN path, GME exercises retail-only with `tech_coverage_available: false`.
- Assert `tech_*` fields are `None` (not 0.5) for uncovered tickers.

**Performance**
- Batches of [10, 25, 50] against a fixture HN corpus. Assert HN cost is flat across batch sizes — if it scales with ticker count, the shared-corpus strategy is broken.

**Backtest**
- Replay against known episodes and measure lead time and precision. Run this **separately per signal class** — the whole point of splitting them is to find out whether `tech_discourse` carries independent predictive value. Blending first makes that unanswerable.

---

## 2.10 Observability

Rev 1 metrics, plus:

* `sentiment.coverage.tickers_with_zero_mentions{provider}` — **watch this one first**
* `sentiment.hn.corpus_size`, `sentiment.hn.entity_match_rate`
* `sentiment.bluesky.auth_refresh_count`, `sentiment.bluesky.pagination_fallback_count`
* `sentiment.calibration.status{provider}`
* `sentiment.blend.providers_available`

**New CLI:** `coverage-report --universe screener_snapshot --days 7` → per-provider `ticker_coverage_pct`, median `mentions_24h`, and count of tickers below `min_mentions`. Run before tuning. If Bluesky coverage on your actual candidate universe is under ~20%, the honest conclusion is that the sentiment term should carry little or no weight in `squeeze_score` — and knowing that is worth more than a feature that looks populated but is mostly imputed neutrals.

---

## 2.11 Security & privacy

* Bluesky **app password** only, never account password. Env vars or secret store.
* **Author IDs are salted-hashed at the adapter boundary** — `SHA256(salt + native_id)`, salt from env, never logged. Preserves `unique_authors_24h` and bot detection without retaining handles or DIDs.
* Do not log message bodies or author identifiers.
* `raw_payload` retained for audit under access control. Add a retention policy — default 90 days — and a purge job. Indefinite retention of third-party social content has no upside for a signal pipeline with a 7-day feature horizon.
* No inference of personal characteristics about individual authors. All output is aggregate and ticker-level.

---

## 2.12 Deployment

Dependencies: `aiohttp`/`httpx`, `atproto`, `transformers`, `torch` (or `onnxruntime`), `aiosqlite`/`asyncpg`.

HN corpus fetching is I/O-bound with high fan-out — size the connection pool and semaphore for it explicitly rather than sharing the default batch concurrency.

---

## 2.13 Integration points

**`run_daily_deep_scan.py`** — unchanged call signature:
```python
sentiment_map = await collect_sentiment_batch(batch_tickers, ...)
candidate.transient_metrics.sentiment_24h = sentiment_map[t].sentiment_normalized
candidate.transient_metrics.tech_sentiment_24h = sentiment_map[t].tech_sentiment_normalized  # may be None
```
Scoring must handle `None` for `tech_*` explicitly. Do not default it to 0.5.

**`run_volume_detector.py`** — quick check with `lookback_hours=3`. Note that at 3h lookback Bluesky mention counts will frequently be 0 for anything but mega-caps; treat the result as a veto signal at most, not a promotion signal.

**`manage_adhoc_candidates.py`**, **`run_weekly_screener.py`**, **`run_finra_collector.py`** — unchanged.

---

# Appendix — Implementation checklist

1. Delete `async_pushshift_adapter` and its config/tests. Null Reddit-derived `mentions_growth_7d`.
2. Verify Stocktwits API status (§2.1.1). Record findings.
3. Implement `AsyncAdapter` ABC + `RawMessage` with `signal_class`.
4. Build `entity/tickers.yml` for the top ~100 tech names, with `ambiguous` flags.
5. Implement `hackernews.py` with shared-corpus fetch + HTML cleaning + code-block stripping.
6. Implement `bluesky.py` — verify docs first, authenticate via app password.
7. Split lexicons per source.
8. Add per-source HF model routing and long-text strategy.
9. Implement calibration table and z-score blending.
10. DB migration.
11. **Run `coverage-report` on your real candidate universe. Decide the sentiment weight from that number before tuning anything else.**
12. Tests, metrics, staging smoke test, backtest per signal class.

---

# Appendix B — Verify before implementing

1. Bluesky `searchPosts` auth requirements and pagination behavior — `https://docs.bsky.app`. **Highest risk.**
2. Stocktwits public API terms and rate limits.
3. HuggingFace model identifiers in §2.5.4.
4. HN Firebase API (`https://github.com/HackerNews/API`) — stable for a decade, safe to build against directly.
