# P17 Catalyst — 8-K Body-Text Improvement Plan

Status: **proposal** (no code yet)
Owner: P17 penny-stock screener
Related: `agents/catalyst_agent.py`, `src/data/downloader/edgar_downloader.py`,
`src/ml/pipeline/p15_hidden_deps/p15_daily.py`

---

## 1. Problem

The current `CatalystAgent` classifies 8-K filings using only the **structured
metadata** available from the EDGAR submissions feed:

- `items` (e.g. `1.01`, `7.01`, `8.01`) — structured but coarse.
- `primaryDocDescription` — almost always the literal string `"Form 8-K"`, so the
  keyword refinement rarely fires.

Consequently the agent effectively runs on **item codes alone**. Item `8.01
"Other Events"` may be an FDA approval, a buyback, a lawsuit, or a nothing-burger —
the code cannot distinguish them. We cannot judge **category**, **magnitude**, or
**direction/sentiment**.

The signal that disambiguates all of this — the actual press release — lives in
the 8-K's **Exhibit 99.1** (and is referenced by the primary 8-K document). To use
it we must fetch and parse the **filing body**, not just the metadata.

## 2. What already exists (reuse, don't rebuild)

| Capability | Where | Reuse for |
|---|---|---|
| Per-form **daily** EFTS download + per-day cache | `EdgarDownloader.download_form4_filings`, `download_13dg_filings` | Template for `download_8k_filings(date)` |
| EFTS search by form/date | `EdgarDownloader._efts_search(forms, start_dt, end_dt)` | 8-K daily index query |
| Rate-limited document fetch (0.5s interval, 503 back-off, retries) | `EdgarDownloader._fetch`, `_fetch_filing_xml` | Fetching 8-K body + exhibits |
| Per-CIK recent filings (item codes) | `EdgarDownloader.get_recent_filings` | What CatalystAgent uses today |
| **Daily** submissions refresh for ~160 watchlist CIKs ("used for 8-K event tracking") | `p15_daily._job_edgar_submissions` | Existing daily EDGAR cadence to extend |
| Ticker→CIK resolution | `EdgarDownloader.load_company_tickers`, `resolve_tickers_to_ciks` | Mapping candidates to filers |

**Answer to "can the P15 daily downloader be extended to get 8-K daily?": yes —
and it is the right home for the universe-wide daily acquisition.** P15 already
runs a daily EDGAR submissions job for a curated ~160-CIK watchlist. We add a new,
**universe-agnostic** daily job that pulls the *entire day's* 8-K filings once,
caches them, and (optionally) pre-fetches bodies. P17 (and the future intraday
monitor) then read from that cache instead of doing per-candidate EDGAR calls.

## 3. Target architecture

Three thin layers, each independently testable, built on the existing downloader:

```
Layer A — Daily 8-K index  (EdgarDownloader.download_8k_filings(date))
   EFTS forms="8-K,8-K/A" for one filing date
   → DATA_CACHE_DIR/edgar/8k/index/{YYYY-MM-DD}.csv.gz
   columns: cik, ticker, company, accession_number, items, filed_date, primary_document

Layer B — 8-K body fetch    (EdgarDownloader.fetch_8k_body(cik, accession, primary_doc))
   Fetch filing index.json → locate primary doc + Exhibit 99.1
   Strip HTML → plain text
   → DATA_CACHE_DIR/edgar/8k/body/{accession}.txt.gz   (cached, fetch-once)

Layer C — Catalyst classifier (catalyst classification of body text)
   v1: richer keyword/regex over the body  →  {category, score}
   v2: LLM (Claude, as P05 already uses)    →  {category, magnitude, sentiment, confidence}
```

### Data flow after the change

```
P15 daily bundle (13:00 UTC)
  └─ _job_edgar_8k_index(date)         # Layer A — one EFTS call, caches the day's 8-K index
        (optional) prefetch bodies      # Layer B — for filings matching bullish item prefilter

P17 daily run (06:00 UTC)
  └─ CatalystAgent.run(candidates)
        read cached 8-K index for last `lookback_days`   # no per-candidate EFTS calls
        for each candidate's recent bullish 8-K:
           fetch_8k_body (cache hit if P15 prefetched)   # Layer B
           classify(body)                                 # Layer C
        → catalyst_score, catalyst_signals
```

This **inverts** today's model (per-candidate `get_recent_filings`) into a
**read-from-daily-cache** model: one cheap universe-wide pull per day, then local
lookups. It also makes the data instantly reusable by the intraday monitor.

## 4. Phased implementation plan

### Phase 1 — Daily 8-K index (structured, no bodies) — *small*
- Add `EdgarDownloader.download_8k_filings(as_of_date, force)` mirroring
  `download_form4_filings` (EFTS `forms="8-K,8-K/A"`, per-day cache). Parse
  `items`, `cik`, `display_names`→ticker/company, `accession_no`, `file_date`,
  primary document from the EFTS `_source`.
- Add `_job_edgar_8k_index(date)` to `p15_daily.py` (self-healing gap-fill like the
  GDELT/TRF jobs, capped at `_GAP_CAP_DAYS`).
- Repoint `CatalystAgent` to read the cached daily index for its lookback window
  instead of per-CIK `get_recent_filings`. **Behaviour identical** to today (still
  item-code based) but universe-wide and far fewer requests.
- Tests: index parse, cache hit/miss, agent reads index.

### Phase 2 — Body fetch + keyword classification — *medium*
- Add `EdgarDownloader.fetch_8k_body(cik, accession, primary_document)`:
  fetch the filing `index.json`, pull primary doc + `EX-99.1`, HTML→text, cache.
- Add a body-aware classifier: expanded keyword/regex sets per catalyst category
  (FDA/clearance, contract/award, M&A, guidance, defense/nuclear/rare-earth, etc.),
  returning `(category, base_points)` with the body as input.
- **Gate the fetch**: only fetch bodies for 8-Ks that (a) passed the cheap
  item-code prefilter AND (b) belong to a candidate that survived earlier P17
  stages. Never fetch for the full universe.
- Optionally pre-fetch bodies in the P15 job for filings matching the bullish
  item prefilter, so P17 mostly hits cache.
- Tests: index.json exhibit resolution, HTML→text, keyword classification, fetch
  gating, cache reuse.

### Phase 3 — LLM classification — *medium, optional but highest precision*
- Add an optional classifier backend that sends the body text to Claude
  (reuse the P05 Anthropic plumbing) with a tool schema returning
  `{category, magnitude_bucket, sentiment, confidence}`.
- Map the structured result to `catalyst_score` (e.g. magnitude + sentiment scale
  the base tier points; negative sentiment → 0).
- Config flag `catalyst_classifier: "keyword" | "llm"` with keyword as default so
  there is no hard dependency / cost unless enabled.
- Cost is bounded: only a handful of *fresh, prefiltered* 8-Ks per day across the
  watched candidates. Cache classifications by accession so re-runs are free.
- Tests: mock the LLM client (as `test_stage3_llm.py` does), assert score mapping.

### Phase 4 — Consolidation & reuse — *small*
- Make the daily 8-K cache the single source for catalyst data across P17 **and**
  the intraday monitor (see `intraday_monitor_design.md`).
- Add a `Tasks.md` / README note documenting the cache layout and the
  keyword→LLM upgrade path.

## 5. Cache layout (new)

```
DATA_CACHE_DIR/edgar/8k/
    index/{YYYY-MM-DD}.csv.gz      # Layer A — one row per 8-K filed that day
    body/{accession}.txt.gz        # Layer B — extracted press-release text, fetch-once
    classified/{accession}.json    # Layer C — cached classification (category, score, …)
```

## 6. Risks & mitigations

- **EDGAR rate limits** — bodies are 1–2 requests per filing. Mitigate with the
  existing 0.5s throttle + 503 back-off, fetch-gating (prefiltered survivors only),
  and fetch-once caching. Prefetch in the off-peak P15 13:00 UTC window.
- **HTML/PR parsing messiness** — press releases vary; start with a tolerant
  HTML→text strip and keyword match; escalate to LLM for robustness.
- **LLM cost/latency** — gated + cached + opt-in via config; keyword default keeps
  the pipeline dependency-free.
- **Universe drift** — the daily index is universe-agnostic (everything filed that
  day), so P17's changing penny-stock set is covered without per-day reconfiguration.

## 7. Recommendation

Do **Phase 1 + Phase 2** first (daily index + body keyword classification): that
captures most of the precision gain with no new external dependency, and centralises
acquisition in P15. Treat **Phase 3 (LLM)** as an opt-in precision upgrade once the
plumbing is proven.
