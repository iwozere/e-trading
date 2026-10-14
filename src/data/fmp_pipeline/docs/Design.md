# Design

## Purpose
Turn a time-boxed FMP Premium subscription into a permanent, reproducible, survivorship-free price and
fundamentals store, read through the existing `DataManager` without changing any of its 147 callers. The base
store holds raw prices plus event tables; every adjusted view is computed at read time (spec D3).

## Architecture

### High-Level Architecture
Target layout (spec §13), with the parts that exist today marked:

```
src/data/fmp_pipeline/
  cli.py                 fmp <command>                          [probe-free only]
  adjust/                adjust(), yfinance_to_raw()            [done — WP-7]
  probe/                 free-key probe                         [done — WP-1]
  client/ manifest/      async client, SQLite task manifest     [done — WP-2]
  runner.py              Downloader worker pool                 [done — WP-2]
  tests/mock_server.py   offline FMP simulator                  [done — WP-3]
  planner/ archive/      task planning, raw jsonl.gz archive    [WP-4, WP-5]
  builders/              archive -> Parquet vintage             [WP-6]
  datamanager_adapter/   FmpStore -> IncrementStore -> legacy   [WP-8]
  increments/            yfinance layer                         [WP-9]
  validators/            fmp validate / verify --final          [WP-A]
```

### Component Design

**`adjust/factors.py`** computes, per bar, `split_factor = Π 1/ratio over splits with S > t` and
`div_factor = Π DF(D) over dividends with D > t`. Each is a suffix product over sorted events, looked up with
`numpy.searchsorted(..., side="right")`, so the cost is O((bars + events) log events) with no Python loop over
bars. `DF(D)` uses the last finite raw close before `D`, restated on `D`'s share basis (decision D13).

**`adjust/adjust.py`** applies the factors: prices × `split_factor` (× `div_factor` in `total` mode), volume ÷
`split_factor`. Unknown columns pass through. Events after the last bar are ignored (spec §8.2).

**`adjust/yfinance_to_raw.py`** inverts the split adjustment yfinance applies at download time, using every
split known at download time, including ones after the window (D15). Dividends are restated to the raw basis.

**`probe/`** has four parts: `probe_client.py` (sync `httpx`, header auth, gzip cache, cross-run call ledger,
response classifier), `free_probe.py` (seven dependent stages; see the module docstring), `analysis.py` (pure
derivations: dividend basis, split ratios, ticker reuse) and `report.py` (Markdown report and synthetic
fixtures). The report keeps hand-written text above an `AUTO-GENERATED` marker.

**`tests/mock_server.py`** is a threaded stdlib `http.server` mock of `/stable`. Defaults reproduce the
probe's observations (401 JSON, 402 plain text, 404 `[]`, newest-first lists, `adj*` field names on the raw
endpoint, bad `from` ignored, gzip + chunked with no `Content-Length`). Unobserved behaviour is a `MockConfig`
knob, never a hard-coded guess. `Fault`s script 429/5xx/error-body/slow responses per endpoint and symbol.
Values are deterministic synthetic prices, so tests can assert exact rows.

**`client/`** — `FmpClient.fetch` loops: breaker wait → rate-limiter slot → one streamed GET → classify. Quota
raises at once; 401/403 streaks raise `AuthAbort`; `rate_limited`/`server_error`/`network_error` back off
(`base·2^(n-1)` ±50 % jitter, at least `Retry-After`) up to 6 attempts; everything else is final. Every attempt
that got a response reports `(wire, decoded)` bytes through `on_bytes`.

**`manifest/`** — SQLite (WAL, `synchronous=FULL`). `claim()` moves the lowest-priority-number `pending` rows to
`running` in one transaction; reopening reverts `running` to `pending`. `task_id` hashing makes planning idempotent.

**`runner.py`** — `Downloader` runs N workers: guard check → claim → fetch → sink → `complete`. Sink-then-manifest
gives at-least-once delivery (D28). Run-level stops (quota, auth, bandwidth) release the in-flight task and are
logged to `run_events`.

### Error handling
- `adjust()` raises `ValueError` on contradictory or impossible inputs rather than repairing them (D11, D16).
- The probe treats budget exhaustion as "skip and continue" (cached calls still resolve) and a quota message as
  "stop now". Transient errors (429, 5xx, transport) are not cached, so a re-run retries them.

## Data Flow
```
FMP /stable --(apikey header)--> ProbeClient --> DATA_CACHE_DIR/fmp-data/probe-free/<sha1>.json.gz
                                      |                         (local only, D5)
                                      v
                     FreeProbe stages -> ProbeRun -> report.py -> docs/fmp_probe_report.md
                                                             \-> tests/fixtures/*.json (synthetic)

raw bars + splits + dividends(amount = FMP `dividend`) --> adjust(mode) --> adjusted OHLCV
yfinance (auto_adjust=False, actions) --> yfinance_to_raw --> raw bars (+ raw-basis dividends)
```

## Design Decisions
See `docs/decisions.md` (D12–D24). The ones shaping this module most:
- D13: restate `C_prev` on the ex-date's share basis. A dividend on a split date is otherwise wrong by the ratio.
- D19/D20: FMP `dividend` is raw; `non-split-adjusted` carries **raw** prices under `adj*` field names.
- D24: the probe client is deliberately separate from the WP-2 production client.

## Integration Patterns
- The API key is resolved only through `src.config.provider_config.get_api_key`.
- `adjust()` is source-agnostic: FMP events and yfinance increment events feed the same function after the
  builder or the increment layer maps them to `date, numerator, denominator` / `date, amount`.
