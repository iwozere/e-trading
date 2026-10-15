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
  archive/               raw jsonl.gz archive (the sink)        [done — WP-5]
  planner/               universe, task planning, projection    [done — WP-4]
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
gives at-least-once delivery (D28). Run-level stops (quota, auth, bandwidth, archive) release the in-flight task
and are logged to `run_events`.

**`archive/`** — `RawArchive` is the production sink. Each response becomes one JSON line, compressed as its own
gzip member and appended to `raw/<dump_id>/<endpoint>/<encoded symbol>.jsonl.gz`, then fsynced. `reader.py`
streams members with `zlib` (one decompressor per member, so member boundaries are exact), which is how a torn
tail is found and how the builder reads the archive; `latest_successful` keeps the newest `ok`/`ok_empty` line per
request (endpoint + canonical params). `symbol_path.py` makes file names Windows-safe and reversible:
percent-encoding, a `_` prefix for device names (`CON`, `NUL`, `COM1`…), and a hash suffix for lower-case symbols
so `ABC` and `abc` never share a file on NTFS (D32).

**`planner/`** — `Planner.fetch_sources` downloads the bootstrap lists through the normal `Downloader` (priority 0,
`max_priority=0`). `Planner.plan` reads them and finished profiles back from the archive, `universe.py` assembles
and tiers the universe, `generate.py` yields every task with a size estimate (or a deferral reason), `projection.py`
sums done (real bytes) and remaining (estimates) per block and tier and schedules months under both counting
hypotheses, and `report.py` renders the two reports. Task priority is `block * 100 + sub`: bootstrap 0, then
delisted before active, and for intraday the tier order. `windowing.py` keeps all date windows on fixed grids so
task ids survive re-planning (D37). `follow_ups.FollowUps` is the `Downloader`'s `Reviewer`: next page while a page
is non-empty, daily bounds into the manifest, truncated intraday windows relabelled and halved (D38, D39).

**`executor.py`** — `DumpExecutor.run` loops over the blocks of a `RunScope`, checks disk space against the plan's
projection, and runs one `Downloader` per block (archive sink, `FollowUps` reviewer, shared guard with the
optional `--max-gb` limit) until done or the first run-level stop (D43). `status.py` renders progress from
`Manifest.totals()` and reads the manifest without recovering `running` tasks (D44).

**`builders/`** — `VintageBuilder.build` (``builder.py``): universe from the archive (same rules as the planner,
`read_sources`/`read_profiles`) → `IdentityBuilder` (``identity.py``: ids from the append-only `IdRegistry`,
reuse segmentation, symbol history, `resolve(symbol, date)`) → `build_events` (``events.py``) → `PriceBuilder`
(``prices.py``: per symbol, newest response per request key, window check, one bar per date, clipping to segments,
`compute_flags` from ``flags.py``, one Parquet file per security) → securities and history → the generic
`RecordTableBuilder` (``records.py``) for every `TableSpec` in ``tables.py`` → `validate_structure`
(``validate.py``) → `build_info.json`, registry save, directory swap, `current.txt`. Parquet writing is in
``parquet_io.py`` (fixed settings, no timestamps in the files, so rebuilds are byte-identical). D45–D49.

**`datamanager_adapter/`** — `FmpStore` (``fmp_store.py``) opens the vintage `current.txt` names, loads
securities and symbol history, and reads prices per security, events per security or for many at once, and many
securities' daily bars with one DuckDB query. `FmpOhlcvAdapter` (``adapter.py``) resolves the request, appends
`IncrementSource` bars (``increments.py``; `NoIncrements` until WP-9), unions events (FMP wins), applies
`adjust()` (hourly bars by exchange-local date with daily `C_prev`), caches the adjusted full history in an LRU
and slices the request. ``factory.py`` builds it from `fmp.adapter` or returns `None`; `DataManager` calls it
first and falls back to `_get_ohlcv_legacy`. ``golden.py`` hashes `get_ohlcv` offline for the regression check.
D50–D55, `docs/datamanager_integration.md`.

**`increments/`** — `IncrementIngester.run` (``ingest.py``) checks the writer host, takes the lock, then per
timeframe groups the alive securities by window start, fetches batches through
`YahooDataDownloader.get_unadjusted_history`, converts each frame with `to_raw_bars` (`yfinance_to_raw`), verifies
the overlap, appends only bars after the last stored one (flagged in context), records new events, and writes
`_meta.json`. `IncrementStore` (``store.py``) owns the files and is the adapter's `IncrementSource`. D56–D60.

**`validators/`** — ``framework.py`` (registry, `CheckResult`, `ValidationContext` with the lazily computed price
scan and the seeded sample, `run_checks`, report rendering), ``scan.py`` (one pass over every price file, NYSE
calendar), the checks in ``checks_prices.py``, ``checks_events.py``, ``checks_reference.py`` and
``checks_pipeline.py`` (order in ``checks.py``), ``checksums.py`` (`sha256sum -c` files, gzip test, backup
check), ``coverage.py`` and ``verify.py`` (the spec §14 checklist). D61–D65.

**`probe/` (paid)** — `PaidProbe` (``paid_probe.py``) runs adaptive stages through the cached `ProbeClient`;
``paid_analysis.py`` holds the pure rules (convention fitting, row cap and truncation side, split verdict, window
completeness, reuse verdict); ``paid_report.py`` renders the report; ``results.py`` stores findings and settings and
`apply_probe_results` overlays them on the YAML config in every CLI command; ``calibrate.py`` decides the
bandwidth basis. D66–D68.

### Error handling
- `adjust()` raises `ValueError` on contradictory or impossible inputs rather than repairing them (D11, D16).
- The probe treats budget exhaustion as "skip and continue" (cached calls still resolve) and a quota message as
  "stop now". Transient errors (429, 5xx, transport) are not cached, so a re-run retries them.
- The archive turns every write problem (root gone, < 2 GB free, `OSError`) into `ArchiveUnavailable`; the runner
  pauses instead of failing tasks (D34). A partial member left by a crash is truncated on the next append (D33).

## Data Flow
```
FMP /stable --(apikey header)--> ProbeClient --> DATA_CACHE_DIR/fmp-data/probe-free/<sha1>.json.gz
                                      |                         (local only, D5)
                                      v
                     FreeProbe stages -> ProbeRun -> report.py -> docs/fmp_probe_report.md
                                                             \-> tests/fixtures/*.json (synthetic)

manifest (pending) --claim--> FmpClient --FetchResult--> RawArchive.append --sha256--> manifest (ok/...)
                                     |                         |
                                     v                         v
                              bandwidth_log        raw/<dump_id>/<endpoint>/<symbol>.jsonl.gz
                                                               |
                                                   reader.latest_successful --> builder (WP-6)

fmp plan: bootstrap lists --Downloader(max_priority=0)--> archive --> universe --> generate --> manifest
                                                                        \--> budget_plan.md, universe_report.md
fmp run:   Downloader --reviewer FollowUps--> next page / window halves / symbol_bounds --> next fmp plan

raw bars + splits + dividends(amount = FMP `dividend`) --> adjust(mode) --> adjusted OHLCV
yfinance (auto_adjust=False, actions) --> yfinance_to_raw --> raw bars (+ raw-basis dividends)
```

## Design Decisions
See `docs/decisions.md` (D12–D68). The ones shaping this module most:
- D13: restate `C_prev` on the ex-date's share basis. A dividend on a split date is otherwise wrong by the ratio.
- D19/D20: FMP `dividend` is raw; `non-split-adjusted` carries **raw** prices under `adj*` field names.
- D24: the probe client is deliberately separate from the WP-2 production client.
- D28/D33: sink-then-manifest plus torn-tail repair make kill-and-resume lose nothing and corrupt nothing.
- D36/D37: planning is iterative (decide only what the data allows) and windows sit on fixed grids, so re-running
  `fmp plan` is always safe and never duplicates work.
- D46/D48: `security_id`s come from an append-only registry saved only after a validated build, and a vintage
  becomes active only by a directory swap followed by `current.txt`, so a failed build changes nothing.
- D50/D54: the adapter sits in front of an unchanged legacy path, is off by default, and the golden snapshot
  proves the legacy output byte-identical (1,294 series, 0 diffs).

## Integration Patterns
- The API key is resolved only through `src.config.provider_config.get_api_key`.
- `adjust()` is source-agnostic: FMP events and yfinance increment events feed the same function after the
  builder or the increment layer maps them to `date, numerator, denominator` / `date, amount`.
