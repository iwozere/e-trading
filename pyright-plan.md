# Pyright Cleanup Plan

Goal: pyright (basic mode) at **0 errors** on active code, enforced in CI alongside mypy,
without ever regressing the mypy=0 invariant.

Baseline (2026-07-10, pyright 1.1.408): **2,377 errors / 5 warnings in 286 of 1,148 files.**

Agreed decisions (2026-07-10):
1. **Scope**: exclude legacy/experimental code from pyright; clean active code to zero.
2. **Test doubles**: introduce typing Protocols for key interfaces; fakes satisfy the Protocol.
3. **Enforcement**: pin pyright version, gate in CI once zero (ratchet until then).

## Baseline profile

| Rule | Count | Nature |
|---|---|---|
| reportAttributeAccessIssue | 978 | mostly pandas/numpy inference noise (no pandas-stubs) |
| reportArgumentType | 621 | same family + test doubles |
| reportCallIssue | 226 | mixed |
| reportOptionalMemberAccess / Subscript / others | ~260 | real None-handling gaps |
| reportPrivateImportUsage | 41 | yfinance/backtrader internals |
| reportUndefinedVariable | 40 | **real bugs** (runtime NameError) |
| reportUnusedCoroutine | 13 | **real bugs** (missing await, mostly async tests) |
| remainder | ~200 | mixed small rules |

Concentration: top 50 files hold 60% of errors; top 100 hold 81%.
~790 errors are in test files. ~900 are in legacy ML pipelines/examples.

## Phase 0 — Fix the measurement first (reshapes every number below)

1. `pyrightconfig.json`: `pythonVersion` 3.11 → **3.13** (`.venv` is 3.13.1, prod Pi is 3.13).
2. Install **pandas-stubs** (matching pandas 2.3.x) into `.venv`. Note the split-brain:
   pyright resolves against `.venv`, while mypy/pytest currently run from the system
   Python 3.13. Decide the canonical dev env, align both checkers to it, and re-verify
   mypy stays at 0 after any stub install.
3. Add `exclude` entries for legacy/non-runtime code (verify each is inactive first):
   - `src/ml/future`, `src/ml/pipeline/p01_hmm_lstm`, `p02_cnn_lstm_xgboost`, `p03_cnn_xgboost`, `p09_arbitrage`
   - `src/data/examples`, `src/notification/docs`
4. Create a `typings/` directory with minimal `.pyi` stubs for untyped third parties
   (backtrader, talib, pandas_ta, yfinance privates). One good stub removes hundreds of
   leaf errors without touching call sites.
5. Re-run pyright → record the new baseline in the progress log.

## Phase A — Real bugs (~80–100 errors, highest value per fix)

Fix rule-by-rule; run the affected module's tests after each:
- `reportUndefinedVariable` (40) — e.g. `run_backtrader_p09.py` `initial_cash`, `delivery_tracker.py` `Task`.
- `reportUnusedCoroutine` (13) — un-awaited async calls; mostly tests that currently assert nothing. Adding `await` may surface real failures — fix those too.
- `reportMissingImports` (14) — missing dep vs dead module, decide per case.
- Small rules: `reportUnusedExcept` (2), `reportOptionalCall` (2), `reportUnhashable` (9), `reportUnusedExpression` (3), `reportAbstractUsage` (2), `reportWildcardImportFromLibrary` (2).

## Phase B — Optional/None discipline (~250 errors)

`reportOptionalMemberAccess` (176), `reportOptionalSubscript` (73), Optional operand/iterable.
Many are genuine edge-case crashes (values from `get_ohlcv() -> DataFrame | None` used
unchecked). Fix the shared providers' return contracts first, then add early-return guards
at call sites.

## Phase C — pandas/numpy inference noise (the bulk, ~1,400+)

Working principles, in priority order:
1. **Source-first**: annotate the shared foundations whose types propagate (DataManager,
   cache pipeline, `src/common/technicals`, indicators). One return annotation clears many leaf files.
2. **Boundary coercion**: `float()`/`bool()`/`int()` at scalar extraction points
   (`.iloc[-1]`, `.mean()`). These are correctness fixes, not lint appeasement — the
   2026-07-10 prod failures (np.float64→psycopg2 in p20_ingest_eod, np.bool_→pydantic in
   SP-2/SP-3) were exactly this class.
3. **Explicit annotations** at extraction sites (`close: pd.Series = df["close"]`) where inference degrades.
4. **Per-line `# pyright: ignore[rule]`** only for real third-party gaps not covered by
   `typings/`. Never file-level suppression in active code.

Work files in error-count order (top of list: `step02_calculate_timeframes.py`,
`fred_downloader.py`, `unified_cache.py`, `strategy_pack/strategies.py`, p18 processors).

## Phase D — Tests via Protocols (~790 errors in tests)

- Define small Protocols in the interface-owning modules, e.g. `src/data/protocols.py`:
  `OHLCVProvider` (`get_ohlcv`, `get_ohlcv_batch`), notification client protocol, etc.
- Type consumers against the Protocol where they only need the narrow surface
  (e.g. `RunContext.dm`) — this both documents the real dependency and lets fakes
  (`_FakeDataManager`) type-check without ignores.
- Update existing fakes; remove any interim per-line ignores.

## Phase E — Enforcement

- Pin the pyright version used for cleanup (pyright-python pin in dev requirements).
- CI: run mypy + pyright. Until zero: ratchet script fails on any per-file count increase
  vs the committed baseline JSON. At zero: switch to plain `pyright` (fail on any error).
- Keep the "fix diagnostics in files you touch" working convention in the meantime.

## Working rules

- One batch = one module/directory. After each batch: module tests green, `mypy` still 0,
  pyright count recorded. One commit per batch, imperative message.
- Never weaken runtime behavior to satisfy the checker; if a fix adds a runtime guard,
  say so in the commit message.

## Measurement

```bash
pyright --outputjson > /tmp/pyright.json
python -c "
import json, collections
d = json.load(open('/tmp/pyright.json'))['generalDiagnostics']
print(len(d), 'errors in', len({x['file'] for x in d}), 'files')
for r, n in collections.Counter(x.get('rule') for x in d).most_common(10): print(f'{n:5d}  {r}')
"
```

## Progress log

- 2026-07-10 — Baseline recorded: 2,377 errors / 286 files (pyright 1.1.408, basic mode). Phases defined, scope decisions agreed.
- 2026-07-10 — **Phase 0 complete: 2,377 → 1,242 errors (−48%), 286 → 196 files.**
  - `pyrightconfig.json`: pythonVersion 3.11 → 3.13; excluded legacy (`src/ml/future`, p00, p01, p02, p03, p09, `src/data/examples`, `src/notification/docs`). p00 added beyond the agreed list (same precursor category, no external imports — verified). **p07/p08 kept in scope**: imported by active `src/strategy/p08_xgb_strategy.py`.
  - Installed into `.venv` (pyright's resolution env): `pandas-stubs` 3.0.3, `redis`, `zstandard`, `san` (the latter three are optional runtime deps imported by active `src/data/utils` code). Recorded in `requirements-dev.txt`.
  - `typings/` local stubs turned out unnecessary — pandas-stubs removed nearly all backtrader/talib-adjacent noise (talib: 1 error left). Item dropped.
  - mypy verified still clean (system env untouched by the .venv installs).
  - Split-brain note: canonical env decision deferred; for now pyright ⇒ `.venv`, mypy/pytest ⇒ system 3.13. Revisit in Phase E.
  - Notable remaining buckets for next phases: reportPrivateImportUsage is 26× `matplotlib.pyplot` (Phase C, likely one idiom fix); reportMissingImports leftovers are 2× p07 onnx extras + dead test imports (`src.notification.async_notification_manager`, `src.risk.controller`, `src.data.data_feed_factory`) — Phase A material.

Phase A starting counts: reportUndefinedVariable 38, reportUnusedCoroutine 13, reportMissingImports 7, small rules ~12.

- 2026-07-10 — **Phase A complete: 1,242 → 1,132 errors, all Phase A rules at zero.**
  Real bugs fixed in prod code:
  - `delivery_tracker.py` scrambled annotation `asyncio.Optional[Task]` → `Optional[asyncio.Task]`.
  - `binance_live_feed.py` + `webui_app_service.py`: unreachable duplicate `except Exception` blocks removed.
  - `performance_optimization.py` `get_chunk()`: parquet path called `pq.read_table(row_groups=...)` — no such parameter, would TypeError at runtime; rewritten to row-range slice consistent with `iter_chunks()`. Dead expression removed.
  - `extract_schema_as_md.py`: dead expression from abandoned approach removed.
  - `ibkr_live_feed.py` + `check_ibkr_live_readonly.py`: `from ib_insync import *` → explicit imports.
  - Installed `onnxruntime`, `onnxconverter-common` into `.venv` for p07 (recorded in requirements-dev).
  Test suites:
  - 13 un-awaited coroutine sites fixed by converting sync tests calling async APIs to `@pytest.mark.asyncio` across 4 indicator test files + `test_client.py`; also removed nonexistent `pytest.subTest` usage and fixed never-running `async def teardown_method`. Result: those 4 indicator files went 47 failed/7 passed → 18 failed/36 passed.
  - Repaired fixable stale tests: `test_live_bot_database.py` (old repo API → `upsert_bot`/`add`/`open_trades`/`BotInstance`), `test_live_bot_config.py` (import path), `test_base_data_downloader.py` (missing abstract `get_provider_name`), `test_e2e_standalone.py` + `test_infrastructure.py` (half-migrated uow leftovers).
  - Deleted 4 dead test files whose subjects no longer exist: `tests/test_screener_bot.py`, `tests/test_emailer.py`, `tests/test_email_attachments.py` (all import removed `async_notification_manager`), `tests/test_risk_controller.py` (`src/risk` removed). These broke default `pytest` collection (`testpaths = tests`).
  mypy re-verified at 0 errors.

### Suite-rot findings (needs its own work item, out of pyright scope)
- `src/indicators/tests/`: remaining ~18 failures are structural — tests patch attributes that no longer exist on `IndicatorService` (`_ta_lib_adapter`, `_compute_indicator`), fixture param `fundamentals_getter` renamed, etc. Needs rewrite against current API.
- `src/notification/tests/test_client.py`: 100% dead — every test errors at setup (`NotificationServiceClient` no longer takes `circuit_breaker_enabled`; `CircuitBreaker`/`NotificationRequest` APIs changed; sync tests mock `requests` but client is aiohttp). Needs full rewrite.
- `src/notification/tests/test_delivery_tracker.py`: async tests unmarked → never ran under strict asyncio mode.

Phase B starting counts: reportOptionalMemberAccess 162, reportOptionalSubscript 70, remaining total 1,132 in 182 files.
