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

- 2026-07-10 — **Phase B complete: 1,132 → 878 errors, all Optional rules at zero** (234 sites; the narrowing also cleared ~20 knock-on errors in other rules).
  Prod fixes (24 sites, 10 files) — all genuine None-crash paths:
  - FINRA downloaders: `yf.download()` result used without None guard.
  - `api/main.py`: conditional-import fallback collapsed `StrategyManager | None` annotation to `None`; fixed with TYPE_CHECKING alias.
  - `binance_live_feed._run_websocket_loop` / `file_data_feed._realtime_simulation_loop`: thread entry points now guard un-initialized loop/df.
  - `notification_db_centric_bot`: `hasattr` on possibly-None processor.
  - p04 scripts (4×): `"Usage:" in __doc__` didn't guard `__doc__ is None` (crashes under `python -OO`).
  - `run_daily_deep_scan`: closure now binds a non-optional local tracker.
  - p07 `data_loader`: None guard before `.empty`/`.to_csv`.
  Tests (~210 sites, 35+ files): inserted `assert x is not None` immediately after Optional-returning calls (script-assisted + 20 manual sites: Optional model attrs like `result.sources`/`run.job_snapshot`, `spec.loader`, `tzinfo`). Strengthens the tests; runtime behavior unchanged (AssertionError replaces AttributeError/TypeError).
  Verified: mypy 0; modified runnable suites byte-identical results vs pre-change baseline (stash comparison).

Phase C starting counts: 878 errors in ~175 files — reportAttributeAccessIssue ~350, reportArgumentType ~260, reportCallIssue ~190, reportIndexIssue ~49, reportOperatorIssue ~37.

- 2026-07-11 — **Phase C complete: 878 → 493 errors; production code at 0 pyright errors** (remaining 493 are all in test files). Batches: C1 feeds+ticker_chart (a8848f8), C2 p04/p07/p08+mixin+drawdowns (37f03d9), C3 services/cache/screeners/CLIs (1fb351d), C4 prod long tail (a4e45d7).
  Real bugs found by the cleanup:
  - `binance_data_feed`: `timezone.utcfromtimestamp` (no such method) crashed backfill + websocket paths.
  - `binance_live_feed.get_status`: `ws.sock` doesn't exist on new websockets API.
  - `health_cli`: async_command applied after @cli.command() — every CLI command returned an un-awaited coroutine and did nothing.
  - `indicators/service.py`: `pd.isfinite` doesn't exist; None fed to IndicatorValue (pydantic rejects).
  - `ScreenerResult.fundamental_score/technical_score`: computed but never attached — message lines + 2 test files were dead code; fields added and populated.
  - `file_based_cache._split_data_by_years`: mask/frame misalignment (ValueError with date filter in DatetimeIndex branch).
  - P07 walk-forward crashed on list-of-segments input (`ohlcv.loc` on a list); now mirrors P08's concat.
  - `run_short_data`: `sort_values(na_last=True)` — parameter doesn't exist (TypeError).
  - `ta_lib_adapter`: unknown string implementation fell through to `fn(...)` → 'str' not callable.
  - `validate_timestamps`: DatetimeIndex input silently skipped tz validation.
  Established idioms: `.to_numpy(dtype=float)` at pandas→numpy/talib boundaries; `pd.DatetimeIndex(df.index)` normalization; `cast(pd.Timestamp, k)` at `.items()` loops; Any-aliases for backtrader metaclass ctors; TYPE_CHECKING mixin contracts (paper_trading_mixin); NewType wraps for indicator requests; `Field(default=None, ...)` not positional; per-line `# pyright: ignore[rule]` only for true third-party gaps (vectorbt dynamic metrics, sanpy, plotly row/col ints, platform signals).
  mypy verified 0 after every batch.

Phase D starting counts: 493 errors in 77 files, 100% test code — reportAttributeAccessIssue 201, reportArgumentType 133, reportCallIssue 114, reportIndexIssue 41.

- 2026-07-11 — **Phase D batch D1: 493 → 362 errors** (tests/ root cluster). Deleted 3 dead suites that test removed/renamed APIs: `test_client.py` (sync `requests`-based client + CircuitBreaker `can_execute`/`record_failure` — real client is async/aiohttp), `test_new_config_system.py` (flat `TradingBotConfig(broker_type=…)` — model is now nested), `test_bot_manager.py` (procedural `start_bot`/`running_bots` module API that exists nowhere). Mechanical fixes: `test_hmm_lstm_backtest.py` (typed `convert_numpy -> Any`; dropped `test_parameter_optimization_disabled` — 4-arg call to a 2-arg method testing a removed disabled-returns-{} contract), `test_sharpe_calculation.py` (`calculate_sharpe_ratio_manual -> Any`, its float-sentinel/dict return broke `result[...]` indexing), `test_scheduler_main_application.py` (`notification_client` → `notification_db_service` rename; `cast` for intentional None config; removed 2 assertions on the deleted `config.notification` section; deleted `TestSignalHandling` — tested the old single-arg `create_task` signal impl, now `setup_signal_handlers(loop, application)`). Suite-rot still latent (not pyright errors): scheduler notification-timeout validation test + `test_main` single-arg `setup_signal_handlers` assertion. mypy 0.

- 2026-07-12 — **Phase D batch D2: 362 → 229 errors** (indicators module + shared test doubles). Committed by the user inside `7a907d6`/`a8cb3dc` alongside unrelated IBKR work. Deleted 1 dead suite: `test_batch_processing.py` (100% built on removed `compute_batch` with `fail_fast` param + `IndicatorResultSet`-per-ticker return that don't match successor `get_batch_indicators`; that API is covered by the migration test). Idiom fixes: backtrader metaclass ctors → module/local `Any` aliases (`test_backtrader_integration` `_RSI/_BBands/_MACD`; `_PandasData` in `test_eom_indicator`/`test_support_resistance_indicator`); `IndicatorRequest`/`TickerIndicatorsRequest`/`BatchIndicatorRequest` NewType wraps (`TickerSymbol`/`TimeFrame`/`Period`/`IndicatorName`/`ProviderName`) across `test_core_functionality`/`test_migration_compatibility`/`test_service`/`test_performance_benchmarks`/`service_fixtures`; `hasattr`→`getattr` for absent attrs; implicit-Optional `Dict = None` → `Dict | None = None` (`service_fixtures`, `indicator_service_mock`); `convert_numpy`/helper `-> Any`. API-drift fixes (restore real coverage): `IndicatorSet.technical`→`technical_indicators`, `IndicatorResult(timestamp=)`→`category=`+`source=`, `CompositeRecommendation.overall_recommendation`→`recommendation`; `IndicatorService._ta_lib_adapter`/`_pandas_ta_adapter`→`adapters["ta-lib"]`/`["pandas-ta"]`, `_config_manager`→`config_manager`, `get_indicator_parameters`→`get_parameters`, `compute_batch`→`get_batch_indicators(BatchIndicatorRequest)`; `TaLibAdapter._compute_indicator`→`_compute_sync`; `FundamentalsAdapter(fundamentals_getter=)`→`fundamentals_data=`. Real bugs fixed: `asyncio.run(await service.compute_for_ticker(...))` (await already resolves the coroutine; also inside a running loop) → plain `await` ×3; sync tests calling now-async `compute`/`adapter.compute` without running the coroutine (`test_service`, `test_adapters`, benchmark ThreadPoolExecutor submitting un-awaited coroutines) → `asyncio.run(...)`; `IndicatorValue(value=None)` on a required `float|Dict` field → `0.0`; mock `should_raise_errors` typed `Dict[str, BaseException | Callable[[], BaseException]]` so `raise error()` is valid. Verified no test regressions: `test_core_functionality` 27F/4P identical pre/post; `test_migration_compatibility` improved 7F/7P → 6F/8P (async-bug fix). mypy 0; 176 tests collect clean. Remaining failures are pre-existing suite-rot (recommendation engine unavailable in test env).
