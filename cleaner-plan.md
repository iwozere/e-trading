# Cleaner Plan — mypy phase 0 (cleaner-phase-0.txt)

Baseline: **1,168 mypy errors in 245 files** (log is UTF-16; decode before grepping).
Generated: 2026-07-08. Previous session crashed; its only surviving work is the
`pyproject.toml` mypy-override diff (keep it — commit first).

## Error profile

| Code | Count | Nature |
|---|---|---|
| assignment | 218 | mostly `self.x = None` w/o Optional annotation |
| attr-defined | 207 | mostly downstream of the same None pattern |
| arg-type | 140 | mixed: None-narrowing + real mismatches |
| var-annotated | 137 | mechanical: annotate empty `[]` / `{}` |
| index / operator / union-attr | 215 | untyped JSON-ish dicts + None unions |
| call-arg / call-overload | 75 | some real bugs (config_loader, p04) |
| import-not-found | 33 | dead imports + dynamic ml pipeline scripts |
| name-defined | 19 | real bugs: missing imports, undefined vars |
| rest | ~120 | long tail |

By module: ml 382, data 256, notification 146, common 95, indicators 89,
trading 53, config 37, telegram 30, strategy 25, other ~55.

## Strategy: fix by root-cause pattern, not file-by-file

### Phase A — Commit prior work + reproducible baseline
1. Commit the `pyproject.toml` mypy overrides (previous session's work).
2. Re-run `mypy src` piped through UTF-8 to regenerate a clean baseline; record count.

### Phase B — Real bugs first (~120 errors, the truly critical set)
These indicate code that crashes or misbehaves at runtime:
1. **name-defined (19)**: missing `setup_logger` import (p07 backtesting_bt),
   `ClientSession` (notification/service/client.py:184), `finra` in
   test_repo_short_squeeze, `args` in p00 x_02_train_hmm, `StructuralMetrics`
   (p04 daily_deep_scan:813), quoted `"pd.DataFrame"` string annotations in
   ml/future/automated_training_pipeline.py.
2. **used-before-def**: `_logger` in p04 daily_deep_scan.py:31.
3. **Missing return statements (2)**: collect_sentiment_async.py:411,
   p04 candidate_store.py:50.
4. **call-arg real bugs**: `Candidate(detection_date=, confidence_score=)` unexpected
   kwargs (volume_squeeze_detector_yf.py:381); `config_loader.py` missing required
   dataclass args (~15) — decide: add defaults to the config dataclasses or pass args.
5. **import-not-found of nonexistent src.* modules (~15)**: src.pricing, src.simulator,
   src.features, src.notification.notification_service / compatibility /
   async_notification_manager / service.main, src.data.utils.download_fundamentals,
   src.data.db.migrations.add_finra_short_interest_table — each is a dead import,
   renamed module, or orphaned test: fix the import or delete the dead file.
6. **abstract (6)**: entry/exit mixin factories instantiate abstract base when lookup
   fails — inspect and fix the fallback path.
7. **truthy-function (2)**: data_downloader_factory.py:124,126 — lambda/function
   used in boolean context, likely missing `()`.

### Phase C — Mechanical high-volume passes (~500+ errors, low risk)
1. **Implicit Optional (58 notes + cascading assignment errors)**: run the
   `no_implicit_optional` codemod over `src/`, review the diff.
2. **var-annotated (137)**: add container annotations; mypy's hint text gives the
   exact fix. Scriptable sweep.
3. **`self.x = None` in `__init__` (~260 attr-defined + large share of 218
   assignment)**: annotate as `X | None`, then add None-guards/asserts where
   accessed. Do prod modules first: trading → data → notification → common.
4. **Untyped state dicts (index/operator/union-attr storms)**: annotate
   `pipeline_state`-style dicts as `dict[str, Any]` (p01/p02 run_pipeline.py,
   binance_utils, logger.py) — collapses ~150 errors cheaply.

### Phase D — Structural decisions (ml pipelines, ~50 errors)
- x_NN scripts imported by bare name via dynamic loading (p01–p03 run_pipeline +
  tests): add per-module override `ignore_errors`/`ignore_missing_imports` for
  `src.ml.pipeline.*.tests` or set `mypy_path`, OR convert to package-relative
  imports. Recommend the mypy override — these are experiment scripts, not prod.
- `ReduceLROnPlateau(verbose=)` removed in newer torch — drop the kwarg (p02 x_03/x_04).

### Phase E — Long tail, prioritized by production criticality
Order: trading (53) → data (256) → notification (146) → common (95) →
config (37) → indicators (89) → telegram/strategy/analytics → ml (382, last;
much is paused experiment code — p17/p19 paused, ml/future speculative).
Batch per module; after each batch: `mypy <module>` + run that module's tests.

## Working rules (crash resilience)
- Commit after every completed batch (imperative-mood messages).
- Keep this file updated with a ✅ per finished phase/batch.
- Regenerate error counts per module after each phase to measure decline.

## Status
- [x] Phase A — commit pyproject + fresh baseline (d553773; baseline confirmed 1,168 errors — overrides were already active in phase-0 log)
- [x] Phase B — real bugs (commits 5ea45ef + 60e0d3a). Post-phase count: **1,028 errors in 221 files** (−140 errors, −24 files vs baseline). Notable findings beyond the plan:
  - Enabled `pydantic.mypy` plugin — killed ~15 false "missing named argument" errors in config_loader/api/portfolio; converted config_models Field defaults to keyword form for Pyright parity.
  - candidate_store.py had `return` inside the for loop (returned after first snapshot) + wrong `service.repo` accessor + wrong deep-scan method name.
  - volume_squeeze_detector_yf built Candidate with nonexistent kwargs AND nonexistent enum member `VOLUME_DETECTOR` — detector could never emit candidates (TypeError swallowed by except).
  - `import numpy as pd` shadowed pandas in ml/future/automated_training_pipeline.py.
  - collect_sentiment_batch: widened history_lookup to accept async callables (runtime already supported both); fixed fall-through return; annotated env config dict (killed ~28 dict-item errors).
  - Deleted orphaned artifacts: 4 notification delivery-history tests (service.main removed in 81aeb50), migration_example.py, client_usage.py (documents nonexistent client API), example_usage.py (FundamentalsDownloader gone), test_regime_colors.py (method removed), FINRA test's run_migration step (alembic owns schema now).
  - p16_taleb tests: converted to absolute imports and removed tests/__init__.py — its `src/` package shadowed the project `src` via pytest basedir insertion. 39/39 tests pass.
  - p13 run_p13.py --notify rewired from removed NotificationService to NotificationServiceClient.send_notification.
  - Also fixed Phase D item early: dropped `ReduceLROnPlateau(verbose=)` (p02 x_03/x_04).
- [x] Phase C — mechanical passes (commits f7989a7, bb38e15, 8f1ad6c, a982cb5).
  C1 no_implicit_optional codemod (1,028→963); C2 var-annotated sweep via mypy's own
  hints, 107 scripted + 27 manual (963→778); C3 Optional attribute annotations + guards
  (→665); C4 heterogeneous-dict annotations (→502). Backtrader `lines`/`params`
  metaclass tuples typed as ClassVar[Any] (subclass assignments re-infer, so each
  subclass needs its own annotation).
- [x] Phase D — x_NN dynamic-import override in pyproject (commit 2913820; −20 errors).
- [x] Phase E — long tail to **zero: `Success: no issues found in 1092 source files`**
  (commits 742c82d…09b63d6). Real runtime bugs found on the way:
  - adhoc_manager + alert_engine used nonexistent `service.repo` (AttributeError on
    every ad-hoc/alert DB call); p04 database helper returned dicts where tickers
    were promised.
  - x_01 data loaders imported get_ohlcv from `src.common` (removed export —
    ImportError at runtime); the call itself also used a nonexistent signature.
  - p07 `models/` artifact dir shadowed `models.py` for type checkers — moved the
    joblib artifact to `model_artifacts/`.
  - optuna `trial.report(epoch=)` (TypeError) and `Study.export_data` (never existed);
    talib EMA/ATR ndarrays used with .shift()/.values (AttributeError paths).
  - UnifiedCache lacked the clear_* methods DataManager.clear_cache called;
    advanced_caching passed retention_days as the `compression` positional and
    called nonexistent super().cleanup(); RedisDataCache.put dropped `overwrite`.
  - cleanup_failed_cache used `Path.parent[3]` (TypeError) instead of `parents[3]`.
  - websocket_manager had two method pairs silently shadowed by later same-name
    definitions (dead ones removed); query_analyzer.disable_monitoring referenced
    listener methods that were never stored.
  - pydantic-v1 `Field(env=...)` in notification service config was silently ignored
    by pydantic v2 — env names actually come from env_prefix + field name (documented).
  - feature_engineering_pipeline.__init__ read configs from the possibly-None param
    instead of self.config; run_daily_deep_scan's progress monkeypatch dropped the
    third argument (TypeError on every scan with progress enabled).
  - PyJWT installed (was missing; requirements.txt already listed it) and stale
    types-PyJWT stub dropped; repo tests migrated off removed UsersRepo convenience
    methods; ReposBundle test fixtures gained the missing kestrel repo.

Verification: mypy 0 errors; p16/p17/p19/strategy_pack suites 180 passed; notification
suite 97 passed with `-o asyncio_mode=auto` — remaining failures are pre-existing
(stale client-API tests, DB-dependent errors, pandas_ta broken on numpy 2.x).

Notes for follow-up (pre-existing, not addressed here):
- pytest.ini uses `[tool:pytest]` section header which pytest ignores in pytest.ini
  (needs `[pytest]`) — config there (testpaths, cov, markers, asyncio_mode) is inert.
  Async tests only pass with `-o asyncio_mode=auto`.
- pandas_ta cannot import on numpy 2.x (`from numpy import NaN`) — indicators adapter
  tests error at collection.
- notification test_client.py tests a removed client API (circuit_breaker_enabled).
- indicators test_registry.py expects fundamental keys (pe, pb, ps, peg, de_ratio,
  div_yield) missing from INDICATOR_META — fails identically before this work.
