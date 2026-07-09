# Mypy Cleanup — Final Report

**Date:** 2026-07-09
**Result:** `mypy src` — **Success: no issues found in 1092 source files** (baseline: 1,168 errors in 245 files)
**Commits:** `d553773` … `957ca12` (19 commits; plan and per-phase details in `cleaner-plan.md`)

## Phase summary

| Phase | What was done | Errors after |
|---|---|---|
| A | Committed pyproject mypy overrides, recorded type stubs, regenerated baseline | 1,168 (baseline) |
| B | Real runtime bugs (name-defined, used-before-def, call-arg, abstract, dead imports); `pydantic.mypy` plugin enabled | 1,028 |
| C1 | `no_implicit_optional` codemod over `src/` (45 files) | 963 |
| C2 | var-annotated sweep — mypy's own hints applied by script (107 sites) + 27 manual | 778 |
| C3 | `Optional[...]` on None-initialized attributes + runtime guards | 665 |
| D | pyproject override for ml `x_NN` bare-name dynamic imports | −20 |
| C4 | Heterogeneous stats/results dicts annotated `Dict[str, Any]` | 502 |
| E | Long tail, file by file across all modules | **0** |

## Real runtime bugs found and fixed (beyond type annotations)

- **`service.repo` did not exist** in `p04` `adhoc_manager` and `alert_engine` — every ad-hoc-candidate and alert-cooldown DB call raised AttributeError (`service.repos.short_squeeze` is correct). The p04 `database.py` helper also returned dicts where tickers were promised.
- **p01–p03 `x_01` data loaders** imported `get_ohlcv` from `src.common` (export removed → ImportError at runtime), and the p02 call used a signature that never existed.
- **`DataManager.clear_cache()` called three `UnifiedCache` methods that didn't exist** (`clear_symbol_timeframe`/`clear_symbol`/`clear_all`) — implemented.
- **`advanced_caching`**: passed `retention_days` into the `compression` positional of `DataCache.__init__`, called nonexistent `super().cleanup()`, and `RedisDataCache.put` silently dropped the `overwrite` parameter.
- **optuna**: `trial.report(epoch=…)` is not a parameter (TypeError) → `step=`; `Study.export_data` never existed → `joblib.dump`.
- **talib returns bare ndarrays** — p07/p08 features and labeling used `.shift()`/`.values` on them (AttributeError paths); wrapped in `pd.Series`.
- **`cleanup_failed_cache`** used `Path(__file__).parent[3]` (TypeError) instead of `parents[3]`.
- **`websocket_manager`** had two method pairs silently shadowed by later same-name definitions (dead versions removed); **`query_analyzer.disable_monitoring`** referenced listener methods that were never stored (AttributeError).
- **pydantic v2 ignores `Field(env=…)`** — the notification service env names actually come from `SettingsConfigDict(env_prefix)` + field name; verified empirically, dead kwargs removed and rule documented in the module.
- **`feature_engineering_pipeline.__init__`** read component configs from the possibly-None parameter instead of `self.config`; **`run_daily_deep_scan`'s** progress monkeypatch dropped the third argument (TypeError whenever progress tracking was enabled).
- **p07 `models/` artifact directory shadowed `models.py`** for type checkers — `macro_hmm.joblib` moved to `model_artifacts/`.
- **PyJWT was not installed** (requirements.txt lists it; `api/auth.py` could not import) — installed, and the stale v1-era `types-PyJWT` stub removed.
- Repo tests migrated off removed `UsersRepo.ensure_user_for_telegram`/`update_telegram_profile` (moved to `UsersService`) via a conftest helper; `ReposBundle` test fixtures gained the missing `kestrel` repo.
- p06 sentiment collector imported two conflicting `SentimentFilterConfig` classes (first import silently shadowed).
- `BinanceEnhancedFeed` had a duplicate `_load` that overrode the robust implementation, and its `super().__init__()` call was missing required arguments.

## Conventions established

- `pydantic.mypy` plugin is enabled; use `Field(default=…, …)` keyword form.
- Backtrader `lines`/`params` class tuples: annotate `ClassVar[Any]`. Subclass assignments re-infer types, so each subclass needs its own annotation (base-class annotation alone is not enough).
- ml `x_NN` scripts imported by bare name have a per-module `disable_error_code = ["import-not-found"]` override in pyproject.
- Import fallbacks (`X = None` in `except ImportError`) are annotated `X: Any = None`.

## Verification

- `mypy src`: 0 errors (89 informational `annotation-unchecked` notes remain — see below).
- p16/p17/p19/strategy_pack suites: 180 passed.
- Notification suite: 97 passed with `-o asyncio_mode=auto`; remaining failures are pre-existing (stale client-API tests, DB-dependent errors).

## Remaining / follow-up (pre-existing, not regressions)

1. **89 mypy notes** `annotation-unchecked` ("bodies of untyped functions are not checked") across 35 files — mypy skips function bodies that have no annotations at all. Options: annotate those functions, enable `check_untyped_defs = true` (will surface new errors inside them), or silence with `disable_error_code = ["annotation-unchecked"]`.
2. **pytest.ini is inert**: it uses the `[tool:pytest]` section header, which pytest ignores in a `pytest.ini` file (needs `[pytest]`). Nothing in it applies — including any `asyncio_mode`; async tests only pass with `-o asyncio_mode=auto`.
3. **pandas_ta is broken on numpy 2.x** (`from numpy import NaN`) — indicators adapter tests fail at collection.
4. **notification `test_client.py`** tests a removed client API (`circuit_breaker_enabled`, old `NotificationRequest` signature).
5. **indicators `test_registry.py`** expects fundamental keys (pe, pb, ps, peg, de_ratio, div_yield) that are absent from `INDICATOR_META` — fails identically before this work.
6. ~93 pre-existing failures in `src/common/sentiments/tests` (verified failing before any changes).
7. DB repo tests need a local PostgreSQL test database (`ALEMBIC_DB_URL`).
