# ML Pipeline Re-tuning Guide — 2026-06-09 Refactor

Reference plan: `src/ml/pipeline/docs/plan-2026-06-09.md`  
Reference done log: `src/ml/pipeline/docs/done-2026-06-09.md`

---

## Why Re-tuning Is Required

The 2026-06-09 refactor changed three things that invalidate every previously saved
model and Optuna study:

| Change | Why it invalidates prior results |
|--------|----------------------------------|
| Triple-barrier labels now computed **after** the train/val/test split | Label distribution on the training set is different; models trained on pre-split labels learned from contaminated targets |
| Optuna objective now scores on `pf_val` (held-out validation portfolio), not `pf_test` | Old "best" hyperparameters were selected on the test set — they are overfit |
| 3-way 60/20/20 split instead of a 2-way split | Effective training size is smaller; models trained on the old split have seen more data than they will going forward |

Do **not** use any model checkpoint, Optuna study, or backtest result produced before
this refactor date as a starting point. Start clean.

---

## Step 0 — Reset Optuna Databases

All existing Optuna SQLite studies must be deleted before any re-tuning begins.
Running new trials in an old DB will mix contaminated and clean trials in the
`best_trial` record, making study results unreliable.

```bash
# Delete all pipeline Optuna DBs (adjust glob as needed for your paths)
rm -f db/cnn_lstm_xgboost.db
rm -f db/p07_combined.db
rm -f db/p08_mtf.db
rm -f db/p01_hmm_lstm.db
# Or nuke the whole db/ directory if pipelines are the only users:
rm -rf db/
mkdir db/
```

Confirm every `storage:` entry in `config/pipeline/*.yaml` points to a now-empty file.
Optuna will recreate the schema on first use.

---

## Step 1 — P01: HMM-LSTM with `multi_regime: true`

### Recommendation

Enable `multi_regime` from the start. Training one LSTM per HMM regime is the
approach P00 was built around; it is now a first-class config flag in P01 rather
than a separate pipeline. Starting with it enabled avoids having to re-tune again
later when regime-specific behavior is needed.

**`config/pipeline/p01.yaml`:**

```yaml
lstm:
  multi_regime: true        # ← enable
  min_regime_samples: 200   # skip regimes with fewer than 200 samples
```

### When to disable

Turn `multi_regime` off only if:
- Your HMM produces fewer than 3 stable regimes (run Stage 3 and inspect
  `hmm_regimes.csv` — a healthy run shows at least 3 distinct states each
  appearing ≥ 15% of the time).
- Any single regime has fewer than `min_regime_samples` rows after filtering;
  that regime is silently skipped, but if *all* regimes are sparse the run will
  produce zero models.

### Re-tuning order for P01

1. Run Stage 3 (HMM) standalone and inspect regime distribution.
2. Increase `min_regime_samples` if regimes are unbalanced — a value of 300–500
   is safer for daily data with < 3 years of history.
3. Run Stage 4 (Optuna indicator selection) — this now uses only the first 60%
   of data, so expect slightly different indicator sets than before the refactor.
4. Run Stage 5 (LSTM Optuna) and Stage 6 (training) with `multi_regime: true`.
5. Validate with Stage 7 (`x_07_validate_lstm.py`). Check that
   `find_regime_models()` returns a model file for each regime you expected.

---

## Step 2 — P02: CNN-LSTM-XGBoost with `cnn_variant: simple`

### Recommendation

Start with `cnn_variant: simple` (CNN1D). The simple variant has fewer parameters,
trains faster, and is less prone to overfitting on the now-smaller 60% training
window. Only switch to `cnn_lstm` if CNN1D's walk-forward directional accuracy
plateaus below ~55% across all folds after tuning.

**`config/pipeline/p02.yaml`:**

```yaml
cnn_variant: simple       # ← start here
target_strategy: single   # keep single until simple variant is tuned
```

### Tuning sequence

1. **CNN1D baseline** (`cnn_variant: simple`, `target_strategy: single`):
   - Run the full Optuna study (20+ trials recommended).
   - Check `walk_forward_validate()` results in the validation report:
     - `avg_directional_accuracy` ≥ 0.54 is the threshold to consider the model
       worth keeping.
     - `avg_mse` trending down across folds (not random) indicates the model is
       learning something.

2. **Multi-target extension** (optional, only after step 1 passes):
   - Set `target_strategy: multi`.
   - Re-run Optuna — the additional targets (direction, volatility, trend,
     magnitude) add complexity; expect longer training and possibly higher
     `avg_mse` on the primary target. Evaluate whether auxiliary targets improve
     downstream XGBoost feature quality.

3. **Upgrade to CNN-LSTM** (optional, only if CNN1D hits a ceiling):
   - Set `cnn_variant: cnn_lstm`.
   - Reset the Optuna study for this symbol (delete the study from the DB, not
     the whole DB at this point — you can use `optuna.delete_study()`).
   - Expect 2–3× longer training time.

### XGBoost

XGBoost hyperparameters are independent of the CNN variant. Tune XGBoost after
the CNN embedding is stable (i.e., after step 1 above passes) so the feature
distribution it sees does not shift under it.

---

## Step 3 — P07: Combined Pipeline with WFE Gate

### Walk-Forward Efficiency (WFE) gate

After each P07 optimization run, `save_artifacts()` computes:

```
WFE = avg_OOS_sharpe / IS_sharpe
```

A `WFE < 0.5` means the strategy loses more than half its in-sample edge when
tested out-of-sample — a strong signal of overfitting.

**This is a soft warning, not a hard rejection.**

The gate will log a `WARNING` and skip writing `completed.flag`, but the artifacts
(model checkpoints, backtest results) are still saved. Practitioners working with
illiquid instruments, low-frequency strategies, or short histories (< 1 year) may
see structurally lower WFE due to noisy OOS samples rather than true overfitting.

**How to interpret and override:**

| WFE | Interpretation | Action |
|-----|----------------|--------|
| ≥ 0.7 | Healthy generalisation | Accept result, write flag |
| 0.5–0.69 | Acceptable but worth investigating | Accept; check whether OOS period covers a regime change |
| 0.3–0.49 | Warning zone | Investigate before deploying; increase `n_trials`, add regularisation, or widen `test_split` |
| < 0.3 | Likely overfit | Do not deploy; reduce model complexity or collect more data |

To accept a WFE < 0.5 result deliberately:

```bash
# Write the flag manually after human review
touch results/p07/<ticker>_<timeframe>/completed.flag
```

Document the override in your run log with the WFE value and the reason.

### MTF mode

Leave `enable_mtf: false` during initial re-tuning. Add MTF features only after
the baseline (single-timeframe) model passes the WFE gate, so you have a clean
reference point to compare against.

### Re-tuning order for P07

1. Run `run_batch()` with `enable_mtf=False` on all tickers.
2. For each ticker: check `completed.flag` presence and the WFE log line.
3. For tickers with WFE 0.5–0.69: re-run with more Optuna trials
   (`n_trials: 50` in `config/pipeline/p07.yaml`).
4. Once baseline is stable: re-run with `--enable-mtf` and compare WFE.

---

## Step 4 — P06: EMPS2 Rolling Memory Bootstrap

The first time `RollingMemoryScanner.check_bootstrap_health()` runs after a clean
deploy, it writes a `.rolling_memory_state.json` sidecar in the results folder
and records `first_run_date`. The health check will not fire an error until the
scanner has been active for more than `lookback_days` (default: 14) without
producing any Phase 1 detections.

**What to do after a fresh deploy:**

1. Run at least 5 consecutive trading days before evaluating Phase 1 output.
2. If after 14+ active days `cumulative_phase1_count` is still 0, the error
   log will fire — check:
   - `results_base_path` is pointing to the correct folder containing
     per-date subdirectories.
   - `05_volatility_filtered.csv` files are being written by Stage 5.
   - `phase1_min_appearances` in `RollingMemoryConfig` is not set too high
     for the number of trading days available (5 appearances out of 14 days
     is the default; the scanner needs ≥ 5 days of actual data files to
     start producing Phase 1 signals).

3. If you are replaying historical results for backtesting, pass the historical
   date as `target_date` — the bootstrap counter accumulates correctly across
   replay runs as long as `results_base_path` stays consistent.

---

## Step 5 — Deprecated Pipelines

Do not schedule or invoke these directly after the refactor:

| Deprecated | Use instead | How |
|------------|-------------|-----|
| `p00_hmm_3lstm/run_pipeline.py` | `p01_hmm_lstm/run_pipeline.py` | Set `lstm.multi_regime: true` in `p01.yaml` |
| `p03_cnn_xgboost/run_pipeline.py` | `p02_cnn_lstm_xgboost/run_pipeline.py` | Set `cnn_variant: simple` in `p02.yaml` |
| `p05_emps/` | Deleted — use `p06_emps2/` | — |
| `p08_mtf/pipeline.py` | `p07_combined/pipeline.py` | Pass `--enable-mtf` flag |
| `p10_emps3/emps3_pipeline.py` | `p06_emps2/emps2_pipeline.py` | Pass `--analyzer-type accumulation` |

The shims still work and will redirect automatically, but they add an extra
subprocess hop and emit deprecation warnings in logs. Update any scheduler
entries (cron, Airflow, task queue) to call the canonical pipelines directly.

---

## Recommended Re-tuning Order (Full)

```
Step 0  Reset all Optuna DBs
Step 1  P01 — HMM (Stage 3)  →  confirm regime distribution
Step 2  P01 — Optuna indicators (Stage 4)  →  60% data only
Step 3  P01 — LSTM Optuna + training (Stages 5–6) with multi_regime: true
Step 4  P01 — Validate (Stage 7)  →  check per-regime model files
Step 5  P02 — Optuna + training with cnn_variant: simple, target_strategy: single
Step 6  P02 — walk_forward_validate()  →  directional_accuracy ≥ 0.54
Step 7  P02 — XGBoost Optuna
Step 8  P07 — run_batch() baseline (enable_mtf=False)  →  check WFE for all tickers
Step 9  P07 — run_batch() MTF (--enable-mtf)  →  compare WFE vs baseline
Step 10 P06 — run 5+ days  →  confirm Phase 1 detections appear
Step 11 Update scheduler entries to remove deprecated pipeline references
```

---

## Quick Reference — Key Thresholds

| Metric | Target | Source |
|--------|--------|--------|
| HMM regime balance (each regime) | ≥ 15% of bars | P01 Stage 3 |
| `min_regime_samples` | ≥ 200 (daily); ≥ 500 (intraday) | `p01.yaml` |
| P02 `avg_directional_accuracy` | ≥ 0.54 across WFA folds | `walk_forward_validate()` |
| P07 Walk-Forward Efficiency | ≥ 0.5 (soft gate; override with documented reason if < 0.5) | `save_artifacts()` |
| P09 cointegration p-value | ≤ 0.05 | `test_cointegration()` |
| P09 half-life | 5–30 bars (instrument-dependent) | `calculate_half_life()` |
| EMPS2 Phase 1 min appearances | 5 in 14 days (default) | `RollingMemoryConfig` |
