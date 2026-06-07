# `src/ml/future/` — Experimental / Not Production-Ready

> **WARNING: Code in this directory is experimental and has NOT been audited for
> production use.  Do NOT use any pipeline in this directory to generate live
> trading signals until it has been through the promotion checklist below.**

## Status

All code here is **work-in-progress** or **proof-of-concept**.  It may contain:

- Temporal data leakage (shuffle=True on time-series, scaler fit before split)
- Missing OOS validation
- Incomplete error handling
- Hardcoded paths / credentials

## Promotion checklist (before moving to `src/ml/pipeline/`)

- [ ] Temporal leakage audit: split must use `shuffle=False`; scaler must be fit on train split only
- [ ] OOS validation: walk-forward or held-out test set with no lookahead
- [ ] Hyperparameter search is reproducible (fixed seed; no data from future folds)
- [ ] Model artifacts serialized with a safe format (or pickle trust boundary documented)
- [ ] Unit tests covering feature engineering and split/scale order
- [ ] Code review by a second engineer

## Known issues fixed so far

- `automated_training_pipeline.py`: `shuffle=True` replaced with `shuffle=False`; scaler now fit on train only (2026-06-06)
