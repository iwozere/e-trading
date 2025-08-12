## **Tasks.md**

```markdown
# Task Breakdown: HMM + LSTM + Optuna Trading Pipeline

## Phase 1: Data & Preprocessing
1. Implement `x_01_data_loader.py`
   - Fetch OHLCV for multiple symbols & timeframes.
   - Store in `data/`.
2. Implement `x_02_preprocess.py`
   - Compute log returns, normalize features.
   - Add rolling statistics and default indicator values.
   - Save in `data/processed/`.

## Phase 2: Market Regime Detection
3. Implement `x_03_train_hmm.py`
   - Train HMM with selected features.
   - Save model.
4. Implement `x_04_apply_hmm.py`
   - Apply HMM to label data with regimes.
   - Save labeled dataset.

## Phase 3: Optimization
5. Implement `x_05_optuna_indicators.py`
   - Define objective: maximize Sharpe ratio / profit factor.
   - Search indicator parameters (RSI, BB, SMA/EMA).
   - Save best parameters JSON.
6. Implement `x_06_optuna_lstm.py`
   - Define objective: minimize MSE & maximize directional accuracy.
   - Use Optuna with TPE sampler & pruning.
   - Save best parameters JSON.

## Phase 4: Model Training & Validation
7. Implement `x_07_train_lstm.py`
   - Train LSTM using optimized parameters.
   - Save trained model.
8. Implement `x_08_validate_lstm.py`
   - Evaluate model vs naive baseline.
   - Produce charts and PDF report.

## Phase 5: Automation & Maintenance
9. Write `retrain_pipeline.sh` for automated runs.
10. Add logging & exception handling for all scripts.
11. Maintain `config/pipeline/` YAML files for flexible configuration.
````
