import optuna
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.notification.logger import setup_logger
from src.ml.pipeline.p07_combined.data_loader import P07DataLoader
from src.ml.pipeline.p07_combined.regime_model import P07RegimeModel
from src.ml.pipeline.p07_combined.optimize import objective
from src.ml.pipeline.p07_combined.visualizer import P07Visualizer
from src.ml.pipeline.p07_combined.labeling import get_triple_barrier_labels
from src.ml.pipeline.p07_combined.features import build_features
from src.ml.pipeline.p07_combined.models import P07XGBModel
from src.ml.pipeline.p07_combined.evaluator import P07Evaluator
from src.ml.pipeline.p07_combined.robustness import P07RobustnessChecker
import vectorbt as vbt

_logger = setup_logger(__name__)

class P07Pipeline:
    """
    Core Orchestrator for p07_combined.
    Supports optional Multi-Timeframe (MTF) mode via enable_mtf flag.
    """

    def __init__(self,
                 result_root: Path = Path("results/p07_combined"),
                 db_url: Optional[str] = None,
                 enable_mtf: bool = False):
        self.result_root = Path(result_root)
        self.enable_mtf = enable_mtf
        self.data_loader = P07DataLoader()
        self.regime_model = P07RegimeModel()
        self._regime_trained: bool = False

        if db_url is None:
            db_path = (PROJECT_ROOT / "src" / "ml" / "pipeline" / "p07_combined" / "optuna_study.db").as_posix()
            self.db_url = f"sqlite:///{db_path}"
        else:
            self.db_url = db_url

        self.result_root.mkdir(parents=True, exist_ok=True)

    def train_macro_regimes(self, anchor_date: Optional[pd.Timestamp] = None, force: bool = False) -> bool:
        """
        Train or load the global HMM regime model.

        After the first successful training, subsequent calls are no-ops unless
        force=True.  Call this once before a batch loop (Phase 7.4) rather than
        per-ticker to avoid repeated HMM fitting.
        """
        if self._regime_trained and not force:
            return True

        vix = self.data_loader.load_vix()
        btc_mc = self.data_loader.load_btc_marketcap()

        macro_df = vix.join(btc_mc, how="outer").ffill().dropna()

        if not macro_df.empty:
            _logger.info("Retraining macro regime model with anchor_date: %s", anchor_date)
            success = self.regime_model.train(macro_df, anchor_date=anchor_date)
            if success:
                self._regime_trained = True
            return success
        return False

    def enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject global regime context into the ticker DataFrame, ensuring no look-ahead bias."""
        if not self._regime_trained:
            anchor_date = df.index.min()
            success = self.train_macro_regimes(anchor_date=anchor_date)
            if not success:
                _logger.warning("Failed to train regime model for anchor %s. Using default states.", anchor_date)
                df["global_regime"] = 0
                return df

        regimes = self.regime_model.predict(df)
        df["global_regime"] = regimes
        return df

    def _load_dataset(self, filepath: Path) -> pd.DataFrame:
        """Load and merge a single data file, applying MTF join when enabled."""
        if self.enable_mtf:
            return self.data_loader.get_mtf_dataset(filepath)
        return self.data_loader.get_merged_dataset(filepath)

    def get_result_dir(self, ticker: str, timeframe: str, start_date: str = "", end_date: str = "") -> Path:
        """Standardized path for results, nested by date range if provided."""
        base = self.result_root / ticker / timeframe
        if start_date and end_date:
            return base / f"{start_date}_{end_date}"
        return base

    def is_completed(self, ticker: str, timeframe: str, start_date: str = "", end_date: str = "") -> bool:
        """Check if result directory contains a completion flag."""
        res_dir = self.get_result_dir(ticker, timeframe, start_date, end_date)
        flag_file = res_dir / "completed.flag"
        return flag_file.exists()

    def mark_completed(self, ticker: str, timeframe: str, start_date: str = "", end_date: str = ""):
        """Create a completion flag."""
        res_dir = self.get_result_dir(ticker, timeframe, start_date, end_date)
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "completed.flag").touch()
        _logger.info("Marked %s_%s (%s_%s) as completed.", ticker, timeframe, start_date, end_date)

    def run_optimization(self, ticker: str, timeframe: str, df_enriched: pd.DataFrame, n_trials: int = 500, start_date: str = "", end_date: str = ""):
        """Execute the optimization loop using Optuna and save artifacts."""
        study_name = f"p07_{ticker}_{timeframe}_{start_date}_{end_date}" if start_date else f"p07_{ticker}_{timeframe}"
        study = optuna.create_study(
            study_name=study_name,
            storage=self.db_url,
            load_if_exists=True,
            direction="maximize"
        )

        enable_mtf = self.enable_mtf
        if len(study.trials) < n_trials:
            _logger.info("Starting optimization for %s_%s %s (%d trials)...", ticker, timeframe, f"[{start_date}_{end_date}]" if start_date else "", n_trials)
            study.optimize(
                lambda trial: objective(trial, df_enriched, timeframe=timeframe, enable_mtf=enable_mtf),
                n_trials=n_trials
            )
        else:
            _logger.info("Optimization already sufficient for %s_%s %s (%d trials).", ticker, timeframe, f"[{start_date}_{end_date}]" if start_date else "", len(study.trials))

        _logger.info("Optimization complete for %s_%s. Best value: %.4f", ticker, timeframe, study.best_value)

        # Save Artifacts
        if self.save_artifacts(ticker, timeframe, df_enriched, study.best_params, start_date, end_date):
            self.mark_completed(ticker, timeframe, start_date, end_date)

    def save_artifacts(self, ticker: str, timeframe: str, ohlcv_clean: pd.DataFrame, params: Dict[str, Any], start_date: str = "", end_date: str = "") -> bool:
        """Re-run the best model and save results using the shared Evaluator."""
        res_dir = self.get_result_dir(ticker, timeframe, start_date, end_date)
        res_dir.mkdir(parents=True, exist_ok=True)

        _logger.info("Saving artifacts to %s", res_dir)

        # Inject enable_mtf into params so evaluator passes it to build_features
        params_with_mtf = {**params, 'enable_mtf': self.enable_mtf}

        # 1. Run Shared Evaluation
        res = P07Evaluator.run_evaluation(ohlcv_clean, params_with_mtf, timeframe=timeframe)
        if "error" in res:
            _logger.error("Failed to run evaluation for artifacts: %s", res["error"])
            return False

        model = res["model"]
        pf = res["pf"]
        ohlcv_test = res["ohlcv_test"]

        # 2. Save Model & Data
        model.save_model(str(res_dir / "best_model.json"))

        # Save detailed trades
        res["trades"].to_json(res_dir / "trades.json", orient="records", indent=4)

        # Save performance metrics
        res["metrics"].to_json(res_dir / "metrics.json", indent=4)

        # 3. Clean Signal Mapping for Plotting
        assets = pf.assets()
        diff = assets.diff()
        if not diff.empty:
            diff.iloc[0] = assets.iloc[0]

        plot_sigs = pd.Series(0, index=ohlcv_test.index)
        plot_sigs[diff > 0] = 1
        plot_sigs[diff < 0] = -1

        # 4. Walk-Forward Efficiency gate
        is_sharpe = pf.sharpe_ratio()
        try:
            rob_dir = res_dir / "robustness"
            checker = P07RobustnessChecker(ticker, timeframe, rob_dir)
            rob_results = checker.run_all_checks(ohlcv_clean, params_with_mtf, res)
            avg_oos_sharpe = rob_results.get('wfa', {}).get('avg_oos_sharpe', float('nan'))
            if is_sharpe and is_sharpe > 0 and not pd.isna(avg_oos_sharpe):
                wfe = avg_oos_sharpe / is_sharpe
            else:
                wfe = float('nan')
            _logger.info("Walk-Forward Efficiency for %s %s: %.3f (threshold 0.50)", ticker, timeframe, wfe)
            if not pd.isna(wfe) and wfe < 0.5:
                _logger.warning(
                    "Strategy REJECTED for %s %s: WFE=%.3f < 0.50. "
                    "completed.flag will NOT be written.", ticker, timeframe, wfe)
                return False
        except Exception as e:
            _logger.warning("WFE check skipped for %s %s due to error: %s", ticker, timeframe, str(e))

        # 5. Save Plots
        viz = P07Visualizer(res_dir)
        viz.plot_tbm_hits(res["y_test"])
        viz.plot_prediction_diagnostics(
            model._map_labels(res["y_test"]).values,
            model.predict_proba(res["X_test"])
        )
        viz.plot_master_overlay(ohlcv_test, plot_sigs, pf)

        _logger.info("Artifacts saved successfully.")
        return True

    def run_robustness(self, ticker: str, timeframe: str):
        """Runs the robustness check for a trained strategy."""
        _logger.info("Running robustness check for %s %s", ticker, timeframe)

        # 1. Load Data
        data_dir = Path("data")
        ticker_files = list(data_dir.glob(f"{ticker}_{timeframe}_*.csv"))

        if not ticker_files:
            _logger.error("No data files found for %s %s", ticker, timeframe)
            return

        dfs = []
        for f in ticker_files:
            dfs.append(self._load_dataset(f))

        ohlcv = pd.concat(dfs).sort_index()
        ohlcv = ohlcv.loc[~ohlcv.index.duplicated(keep='last')]
        ohlcv = self.enrich_data(ohlcv)

        # 2. Get Best Params from Study
        all_studies = optuna.get_all_study_summaries(storage=self.db_url)
        study_name = None
        for s in all_studies:
            if s.study_name.startswith(f"p07_{ticker}_{timeframe}"):
                study_name = s.study_name
                break

        if not study_name:
            _logger.error("No study found for %s %s", ticker, timeframe)
            return

        study = optuna.load_study(study_name=study_name, storage=self.db_url)
        params = study.best_params

        # 3. Initialize Robustness Checker
        res_dir = self.get_result_dir(ticker, timeframe) / "robustness"
        checker = P07RobustnessChecker(ticker, timeframe, res_dir)

        # 4. Run Evaluation to get base result
        res = P07Evaluator.run_evaluation(ohlcv, params, timeframe=timeframe)
        if "error" in res:
            _logger.error("Base evaluation failed for robustness: %s", res["error"])
            return

        # 5. Execute Checks
        summary = checker.run_all_checks(ohlcv, params, res)

        # 6. Plot Robustness Artifacts
        viz = P07Visualizer(res_dir)
        if (res_dir / "wfa_equity.json").exists():
            wfa_equity = pd.read_json(res_dir / "wfa_equity.json", typ='series')
            viz.plot_walk_forward_equity(wfa_equity)

        if 'monte_carlo' in summary and 'shuffled_returns' in summary['monte_carlo'] and 'random_skips' in summary['monte_carlo']:
            viz.plot_monte_carlo_distribution(
                summary['monte_carlo']['shuffled_returns'],
                summary['monte_carlo']['random_skips']
            )
        else:
            _logger.warning("Skipping Monte Carlo plots due to missing data: %s", summary.get('monte_carlo', {}).get('error', 'Unknown error'))

        viz.plot_sensitivity_report(summary['sensitivity']['sensitivity_results'])

        _logger.info("Robustness complete for %s %s. Artifacts in %s", ticker, timeframe, res_dir)

    def run_batch(self, ticker_files: List[Path], mode: str = "optimize", train_years: Optional[List[str]] = None):
        """Process multiple files with Cross-File Validation logic."""
        # 1. Group files by (ticker, timeframe)
        groups = {}
        for filepath in ticker_files:
            ticker, timeframe, start, end = self.data_loader.parse_filename(filepath)
            if not ticker: continue

            if train_years:
                if not any(yr in start or yr in end for yr in train_years):
                    _logger.debug("Skipping file %s as it doesn't match training years %s", filepath.name, train_years)
                    continue

            key = (ticker, timeframe)
            if key not in groups: groups[key] = []
            groups[key].append({'path': filepath, 'start': start, 'end': end, 'year': start[:4]})

        # 2. Pre-train macro regime model ONCE before the batch loop (Phase 7.4)
        self.train_macro_regimes()

        # 3. Process each group
        for (ticker, timeframe), files in groups.items():
            # Sort by start date to find latest for validation
            files.sort(key=lambda x: x['start'])

            if len(files) > 1:
                opt_files = files[:-1]
                val_file = files[-1]
                _logger.info("Cross-File Setup for %s %s: Opt on %d files, Val on %s",
                             ticker, timeframe, len(opt_files), val_file['path'].name)
            else:
                opt_files = files
                val_file = None
                _logger.info("Single-File Setup for %s %s: Opt on %s",
                             ticker, timeframe, opt_files[0]['path'].name)

            # --- A. Optimization Phase ---
            try:
                dfs = []
                for f in opt_files:
                    df_merged = self._load_dataset(f['path'])
                    dfs.append(self.enrich_data(df_merged))

                agg_start = min(f['start'] for f in opt_files)
                agg_end = max(f['end'] for f in opt_files)

                if self.is_completed(ticker, timeframe, agg_start, agg_end):
                    _logger.info("Optimization already completed for aggregated %s %s (%s_%s)", ticker, timeframe, agg_start, agg_end)
                else:
                    self.run_optimization(ticker, timeframe, dfs, start_date=agg_start, end_date=agg_end)

                # --- B. Validation Phase (Optional) ---
                if val_file:
                    _logger.info("Running Out-of-Sample Validation for %s %s on %s", ticker, timeframe, val_file['path'].name)
                    study_name = f"p07_{ticker}_{timeframe}_{agg_start}_{agg_end}"
                    study = optuna.load_study(study_name=study_name, storage=self.db_url)

                    df_val = self._load_dataset(val_file['path'])
                    df_val = self.enrich_data(df_val)

                    self.save_artifacts(ticker, timeframe, df_val, study.best_params,
                                        start_date=f"{val_file['start']}_VAL",
                                        end_date=val_file['end'])

            except Exception as e:
                _logger.error("Failed to process group %s %s: %s", ticker, timeframe, str(e), exc_info=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="P07 Combined Pipeline")
    parser.add_argument("--ticker", type=str, help="Specific ticker to run")
    parser.add_argument("--tf", type=str, help="Specific timeframe to run")
    parser.add_argument("--years", type=str, help="Comma-separated years to train on (e.g., 2022,2023,2024)")
    parser.add_argument("--enable-mtf", action="store_true", default=False,
                        help="Enable Multi-Timeframe (anchor TF) features")
    args = parser.parse_args()

    p = P07Pipeline(enable_mtf=args.enable_mtf)
    _logger.info("Starting P07 Pipeline Batch... (MTF=%s)", args.enable_mtf)
    data_dir = Path("data")

    # Standard segment files (ticker_tf_start_end.csv)
    pattern = "*_*_*.csv"
    if args.ticker and args.tf:
        pattern = f"{args.ticker}_{args.tf}_*.csv"
    elif args.ticker:
        pattern = f"{args.ticker}_*_*.csv"

    ticker_files = list(data_dir.glob(pattern))

    train_years = args.years.split(",") if args.years else None

    if ticker_files:
        p.run_batch(ticker_files, train_years=train_years)
    else:
        _logger.warning("No data files found matching pattern %s", pattern)

    # Also run aggregation at the end
    try:
        from src.ml.pipeline.p07_combined.json2csv import aggregate_results
        aggregate_results()
    except Exception as e:
        _logger.error("Failed to aggregate results: %s", e)
