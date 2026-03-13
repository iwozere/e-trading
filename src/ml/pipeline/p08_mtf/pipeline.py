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
from src.ml.pipeline.p08_mtf.data_loader import P08DataLoader
from src.ml.pipeline.p08_mtf.evaluator import P08Evaluator
from src.ml.pipeline.p08_mtf.optimize import objective
from src.ml.pipeline.p07_combined.pipeline import P07Pipeline
from src.ml.pipeline.p07_combined.visualizer import P07Visualizer

_logger = setup_logger(__name__)

class P08Pipeline(P07Pipeline):
    """
    P08 MTF Pipeline: Extends P07 with Multi-Timeframe capabilities.
    Uses Anchor TFs for trend-aware context.
    """

    def __init__(self, db_url: Optional[str] = None, result_root: Path = Path("results/p08_mtf")):
        if db_url is None:
            db_path = (PROJECT_ROOT / "src" / "ml" / "pipeline" / "p08_mtf" / "test_optuna.db").as_posix()
            db_url = f"sqlite:///{db_path}"
        super().__init__(result_root=result_root, db_url=db_url)
        self.data_loader = P08DataLoader()

    def get_result_dir(self, ticker: str, timeframe: str, start_date: str, end_date: str) -> Path:
        """Standardized results path for P08."""
        return Path("results/p08_mtf") / ticker / timeframe / f"{start_date}_{end_date}"

    def run_optimization(self, ticker: str, timeframe: str, ohlcv_segments: List[pd.DataFrame],
                         n_trials: int = 100, start_date: str = "", end_date: str = ""):
        """Runs the MTF Optuna study."""
        study_name = f"p08_{ticker}_{timeframe}_{start_date}_{end_date}"
        _logger.info("Starting MTF optimization for %s (%d trials)...", study_name, n_trials)

        study = optuna.create_study(
            study_name=study_name,
            storage=self.db_url,
            direction="maximize",
            load_if_exists=True
        )

        study.optimize(lambda t: objective(t, ohlcv_segments, timeframe=timeframe), n_trials=n_trials)
        _logger.info("Best Adjusted Sharpe: %f", study.best_value)
        return study.best_params

    def save_artifacts(self, ticker: str, timeframe: str, ohlcv_clean: Any,
                      params: Dict[str, Any], start_date: str = "", end_date: str = "") -> bool:
        """Saves MTF models, metrics, and diagnostic plots."""
        res_dir = self.get_result_dir(ticker, timeframe, start_date, end_date)
        res_dir.mkdir(parents=True, exist_ok=True)

        _logger.info("Saving P08 MTF artifacts to %s", res_dir)

        # 1. Run Evaluation
        res = P08Evaluator.run_evaluation(ohlcv_clean, params, timeframe=timeframe)
        if "error" in res:
            _logger.error("Failed to run P08 evaluation: %s", res["error"])
            return False

        # 2. Save JSONs
        res["model"].save_model(str(res_dir / "best_model.json"))
        res["trades"].to_json(res_dir / "trades.json", orient="records", indent=4)
        res["metrics"].to_json(res_dir / "metrics.json", indent=4)

        # 3. Save Visuals (Reusing P07 visualizer for now)
        viz = P07Visualizer(res_dir)
        viz.plot_tbm_hits(res["y_f"])
        viz.plot_prediction_diagnostics(
            res["model"]._map_labels(res["y_test"]).values,
            res["model"].predict_proba(res["X_test"])
        )
        # For overlay, we need signals mapped to index
        pf = res["pf"]
        ohlcv_test = res["ohlcv_test"]
        assets = pf.assets()
        diff = assets.diff()
        if not diff.empty:
            diff.iloc[0] = assets.iloc[0]
        plot_sigs = pd.Series(0, index=ohlcv_test.index)
        plot_sigs[diff > 0] = 1
        plot_sigs[diff < 0] = -1

        viz.plot_master_overlay(ohlcv_test, plot_sigs, pf)
        return True

    def run_batch(self, ticker_files: List[Path], train_years: Optional[List[str]] = None):
        """MTF version: merges with anchor file before optimization/validation."""
        # 1. Group files by (ticker, timeframe)
        groups = {}
        for filepath in ticker_files:
            ticker, timeframe, start, end = self.data_loader.parse_filename(filepath)
            if not ticker: continue

            # (Optional) Filter by year if train_years is specified
            if train_years:
                if not any(yr in start or yr in end for yr in train_years):
                    _logger.debug("Skipping file %s as it doesn't match training years %s", filepath.name, train_years)
                    continue

            key = (ticker, timeframe)
            if key not in groups: groups[key] = []
            groups[key].append({'path': filepath, 'start': start, 'end': end, 'year': start[:4]})

        # 2. Process each group
        for (ticker, timeframe), files in groups.items():
            try:
                files.sort(key=lambda x: x['start'])

                # Cross-File Validation Logic
                if len(files) > 1:
                    opt_files = files[:-1]
                    val_file = files[-1]
                    _logger.info("P08 Cross-File Setup for %s %s: Opt on %d files, Val on %s",
                                 ticker, timeframe, len(opt_files), val_file['path'].name)
                else:
                    opt_files = files
                    val_file = None
                    _logger.info("P08 Single-File Setup for %s %s: Opt on %s",
                                 ticker, timeframe, opt_files[0]['path'].name)

                # Use AGGREGATE range for naming
                agg_start = min(f['start'] for f in opt_files)
                agg_end = max(f['end'] for f in opt_files)

                if self.is_completed(ticker, timeframe, agg_start, agg_end):
                    _logger.info("P08 skipping %s %s: already completed.", ticker, timeframe)
                else:
                    # Load and merge MTF for each segment
                    dfs = []
                    for f in opt_files:
                        df_mtf = self.data_loader.get_mtf_dataset(f['path'])
                        dfs.append(df_mtf)

                    # Train/Optimize
                    best_params = self.run_optimization(ticker, timeframe, dfs, start_date=agg_start, end_date=agg_end)

                # Validate on the Holdout year (if exists)
                if val_file:
                    _logger.info("Running P08 Validation for %s on %s", ticker, val_file['path'].name)
                    # Load best params from study
                    study_name = f"p08_{ticker}_{timeframe}_{agg_start}_{agg_end}"
                    study = optuna.load_study(study_name=study_name, storage=self.db_url)

                    df_val = self.data_loader.get_mtf_dataset(val_file['path'])

                    self.save_artifacts(ticker, timeframe, df_val, study.best_params,
                                       start_date=f"{val_file['start']}_VAL",
                                       end_date=val_file['end'])

            except Exception as e:
                _logger.error("P08 Group Error (%s %s): %s", ticker, timeframe, str(e), exc_info=True)

    def run_robustness(self, ticker: str, timeframe: str):
        """MTF-Aware Robustness: Uses P08RobustnessChecker."""
        _logger.info("Running P08 MTF robustness check for %s %s", ticker, timeframe)
        from src.ml.pipeline.p08_mtf.robustness import P08RobustnessChecker

        # 1. Load All Data (MTF merged)
        files = list(Path("data").glob(f"{ticker}_{timeframe}_*.csv"))
        if not files:
            _logger.error("No data files for robustness.")
            return

        dfs = [self.data_loader.get_mtf_dataset(f) for f in sorted(files)]

        # 2. Get Best Params
        all_studies = optuna.get_all_study_summaries(storage=self.db_url)
        study_name = next((s.study_name for s in all_studies if s.study_name.startswith(f"p08_{ticker}_{timeframe}")), None)

        if not study_name:
            _logger.error("No study found for robustness.")
            return

        study = optuna.load_study(study_name=study_name, storage=self.db_url)
        params = study.best_params

        # 3. Checker
        res_dir = self.result_root / ticker / timeframe / "robustness"
        checker = P08RobustnessChecker(ticker, timeframe, res_dir)

        # 4. Base Eval
        res = P08Evaluator.run_evaluation(dfs, params, timeframe=timeframe)
        if "error" in res:
             _logger.error("Base eval failed.")
             return

        # 5. Run Suite
        checker.run_all_checks(dfs, params, res)

        # 6. Visualization (Reuse P07 Visualizer)
        from src.ml.pipeline.p07_combined.visualizer import P07Visualizer
        viz = P07Visualizer(res_dir)
        if (res_dir / "wfa_equity.json").exists():
             wfa_equity = pd.read_json(res_dir / "wfa_equity.json", typ='series')
             viz.plot_walk_forward_equity(wfa_equity)
        # Monte Carlo & Sensitivity from checker summary
        summary_path = res_dir / "robustness_summary.json"
        if summary_path.exists():
            import json
            with open(summary_path, "r") as f:
                summary = json.load(f)
            viz.plot_monte_carlo_distribution(
                summary['monte_carlo']['shuffled_returns'],
                summary['monte_carlo']['random_skips']
            )
            viz.plot_sensitivity_report(summary['sensitivity']['sensitivity_results'])

        _logger.info("P08 robustness complete.")

def aggregate_results_p08():
    """Wrapper for json2csv modified for P08 path."""
    from src.ml.pipeline.p07_combined.json2csv import aggregate_results
    _logger.info("Aggregating P08 MTF results into flattened CSVs...")
    aggregate_results(results_root="results/p08_mtf")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="P08 MTF Pipeline")
    parser.add_argument("--ticker", type=str, help="Specific ticker to run")
    parser.add_argument("--tf", type=str, help="Specific timeframe to run")
    parser.add_argument("--years", type=str, help="Comma-separated years to train on")

    args = parser.parse_args()

    p = P08Pipeline()
    _logger.info("--- P08 MTF Pipeline Initiative ---")
    data_dir = Path("data")

    # Filter files
    pattern = "*_*_*.csv"
    if args.ticker and args.tf:
        pattern = f"{args.ticker}_{args.tf}_*.csv"
    elif args.ticker:
        pattern = f"{args.ticker}_*_*.csv"

    ticker_files = list(data_dir.glob(pattern))
    train_years = args.years.split(",") if args.years else None

    if ticker_files:
        p.run_batch(ticker_files, train_years=train_years)
        aggregate_results_p08()
        
        # --- Automatic Post-Optimization Suite ---
        _logger.info("--- Starting Automated Post-Optimization Suite ---")
        
        # Move imports here to avoid circular dependencies
        from src.ml.pipeline.p08_mtf.select_candidates import select_top_candidates
        from src.ml.pipeline.p08_mtf.run_robustness_checks import run_robustness_batch
        from src.ml.pipeline.p08_mtf.run_generalization_test import run_generalization
        
        # 1. Select Candidates
        _logger.info("Step 1: Selecting Top Candidates...")
        select_top_candidates(results_root="results/p08_mtf", top_n=5)
        
        # 2. Run Robustness Checks
        candidates_file = Path("results/p08_mtf/p08_robustness_candidates.csv")
        if candidates_file.exists():
            _logger.info("Step 2: Running Robustness Checks for candidates...")
            run_robustness_batch(candidates_file)
            
            # 3. Run Generalization Tests
            _logger.info("Step 3: Running Generalization Tests for candidates...")
            run_generalization(candidates_file)
        else:
            _logger.warning("No candidates found for robustness/generalization.")
            
        _logger.info("--- P08 MTF Pipeline Suite Complete ---")
    else:
        _logger.warning("No data files found matching pattern %s", pattern)
