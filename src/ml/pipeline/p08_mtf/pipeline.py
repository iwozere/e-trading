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

    def __init__(self, db_url: str = "sqlite:///src/ml/pipeline/p08_mtf/optuna_study.db"):
        super().__init__(db_url)
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

    def run_batch(self, ticker_files: List[Path]):
        """MTF version: merges with anchor file before optimization/validation."""
        # 1. Group files by (ticker, timeframe)
        groups = {}
        for filepath in ticker_files:
            ticker, timeframe, start, end = self.data_loader.parse_filename(filepath)
            if not ticker: continue
            key = (ticker, timeframe)
            if key not in groups: groups[key] = []
            groups[key].append({'path': filepath, 'start': start, 'end': end})

        # 2. Process each group
        for (ticker, timeframe), files in groups.items():
            try:
                files.sort(key=lambda x: x['start'])
                if len(files) < 2:
                    _logger.debug("Skipping %s %s: Need at least 2 years for cross-file validation.", ticker, timeframe)
                    continue

                opt_files = files[:-1]
                val_file = files[-1]

                # Use AGGREGATE range for naming
                agg_start = min(f['start'] for f in opt_files)
                agg_end = max(f['end'] for f in opt_files)

                if self.is_completed(ticker, timeframe, agg_start, agg_end):
                    _logger.info("P08 skipping %s %s: already completed.", ticker, timeframe)
                    continue

                # Load and merge MTF for each segment
                dfs = []
                for f in opt_files:
                    df_mtf = self.data_loader.get_mtf_dataset(f['path'])
                    dfs.append(df_mtf)

                # Train/Optimize
                best_params = self.run_optimization(ticker, timeframe, dfs, start_date=agg_start, end_date=agg_end)

                # Validate on the Holdout year
                _logger.info("Running P08 Validation for %s on %s", ticker, val_file['path'].name)
                df_val = self.data_loader.get_mtf_dataset(val_file['path'])

                self.save_artifacts(ticker, timeframe, df_val, best_params,
                                   start_date=f"{val_file['start']}_VAL",
                                   end_date=val_file['end'])

            except Exception as e:
                _logger.error("P08 Group Error (%s %s): %s", ticker, timeframe, str(e), exc_info=True)

def aggregate_results_p08():
    """Wrapper for json2csv modified for P08 path."""
    from src.ml.pipeline.p07_combined.json2csv import aggregate_results
    aggregate_results(results_root="results/p08_mtf")

if __name__ == "__main__":
    p = P08Pipeline()
    _logger.info("--- P08 MTF Pipeline Initiative ---")
    data_dir = Path("data")
    ticker_files = list(data_dir.glob("BTCUSDT_*.csv")) + list(data_dir.glob("ETHUSDT_*.csv")) \
                   + list(data_dir.glob("LTCUSDT_*.csv")) + list(data_dir.glob("XRPUSDT_*.csv"))

    if ticker_files:
        p.run_batch(ticker_files)
        aggregate_results_p08()
