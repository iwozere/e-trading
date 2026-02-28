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
import vectorbt as vbt

_logger = setup_logger(__name__)

class P07Pipeline:
    """
    Core Orchestrator for p07_combined.
    """

    def __init__(self, result_root: Path = Path("results/p07_combined")):
        self.result_root = result_root
        self.db_path = Path("src/ml/pipeline/p07_combined/optuna_study.db")
        self.data_loader = P07DataLoader()
        self.regime_model = P07RegimeModel()

        self.result_root.mkdir(parents=True, exist_ok=True)
        self.db_url = f"sqlite:///{self.db_path}"

    def train_macro_regimes(self, anchor_date: Optional[pd.Timestamp] = None):
        """Train or load the global HMM regime model with an anchor date constraint."""
        vix = self.data_loader.load_vix()
        btc_mc = self.data_loader.load_btc_marketcap()

        macro_df = vix.join(btc_mc, how="outer").ffill().dropna()

        if not macro_df.empty:
            _logger.info("Retraining macro regime model with anchor_date: %s", anchor_date)
            return self.regime_model.train(macro_df, anchor_date=anchor_date)
        return False

    def enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inject global regime context into the ticker DataFrame, ensuring no look-ahead bias."""
        anchor_date = df.index.min()
        success = self.train_macro_regimes(anchor_date=anchor_date)

        if not success:
            _logger.warning("Failed to train regime model for anchor %s. Using default states.", anchor_date)
            df["global_regime"] = 0
            return df

        regimes = self.regime_model.predict(df)
        df["global_regime"] = regimes
        return df

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

    def run_optimization(self, ticker: str, timeframe: str, df_enriched: pd.DataFrame, n_trials: int = 100, start_date: str = "", end_date: str = ""):
        """Execute the optimization loop using Optuna and save artifacts."""
        study_name = f"p07_{ticker}_{timeframe}_{start_date}_{end_date}" if start_date else f"p07_{ticker}_{timeframe}"
        study = optuna.create_study(
            study_name=study_name,
            storage=self.db_url,
            load_if_exists=True,
            direction="maximize"
        )

        if len(study.trials) < n_trials:
            _logger.info("Starting optimization for %s_%s %s (%d trials)...", ticker, timeframe, f"[{start_date}_{end_date}]" if start_date else "", n_trials)
            study.optimize(lambda trial: objective(trial, df_enriched), n_trials=n_trials)
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

        # 1. Run Shared Evaluation
        res = P07Evaluator.run_evaluation(ohlcv_clean, params)
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

        # 4. Save Plots
        viz = P07Visualizer(res_dir)
        viz.plot_tbm_hits(res["y_f"])
        viz.plot_prediction_diagnostics(
            model._map_labels(res["y_test"]).values,
            model.predict_proba(res["X_test"])
        )
        viz.plot_master_overlay(ohlcv_test, plot_sigs, pf)

        _logger.info("Artifacts saved successfully.")
        return True

    def run_batch(self, ticker_files: List[Path], mode: str = "optimize"):
        """Process multiple files with recovery logic."""
        for filepath in ticker_files:
            ticker, timeframe, start_date, end_date = self.data_loader.parse_filename(filepath)
            if not ticker or self.is_completed(ticker, timeframe, start_date, end_date):
                continue

            _logger.info("Processing %s_%s (%s to %s)...", ticker, timeframe, start_date, end_date)
            try:
                df_merged = self.data_loader.get_merged_dataset(filepath)
                df_enriched = self.enrich_data(df_merged)
                if mode == "optimize":
                    self.run_optimization(ticker, timeframe, df_enriched, start_date=start_date, end_date=end_date)
            except Exception as e:
                _logger.error("Failed to process %s_%s: %s", ticker, timeframe, str(e))

if __name__ == "__main__":
    p = P07Pipeline()
    _logger.info("Starting P07 Pipeline Batch...")
    data_dir = Path("data")
    ticker_files = list(data_dir.glob("*_*_*.csv")) # Process all available data files
    if ticker_files:
        p.run_batch(ticker_files)

    # Also run aggregation at the end
    try:
        from src.ml.pipeline.p07_combined.json2csv import aggregate_results
        aggregate_results()
    except Exception as e:
        _logger.error("Failed to aggregate results: %s", e)
