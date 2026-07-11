from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

import vectorbt as vbt
from src.ml.pipeline.p07_combined.models import P07XGBModel
from src.ml.pipeline.p07_combined.robustness import P07RobustnessChecker
from src.ml.pipeline.p08_mtf.data_loader import P08DataLoader
from src.ml.pipeline.p08_mtf.evaluator import P08Evaluator
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class P08RobustnessChecker(P07RobustnessChecker):
    """
    MTF-Aware Robustness Suite for P08.
    Inherits from P07RobustnessChecker and overrides methods for MTF compliance.
    """

    def __init__(self, ticker: str, timeframe: str, res_dir: Path):
        super().__init__(ticker, timeframe, res_dir)
        self.loader = P08DataLoader()

    def run_walk_forward_analysis(self, ohlcv: Any, params: Dict[str, Any], n_windows: int = 5) -> Dict[str, Any]:
        """MTF-Aware WFA: Processes segments safely."""
        _logger.info("Starting P08 WFA (%d windows)...", n_windows)

        # 1. Prepare data using MTF Evaluator
        X_all, y_all = P08Evaluator.prepare_data(ohlcv, params)
        if len(X_all) < 500:
            return {"error": "Insufficient data for P08 WFA"}

        # 2. Segment-based splitting
        total_len = len(X_all)
        oos_size = total_len // (n_windows + 1)
        oos_results = []

        # If ohlcv is a list, we need the full concat version for indexing visuals
        if isinstance(ohlcv, list):
            ohlcv_full = pd.concat(ohlcv).sort_index()
            ohlcv_full = ohlcv_full.loc[~ohlcv_full.index.duplicated(keep="last")]
        else:
            ohlcv_full = ohlcv

        for i in range(n_windows):
            train_end = total_len - (n_windows - i) * oos_size
            test_end = train_end + oos_size

            X_train, y_train = X_all[:train_end], y_all[:train_end]
            X_test, y_test = X_all[train_end:test_end], y_all[train_end:test_end]

            # Train P07-based XGB Model (shares weights logic)
            model = P07XGBModel(
                params={
                    "max_depth": params.get("max_depth", 6),
                    "learning_rate": params.get("learning_rate", 0.1),
                    "n_estimators": params.get("n_estimators", 100),
                }
            )
            model.fit(X_train, y_train)

            signals = model.predict_signal(
                X_test,
                thresholds={
                    "buy_prob_min": params.get("buy_prob_min", 0.5),
                    "sell_prob_min": params.get("sell_prob_min", 0.5),
                },
            )

            ohlcv_test = ohlcv_full.loc[X_test.index]
            pf = vbt.Portfolio.from_signals(
                ohlcv_test["close"],
                signals == 1,
                signals == -1,
                fees=0.001,
                slippage=0.0005,
                freq=self.timeframe,
                direction="both",
            )

            oos_results.append(
                {
                    # vectorbt attaches metric accessors dynamically; not visible to pyright
                    "sharpe": pf.sharpe_ratio(),  # pyright: ignore[reportAttributeAccessIssue]
                    "return": pf.total_return(),
                    "window": i + 1,
                    "pf": pf,
                }
            )

        combined_equity = pd.concat([res["pf"].value() for res in oos_results]).sort_index()
        avg_oos_sharpe = np.mean([res["sharpe"] for res in oos_results if not np.isnan(res["sharpe"])])

        return {"oos_results": oos_results, "combined_equity": combined_equity, "avg_oos_sharpe": avg_oos_sharpe}

    def run_parameter_sensitivity(self, ohlcv: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """MTF-Aware Sensitivity: Perturbs MTF specific params."""
        _logger.info("Starting P08 MTF Sensitivity...")
        results = []

        # P08 specific perturbations
        perturbations = [
            {"name": "Original", "pt_mult": 0, "sl_mult": 0, "anchor_ema": 0},
            {"name": "PT High", "pt_mult": 0.5, "sl_mult": 0, "anchor_ema": 0},
            {"name": "PT Low", "pt_mult": -0.5, "sl_mult": 0, "anchor_ema": 0},
            {"name": "Anchor EMA+5", "pt_mult": 0, "sl_mult": 0, "anchor_ema": 5},
            {"name": "Anchor EMA-5", "pt_mult": 0, "sl_mult": 0, "anchor_ema": -5},
        ]

        for p in perturbations:
            trial_params = params.copy()
            trial_params["pt_mult"] = max(0.5, trial_params.get("pt_mult", 2.0) + p["pt_mult"])
            trial_params["anchor_ema_period"] = max(5, trial_params.get("anchor_ema_period", 20) + p["anchor_ema"])

            res = P08Evaluator.run_evaluation(ohlcv, trial_params, timeframe=self.timeframe)
            if "error" not in res:
                results.append(
                    {"perturbation": p["name"], "sharpe": res["pf"].sharpe_ratio(), "return": res["pf"].total_return()}
                )

        return {"sensitivity_results": results}

    def run_all_checks(self, ohlcv: Any, params: Dict[str, Any], backtest_res: Dict[str, Any]):
        """Ported from P07: Runs full suite and saves results."""
        _logger.info("Running full P08 MTF robustness suite for %s %s", self.ticker, self.timeframe)

        results: Dict[str, Any] = {}

        # 1. Walk Forward
        wfa = self.run_walk_forward_analysis(ohlcv, params)
        if "error" not in wfa:
            results["wfa"] = {k: v for k, v in wfa.items() if k not in ["oos_results", "combined_equity"]}
            results["wfa_details"] = [
                {"window": r["window"], "sharpe": r["sharpe"], "return": r["return"]}
                for r in wfa.get("oos_results", [])
            ]
            if "combined_equity" in wfa:
                wfa["combined_equity"].to_json(self.res_dir / "wfa_equity.json")

        # 2. Monte Carlo (Inherited from P07 works with P08 Portfolio)
        mc = self.run_monte_carlo(backtest_res["pf"])
        results["monte_carlo"] = mc

        # 3. Sensitivity
        sens = self.run_parameter_sensitivity(ohlcv, params)
        results["sensitivity"] = sens

        # Save summary JSON
        import json

        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(self.res_dir / "robustness_summary.json", "w") as f:
            json.dump(results, f, indent=4, cls=NpEncoder)

        _logger.info("P08 Robustness completed. Results in %s", self.res_dir)
        return results


if __name__ == "__main__":
    pass
