import pandas as pd
import numpy as np
import vectorbt as vbt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from src.ml.pipeline.p07_combined.evaluator import P07Evaluator
from src.ml.pipeline.p07_combined.models import P07XGBModel
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class P07RobustnessChecker:
    """
    Suite of robustness tests for P07 strategies.
    """

    def __init__(self, ticker: str, timeframe: str, res_dir: Path):
        self.ticker = ticker
        self.timeframe = timeframe
        self.res_dir = res_dir
        self.res_dir.mkdir(parents=True, exist_ok=True)

    def run_walk_forward_analysis(self, ohlcv: pd.DataFrame, params: Dict[str, Any], n_windows: int = 5) -> Dict[str, Any]:
        """
        Performs an Anchored Walk-Forward Analysis.
        Trains on increasing windows and tests on the subsequent out-of-sample (OOS) period.
        """
        _logger.info("Starting Walk-Forward Analysis with %d windows...", n_windows)

        # Prepare all data first
        X_all, y_all = P07Evaluator.prepare_data(ohlcv, params)
        if len(X_all) < 500:
            return {"error": "Insufficient data for WFA"}

        # Calculate window sizes
        total_len = len(X_all)
        oos_size = total_len // (n_windows + 1)

        oos_results = []

        for i in range(n_windows):
            train_end = total_len - (n_windows - i) * oos_size
            test_end = train_end + oos_size

            X_train, y_train = X_all[:train_end], y_all[:train_end]
            X_test, y_test = X_all[train_end:test_end], y_all[train_end:test_end]

            _logger.info("WFA Window %d: Train %d samples, Test %d samples", i+1, len(X_train), len(X_test))

            # Train model on this window
            xgb_params = {
                'max_depth': params.get('max_depth', 6),
                'learning_rate': params.get('learning_rate', 0.1),
                'n_estimators': params.get('n_estimators', 100)
            }
            model = P07XGBModel(params=xgb_params)
            model.fit(X_train, y_train)

            # Predict and evaluate OOS
            thresholds = {
                'buy_prob_min': params.get('buy_prob_min', 0.5),
                'sell_prob_min': params.get('sell_prob_min', 0.5)
            }
            signals = model.predict_signal(X_test, thresholds=thresholds)

            # Backtest OOS segment
            ohlcv_test = ohlcv.loc[X_test.index]
            pf = vbt.Portfolio.from_signals(
                ohlcv_test['close'],
                signals == 1,
                signals == -1,
                fees=0.001,
                slippage=0.0005,
                freq=self.timeframe,
                direction='both'
            )

            oos_results.append({
                'window': i + 1,
                'sharpe': pf.sharpe_ratio(),
                'return': pf.total_return(),
                'pf': pf
            })

        # Combine OOS result series for overall WFA equity
        equity_curves = [res['pf'].value() for res in oos_results]
        # Normalize and stitch
        combined_equity = pd.concat(equity_curves).sort_index()
        # We need to handle the overlap/continuity if necessary, but concat handles index

        # Calculate Walk-Forward Efficiency (WFE)
        # Simplified: Avg OOS Sharpe / IS Sharpe of best trial (not perfect but indicative)
        avg_oos_sharpe = np.mean([res['sharpe'] for res in oos_results if not np.isnan(res['sharpe'])])

        return {
            'oos_results': oos_results,
            'combined_equity': combined_equity,
            'avg_oos_sharpe': avg_oos_sharpe
        }

    def run_monte_carlo(self, pf: Any, n_sims: int = 100) -> Dict[str, Any]:
        """
        Performs Monte Carlo simulations by shuffling trade returns and randomly skipping trades.
        """
        try:
            trades = pf.trades
            if trades.count().sum() < 5:
                return {"error": "Insufficient trades for MC"}

            # Use trade-level returns (percentage)
            returns = trades.returns.values
        except Exception as e:
            _logger.error("Failed to extract trade returns for MC: %s", e)
            return {"error": str(e)}

        # 1. Shuffle Returns Simulation
        shuffled_final_values = []
        for _ in range(n_sims):
            if len(returns) > 0:
                shuffled_rets = np.random.choice(returns, size=len(returns), replace=True)
                equity = (1 + shuffled_rets).prod()
            else:
                equity = 1.0
            shuffled_final_values.append(equity)

        # 2. Skip Trades Simulation (90% survival)
        skipped_final_values = []
        for _ in range(n_sims):
            mask = np.random.random(len(returns)) > 0.1 # 10% chance to skip
            skipped_rets = returns[mask]
            equity = (1 + skipped_rets).prod()
            skipped_final_values.append(equity)

        return {
            'shuffled_returns': shuffled_final_values,
            'random_skips': skipped_final_values,
            'positivity_rate': np.mean(np.array(shuffled_final_values) > 1.0)
        }

    def run_parameter_sensitivity(self, ohlcv: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks performance sensitivity by perturbing key parameters.
        """
        _logger.info("Starting Parameter Sensitivity Analysis...")

        results = []
        # Perturb PT/SL and Indicators
        perturbations = [
            {'name': 'Original', 'pt_mult': 0, 'sl_mult': 0, 'rsi_period': 0},
            {'name': 'PT High', 'pt_mult': 0.5, 'sl_mult': 0, 'rsi_period': 0},
            {'name': 'PT Low', 'pt_mult': -0.5, 'sl_mult': 0, 'rsi_period': 0},
            {'name': 'SL High', 'pt_mult': 0, 'sl_mult': 0.5, 'rsi_period': 0},
            {'name': 'SL Low', 'pt_mult': 0, 'sl_mult': -0.5, 'rsi_period': 0},
            {'name': 'RSI+2', 'pt_mult': 0, 'sl_mult': 0, 'rsi_period': 2},
            {'name': 'RSI-2', 'pt_mult': 0, 'sl_mult': 0, 'rsi_period': -2},
        ]

        for p in perturbations:
            trial_params = params.copy()
            trial_params['pt_mult'] = max(0.5, trial_params.get('pt_mult', 2.0) + p['pt_mult'])
            trial_params['sl_mult'] = max(0.25, trial_params.get('sl_mult', 1.0) + p['sl_mult'])
            trial_params['rsi_period'] = max(5, trial_params.get('rsi_period', 14) + p['rsi_period'])

            res = P07Evaluator.run_evaluation(ohlcv, trial_params, timeframe=self.timeframe)
            if "error" not in res:
                results.append({
                    'perturbation': p['name'],
                    'sharpe': res['pf'].sharpe_ratio(),
                    'return': res['pf'].total_return()
                })

        return {'sensitivity_results': results}

    def run_all_checks(self, ohlcv: pd.DataFrame, params: Dict[str, Any], backtest_res: Dict[str, Any]):
        """Runs the full robustness suite and saves results."""
        _logger.info("Running full robustness suite for %s %s", self.ticker, self.timeframe)

        results = {}

        # 1. Walk Forward
        wfa = self.run_walk_forward_analysis(ohlcv, params)
        results['wfa'] = {k: v for k, v in wfa.items() if k != 'oos_results' and k != 'combined_equity'}
        if 'combined_equity' in wfa:
            wfa['combined_equity'].to_json(self.res_dir / "wfa_equity.json")

        # 2. Monte Carlo
        mc = self.run_monte_carlo(backtest_res['pf'])
        results['monte_carlo'] = mc

        # 3. Sensitivity
        sens = self.run_parameter_sensitivity(ohlcv, params)
        results['sensitivity'] = sens

        # Save summary JSON
        import json
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NpEncoder, self).default(obj)

        with open(self.res_dir / "robustness_summary.json", "w") as f:
            json.dump(results, f, indent=4, cls=NpEncoder)

        _logger.info("Robustness checks completed for %s %s. Results saved to %s", self.ticker, self.timeframe, self.res_dir)
        return results

if __name__ == "__main__":
    # Example test
    pass
