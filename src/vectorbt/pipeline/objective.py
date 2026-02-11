import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src.vectorbt.pipeline.engine import StrategyEngine
from src.notification.logger import setup_logger

_logger = setup_logger(__name__, use_multiprocessing=True)

class Objective:
    """
    Dynamic Objective function for Optuna optimization.
    Parses strategy configuration from JSON and evaluates signals via StrategyEngine.
    """

    def __init__(self, data_splits: List[pd.DataFrame], strategy_config: Dict[str, Any], funding_rate: float = 0.0001):
        """
        Initialize the objective.

        Args:
            data_splits: List of splits.
            strategy_config: Dynamic strategy configuration dict.
            funding_rate: Periodic funding rate (proxy).
        """
        self.data_splits = data_splits
        self.strategy_config = strategy_config
        self.funding_rate = funding_rate
        self.engine = StrategyEngine(strategy_config)

    def __call__(self, trial):
        # 1. Suggest parameters from JSON configuration
        params = {}

        # Suggest leverage (global parameter)
        params["leverage"] = trial.suggest_float("leverage", 1.0, 10.0, step=1.0)

        # Dynamic search space for each indicator
        for ind_id, ind_cfg in self.strategy_config.get("indicators", {}).items():
            space = ind_cfg.get("space", {})
            for p_name, p_cfg in space.items():
                p_type = p_cfg.get("type", "int")
                p_min = p_cfg.get("min")
                p_max = p_cfg.get("max")
                p_key = f"{ind_id}_{p_name}"

                if p_type == "int":
                    params[p_key] = trial.suggest_int(p_key, p_min, p_max)
                elif p_type == "float":
                    step = p_cfg.get("step")
                    params[p_key] = trial.suggest_float(p_key, p_min, p_max, step=step)
                elif p_type == "categorical":
                    params[p_key] = trial.suggest_categorical(p_key, p_cfg.get("choices"))

        leverage = params["leverage"]

        split_scores = []
        all_trades_count = 0
        all_max_dd = []

        for data in self.data_splits:
            # CRITICAL: Create a deep copy for thread safety
            # Multiple parallel trials can mutate shared data structures
            data = data.copy(deep=True)
            # 2. Generate signals via Dynamic Engine
            # Pass full OHLC for indicators like ADX/ATR
            close = data.xs('Close', level='column', axis=1)
            res_full = self.engine.run(data, params)
            res = res_full["signals"]
            results = res_full["results"]

            # 3. Trailing Stop Support
            # Check if strategy config or params has sl_trailing
            sl_stop = None
            sl_trail = False
            if "exits" in self.strategy_config and "sl_trailing" in self.strategy_config["exits"]:
                sl_cfg = self.strategy_config["exits"]["sl_trailing"]
                # Resolve multiplier
                if "multiplier" in sl_cfg.get("space", {}):
                    multiplier = params.get(f"sl_trailing_multiplier", 2.0)
                else:
                    multiplier = sl_cfg.get("params", {}).get("multiplier", 2.0)

                # Dynamic ATR-based stop
                if "indicator" in sl_cfg:
                    ind_id = sl_cfg["indicator"]
                    field = sl_cfg.get("field", "out")
                    if ind_id in results:
                        atr_val = results[ind_id].get_field(field)
                        # vbt sl_stop is often a ratio: distance / price
                        sl_stop = (atr_val * multiplier) / close
                        sl_trail = True
                else:
                    # Fixed percentage stop
                    sl_stop = multiplier / 100.0
                    sl_trail = True
            # Settings for Binance Futures:
            # fees=0.0004 (VIP0 taker), slippage=0.0001
            # cash_sharing=True (cross-margin)
            # To simulate portfolio-wide leverage with cash_sharing=True in vbt-core,
            # we use size_type='Value' and calculate the target dollar amount.
            init_cash = 1000.0
            n_assets = len(close.columns)
            # Allocation per asset in dollars
            target_value = (init_cash * leverage) / n_assets

            pf = vbt.Portfolio.from_signals(
                close,
                entries=res['entries'],
                exits=res['exits'],
                short_entries=res['short_entries'],
                short_exits=res['short_exits'],
                fees=0.0004,
                slippage=0.0001,
                size=target_value,
                size_type='Value',
                cash_sharing=True,
                init_cash=init_cash,
                sl_stop=sl_stop,
                sl_trail=sl_trail,
                freq='15m'
            )

            # 4. Custom Liquidation Proxy (Senior Architect Requirement)
            # Discard if Max Drawdown > 60% (conservative threshold)
            max_dd = pf.max_drawdown().max() # Max across all assets/columns
            all_max_dd.append(max_dd)
            if max_dd > 0.6:
                split_scores.append(-1.0) # Penalty for this split
                continue

            # 5. Production-grade Scoring per split
            # Score = CAGR * Calmar * WinRate / (Leverage^2)
            cagr = pf.annualized_return().mean()
            calmar = pf.calmar_ratio().mean()
            win_rate = pf.trades.win_rate().mean()
            all_trades_count += pf.trades.count().sum()

            if np.isnan(cagr) or np.isnan(calmar) or np.isnan(win_rate):
                split_scores.append(-1.0)
            else:
                s = (cagr * calmar * win_rate) / (leverage ** 2)
                split_scores.append(s)

        # 6. Stability Score (Senior Architect Requirement)
        # FinalScore = median(split_scores) - std(split_scores)
        if not split_scores:
            return -1e6

        final_score = np.median(split_scores) - np.std(split_scores)

        # 7. Minimum Trades Filter (Guardroom Requirement)
        if all_trades_count < 20:
            return -1e6

        # Store metadata
        trial.set_user_attr("total_trades", int(all_trades_count))
        trial.set_user_attr("avg_max_drawdown", float(np.mean(all_max_dd)))
        trial.set_user_attr("stability_score", float(final_score))

        return final_score
