import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, Optional, Dict, Any, List
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

class CointegrationAnalyzer:
    """
    Handles pair identification through cointegration tests and hedge ratio calculation.
    """

    def test_cointegration(self, series_a: pd.Series, series_b: pd.Series) -> Dict:
        """
        Performs the Engle-Granger cointegration test.
        Returns p-value and other stats.
        """
        # Align series
        common_idx = series_a.index.intersection(series_b.index)
        y = series_a.loc[common_idx]
        x = series_b.loc[common_idx]
        
        if len(common_idx) < 100:
             return {"p_value": 1.0, "error": "Insufficient common data"}

        score, pvalue, _ = coint(y, x)
        
        # Calculate Hedge Ratio (Beta) via OLS
        x_const = sm.add_constant(x)
        model = sm.OLS(y, x_const).fit()
        beta = model.params.iloc[1] # hedge ratio
        
        # Calculate residuals (spread)
        spread = y - beta * x
        
        # Calculate Half-Life
        half_life = self.calculate_half_life(spread)
        
        return {
            "p_value": pvalue,
            "t_stat": score,
            "beta": beta,
            "half_life": half_life,
            "n_obs": len(common_idx)
        }

    def walk_forward_backtest(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        n_splits: int = 5,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Walk-forward cointegration backtest.

        For each fold, the hedge ratio is estimated on the training window and the
        mean-reversion strategy is simulated on the held-out test window.  This
        avoids in-sample look-ahead from fitting the hedge ratio on the full series.

        Args:
            series_a: Price series for asset A.
            series_b: Price series for asset B (the "hedge leg").
            n_splits: Number of TimeSeriesSplit folds.
            entry_z: Z-score threshold to open a position.
            exit_z: Z-score threshold to close a position.

        Returns:
            Dict with ``fold_metrics`` (list) and aggregate OOS metrics:
            ``avg_sharpe``, ``avg_total_return``, ``avg_half_life``, ``n_folds``.
        """
        common_idx = series_a.index.intersection(series_b.index)
        a = series_a.loc[common_idx]
        b = series_b.loc[common_idx]

        if len(common_idx) < 100:
            _logger.warning("walk_forward_backtest: fewer than 100 common observations — skipping")
            return {"fold_metrics": [], "avg_sharpe": float("nan"),
                    "avg_total_return": float("nan"), "avg_half_life": float("nan"), "n_folds": 0}

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_metrics: List[Dict[str, Any]] = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(a)):
            if len(train_idx) < 50 or len(test_idx) < 10:
                continue

            a_train, b_train = a.iloc[train_idx], b.iloc[train_idx]
            a_test, b_test = a.iloc[test_idx], b.iloc[test_idx]

            # Estimate hedge ratio on training window only
            x_const = sm.add_constant(b_train)
            ols = sm.OLS(a_train, x_const).fit()
            beta = float(ols.params.iloc[1])
            mu = float(ols.params.iloc[0])

            # Compute spread z-scores on test window (normalised with train stats)
            train_spread = a_train - beta * b_train - mu
            spread_mean = float(train_spread.mean())
            spread_std = float(train_spread.std())
            if spread_std < 1e-10:
                continue

            test_spread = a_test - beta * b_test - mu
            z_scores = (test_spread - spread_mean) / spread_std

            # Simulate a simple mean-reversion strategy
            position = 0  # +1 long spread, -1 short spread, 0 flat
            daily_returns: List[float] = []
            for i in range(1, len(z_scores)):
                z_prev = z_scores.iloc[i - 1]
                # Entry
                if position == 0:
                    if z_prev > entry_z:
                        position = -1  # short spread (expect reversion down)
                    elif z_prev < -entry_z:
                        position = 1   # long spread (expect reversion up)
                # Exit
                elif position == 1 and z_prev >= -exit_z:
                    position = 0
                elif position == -1 and z_prev <= exit_z:
                    position = 0

                spread_ret = test_spread.iloc[i] - test_spread.iloc[i - 1]
                daily_returns.append(position * spread_ret / max(abs(spread_mean), 1e-10))

            if not daily_returns:
                continue

            rets = np.array(daily_returns)
            total_return = float(np.sum(rets))
            sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
            half_life = self.calculate_half_life(test_spread)

            fold_metrics.append({
                "fold": fold_idx,
                "beta": beta,
                "half_life": half_life,
                "total_return": total_return,
                "sharpe": sharpe,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            })
            _logger.debug(
                "WFB fold %d: beta=%.4f, half_life=%.1f, sharpe=%.3f, total_return=%.4f",
                fold_idx, beta, half_life, sharpe, total_return,
            )

        if not fold_metrics:
            return {"fold_metrics": [], "avg_sharpe": float("nan"),
                    "avg_total_return": float("nan"), "avg_half_life": float("nan"), "n_folds": 0}

        avg_sharpe = float(np.mean([f["sharpe"] for f in fold_metrics]))
        avg_total_return = float(np.mean([f["total_return"] for f in fold_metrics]))
        avg_half_life = float(np.mean([f["half_life"] for f in fold_metrics if f["half_life"] < 900]))

        _logger.info(
            "Walk-forward backtest: avg_sharpe=%.3f, avg_return=%.4f, avg_half_life=%.1f over %d folds",
            avg_sharpe, avg_total_return, avg_half_life, len(fold_metrics),
        )
        return {
            "fold_metrics": fold_metrics,
            "avg_sharpe": avg_sharpe,
            "avg_total_return": avg_total_return,
            "avg_half_life": avg_half_life,
            "n_folds": len(fold_metrics),
        }

    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculates the half-life of mean reversion using the Ornstein-Uhlenbeck process.
        """
        spread_lag = spread.shift(1).iloc[1:]
        spread_diff = spread.diff().iloc[1:]
        
        if spread_lag.empty or spread_diff.empty:
            return 999.0

        # Regression of delta(spread) on lag(spread)
        # dS = -lambda * S_lag * dt + noise
        x = sm.add_constant(spread_lag)
        model = sm.OLS(spread_diff, x).fit()
        
        # lambda (mean reversion rate)
        lambda_val = -model.params.iloc[1]
        
        if lambda_val <= 0:
            return 999.0 # Non-stationary / Diverging
        
        half_life = np.log(2) / lambda_val
        return half_life
