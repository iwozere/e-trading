import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from typing import Tuple, Optional, Dict
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
        beta = model.params[1] # hedge ratio
        
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
        lambda_val = -model.params[1]
        
        if lambda_val <= 0:
            return 999.0 # Non-stationary / Diverging
        
        half_life = np.log(2) / lambda_val
        return half_life
