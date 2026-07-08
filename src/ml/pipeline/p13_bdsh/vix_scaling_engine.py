import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from .models import P13Config

logger = logging.getLogger(__name__)


class VIXScalingEngine:
    def __init__(self, config: P13Config):
        self.config = config
        self.z_lookback = config.vix_lookback
        self.entry_tiers = config.entry_tiers
        self.exit_threshold = config.exit_z_threshold
        self.initial_capital = config.initial_capital
        self.slippage = config.slippage_pct

    def _compute_target_exposure(self, z: float) -> float:
        """Helper to calculate target exposure based on Z-score tiers."""
        if pd.isna(z) or z < self.exit_threshold:
            return 0.0

        # Entry Logic - Multi-stage
        # Sort tiers by threshold to check correctly
        sorted_tiers = sorted(self.entry_tiers.values(), key=lambda x: x["z_threshold"])
        accumulated_allocation = 0.0
        for tier in sorted_tiers:
            if z > tier["z_threshold"]:
                accumulated_allocation += tier["allocation"]

        return min(1.0, accumulated_allocation)

    def calculate_vix_zscore(self, vix_series: pd.Series) -> pd.Series:
        """
        Calculates Z-Score manually using Rolling Mean and Std.
        Z = (VIX - MA) / StdDev
        """
        rolling_mean = vix_series.rolling(window=self.z_lookback).mean()
        rolling_std = vix_series.rolling(window=self.z_lookback).std()

        z_score = (vix_series - rolling_mean) / rolling_std
        return z_score

    def generate_signals(self, z_series: pd.Series) -> pd.Series:
        """
        Generates Target Exposure signals (0.0 to 1.0) based on VIX.
        Note: This only considers VIX. Stop-loss overrides happen in the backtest loop.
        """
        exposure = pd.Series(0.0, index=z_series.index)
        current_vix_exposure = 0.0

        for i in range(len(z_series)):
            z = z_series.iloc[i]

            # Use deduplicated logic
            target = self._compute_target_exposure(z)

            # Use target to set current exposure if it’s more aggressive
            if target > current_vix_exposure:
                current_vix_exposure = target
            elif z < self.exit_threshold:
                current_vix_exposure = 0.0

            exposure.iloc[i] = current_vix_exposure

        return exposure

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculates Average True Range (ATR).
        ATR = EMA(True Range, period)
        True Range = Max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()  # Using Simple Moving Average for ATR as per modern standard, or EMA?
        # Many use EMA for ATR. Let's use SMA to keep it simple and consistent with rolling mean Z-score.
        return atr

    def run_backtest(self, ticker_df: pd.DataFrame, z_series: pd.Series) -> pd.DataFrame:
        """
        Runs the backtest simulation with path-dependent ATR stop-loss and cooldown logic.
        R5: ATR Stop-Loss (2x ATR from Avg Acquisition Price).
        """
        atr_multiplier = self.config.atr_multiplier
        atr_period = self.config.atr_period

        atr_series = self.calculate_atr(ticker_df, atr_period)
        price_series = ticker_df["close"]

        # State variables
        current_capital = self.initial_capital
        exposure = 0.0  # Current weight in asset
        avg_entry_price = 0.0
        current_stop_loss = 0.0
        in_cooldown = False

        results_data = []
        markers: dict[str, list[Any]] = {"buy": [], "sell": [], "stop_loss": []}

        for i in range(len(price_series)):
            current_date = price_series.index[i]
            current_price = price_series.iloc[i]
            z = z_series.iloc[i]
            atr = atr_series.iloc[i]

            # 1. Check Stop-Loss (Path-dependent)
            stop_loss_triggered = False
            if exposure > 0 and current_stop_loss > 0:
                if current_price < current_stop_loss:
                    stop_loss_triggered = True
                    in_cooldown = True
                    # Exit at current price (with slippage)
                    exit_value = (current_capital * exposure) * (1 - self.slippage)
                    current_capital = (current_capital * (1 - exposure)) + exit_value
                    exposure = 0.0
                    avg_entry_price = 0.0
                    current_stop_loss = 0.0
                    markers["stop_loss"].append(current_date)
                    logger.info(
                        f"ATR Stop-loss triggered for {current_date} at {current_price:.2f} (SL: {current_stop_loss:.2f})"
                    )

            # 2. Update Cooldown State
            if in_cooldown and z < self.exit_threshold:
                in_cooldown = False
                logger.info(f"Cooldown reset for {current_date}")

            # 3. VIX Signal Logic (based on YESTERDAY'S Z-score)
            if i > 0:
                prev_z = z_series.iloc[i - 1]
                target_exposure = 0.0

                if not in_cooldown:
                    target_exposure = self._compute_target_exposure(prev_z)

                if target_exposure < exposure:  # Exit or Reduce
                    cost = abs(target_exposure - exposure) * current_capital * self.slippage
                    current_capital -= cost
                    if target_exposure == 0 and exposure > 0:
                        markers["sell"].append(current_date)
                    exposure = target_exposure
                    if exposure == 0:
                        avg_entry_price = 0.0
                        current_stop_loss = 0.0

                elif target_exposure > exposure and not in_cooldown:  # Increase / Scaling
                    added_exposure = target_exposure - exposure
                    cost = added_exposure * current_capital * self.slippage
                    current_capital -= cost

                    # Update Avg Entry Price
                    if exposure == 0:
                        avg_entry_price = current_price
                    else:
                        avg_entry_price = (
                            exposure * avg_entry_price + added_exposure * current_price
                        ) / target_exposure

                    # R5 Logic: Set/Update Stop-Loss based on ATR at entry
                    # If we already have a stop-loss, we might want to update it for the new weighted average?
                    # "Average Acquisition Price minus 2x ATR (measured at the time of entry/scaling)"
                    if not pd.isna(atr):
                        current_stop_loss = avg_entry_price - (atr_multiplier * atr)

                    exposure = target_exposure
                    markers["buy"].append(current_date)

            # 4. Daily Portfolio Value Update
            if i > 0:
                prev_price = price_series.iloc[i - 1]
                daily_ret = (current_price - prev_price) / prev_price
                current_capital *= 1 + daily_ret * exposure

            results_data.append(
                {
                    "Date": current_date,
                    "Price": current_price,
                    "Z_Score": z,
                    "ATR": atr,
                    "Target_Exposure": exposure,
                    "Avg_Entry_Price": avg_entry_price,
                    "Stop_Loss_Price": current_stop_loss,
                    "Portfolio_Value": current_capital,
                    "In_Cooldown": in_cooldown,
                }
            )

        results = pd.DataFrame(results_data).set_index("Date")
        # Add a placeholder for cumulative returns for metrics compatibility
        results["Cumulative_Return_Net"] = results["Portfolio_Value"] / self.initial_capital
        results["Strategy_Return_Net"] = results["Cumulative_Return_Net"].pct_change().fillna(0.0)

        # Store markers for plotting
        self.markers = markers
        return results

    def calculate_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Calculates Sharpe Ratio and MDD."""
        returns = results["Strategy_Return_Net"]

        # Sharpe Ratio (Annualized, assuming 252 trading days, 0 risk-free rate)
        if returns.std() == 0:
            sharpe = 0.0
        else:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

        # Maximum Drawdown
        cum_ret = results["Cumulative_Return_Net"]
        rolling_max = cum_ret.cummax()
        drawdown = cum_ret / rolling_max - 1.0
        mdd = drawdown.min()

        # Total Return
        total_return = results["Cumulative_Return_Net"].iloc[-1] - 1.0

        return {"Total Return": total_return, "Annualized Sharpe Ratio": sharpe, "Max Drawdown": mdd}


