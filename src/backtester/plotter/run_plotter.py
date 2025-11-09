"""
Run Plotter Module

This module provides functionality to create plots from optimization results.
It handles:
1. Reading JSON files from results folder
2. Loading corresponding data files
3. Creating plots with indicators and trades
4. Saving plots as PNG files

Features:
- Dynamic indicator selection based on strategy mixins
- Configurable subplot layout
- Trade visualization with buy/sell markers
- Equity curve calculation from trades
- Support for multiple output formats
"""

import glob
import json
import os
import sys
from typing import Dict, List, Tuple
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class ResultPlotter:
    """Main class for creating plots from optimization results"""

    def __init__(self, config_path: str = "config/optimizer/optimizer.json"):
        """
        Initialize the plotter

        Args:
            config_path: Path to the optimizer configuration file
        """
        self.config = self._load_config(config_path)
        self.mixin_config = self._load_mixin_config()
        self.vis_settings = self.config.get("visualization_settings", {})

    def _load_config(self, config_path: str) -> Dict:
        """Load optimizer configuration"""
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception:
            _logger.exception("Error loading config from %s: %s")
            return {}

    def _load_mixin_config(self) -> Dict:
        """Load mixin indicators configuration from unified config"""
        try:
            with open("config/indicators.json", "r") as f:
                config = json.load(f)
                plotter_config = config.get("plotter_config", {})
                return plotter_config
        except Exception as e:
            _logger.exception("Error loading mixin config from unified indicators config: %s", e)
            return {"entry_mixins": {}, "exit_mixins": {}}

    def get_json_files(self, results_dir: str = "results") -> List[str]:
        """Get all JSON files from results directory"""
        pattern = os.path.join(results_dir, "*.json")
        json_files = glob.glob(pattern)
        _logger.info("Found %d JSON files in %s", len(json_files), results_dir)
        return json_files

    def load_result_data(self, json_file: str) -> Dict:
        """Load data from JSON result file"""
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            _logger.debug("Loaded result data from %s", json_file)
            return data
        except Exception:
            _logger.exception("Error loading JSON file %s: %s")
            return {}

    def load_price_data(self, data_file: str) -> pd.DataFrame:
        """Load price data from CSV file"""
        try:
            csv_path = os.path.join("data", data_file)
            df = pd.read_csv(csv_path)

            # Convert timestamp to datetime
            df["datetime"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("datetime", ascending=True)
            df.set_index("datetime", inplace=True)

            # Select OHLCV columns
            df = df[["open", "high", "low", "close", "volume"]]

            # Convert to numeric
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df.ffill(inplace=True)
            df.bfill(inplace=True)

            _logger.debug("Loaded price data from %s", csv_path)
            return df

        except Exception:
            _logger.exception("Error loading price data from %s: %s")
            return pd.DataFrame()

    def get_indicators_for_strategy(self, result_data: Dict) -> List[str]:
        """Get list of indicators based on strategy mixins"""
        indicators = set()

        # Get entry and exit mixin names
        best_params = result_data.get("best_params", {})
        entry_logic = best_params.get("entry_logic", {})
        exit_logic = best_params.get("exit_logic", {})

        entry_mixin_name = entry_logic.get("name", "")
        exit_mixin_name = exit_logic.get("name", "")

        # Add indicators from entry mixin
        if entry_mixin_name in self.mixin_config.get("entry_mixins", {}):
            entry_indicators = self.mixin_config["entry_mixins"][entry_mixin_name].get(
                "indicators", []
            )
            indicators.update(entry_indicators)

        # Add indicators from exit mixin
        if exit_mixin_name in self.mixin_config.get("exit_mixins", {}):
            exit_indicators = self.mixin_config["exit_mixins"][exit_mixin_name].get(
                "indicators", []
            )
            indicators.update(exit_indicators)

        _logger.debug("Strategy uses indicators: %s", list(indicators))
        return list(indicators)

    def get_subplot_layout(self, indicators: List[str]) -> Dict[str, str]:
        """Get subplot layout for indicators"""
        layout = {}

        # Default layout: RSI and Volume get separate subplots, others overlay on price
        for indicator in indicators:
            if indicator in ["rsi", "volume", "atr"]:
                layout[indicator] = "separate"
            else:
                layout[indicator] = "overlay"

        return layout

    def calculate_indicators(
        self, df: pd.DataFrame, indicators: List[str], strategy_params: Dict
    ) -> Dict:
        """Calculate indicators based on strategy parameters"""
        calculated_indicators = {}

        for indicator in indicators:
            try:
                if indicator == "rsi":
                    period = self._get_param_value(strategy_params, "rsi_period", 14)
                    calculated_indicators["rsi"] = self._calculate_rsi(
                        df["close"], period
                    )

                elif indicator == "bollinger_bands":
                    period = self._get_param_value(strategy_params, "bb_period", 20)
                    std_dev = self._get_param_value(strategy_params, "bb_std", 2)
                    calculated_indicators["bollinger_bands"] = (
                        self._calculate_bollinger_bands(df["close"], period, std_dev)
                    )

                elif indicator == "ichimoku":
                    tenkan_period = self._get_param_value(
                        strategy_params, "tenkan_period", 9
                    )
                    kijun_period = self._get_param_value(
                        strategy_params, "kijun_period", 26
                    )
                    senkou_span_b_period = self._get_param_value(
                        strategy_params, "senkou_span_b_period", 52
                    )
                    calculated_indicators["ichimoku"] = self._calculate_ichimoku(
                        df, tenkan_period, kijun_period, senkou_span_b_period
                    )

                elif indicator == "supertrend":
                    period = self._get_param_value(
                        strategy_params, "supertrend_period", 10
                    )
                    multiplier = self._get_param_value(
                        strategy_params, "supertrend_multiplier", 3
                    )
                    calculated_indicators["supertrend"] = self._calculate_supertrend(
                        df, period, multiplier
                    )

                elif indicator == "volume":
                    calculated_indicators["volume"] = df["volume"]

                elif indicator == "atr":
                    period = self._get_param_value(strategy_params, "atr_period", 14)
                    calculated_indicators["atr"] = self._calculate_atr(df, period)

            except Exception as e:
                _logger.warning("Error calculating %s: %s", indicator, e)

        return calculated_indicators

    def _get_param_value(
        self, strategy_params: Dict, param_name: str, default: float
    ) -> float:
        """Get parameter value from strategy parameters"""
        # Check entry logic params
        entry_params = strategy_params.get("entry_logic", {}).get("params", {})
        if param_name in entry_params:
            return entry_params[param_name]

        # Check exit logic params
        exit_params = strategy_params.get("exit_logic", {}).get("params", {})
        if param_name in exit_params:
            return exit_params[param_name]

        return default

    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int, std_dev: float
    ) -> Dict:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return {"upper": upper_band, "middle": sma, "lower": lower_band}

    def _calculate_ichimoku(
        self,
        df: pd.DataFrame,
        tenkan_period: int,
        kijun_period: int,
        senkou_span_b_period: int,
    ) -> Dict:
        """Calculate Ichimoku Cloud indicators"""
        high = df["high"]
        low = df["low"]

        tenkan = (
            high.rolling(window=tenkan_period).max()
            + low.rolling(window=tenkan_period).min()
        ) / 2
        kijun = (
            high.rolling(window=kijun_period).max()
            + low.rolling(window=kijun_period).min()
        ) / 2
        senkou_span_a = ((tenkan + kijun) / 2).shift(kijun_period)
        senkou_span_b = (
            (
                high.rolling(window=senkou_span_b_period).max()
                + low.rolling(window=senkou_span_b_period).min()
            )
            / 2
        ).shift(kijun_period)

        return {
            "tenkan": tenkan,
            "kijun": kijun,
            "senkou_span_a": senkou_span_a,
            "senkou_span_b": senkou_span_b,
        }

    def _calculate_supertrend(
        self, df: pd.DataFrame, period: int, multiplier: float
    ) -> pd.Series:
        """Calculate SuperTrend indicator"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Calculate ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Calculate SuperTrend
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)

        for i in range(1, len(df)):
            if close.iloc[i] > upper_band.iloc[i - 1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]

            if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i - 1]:
                lower_band.iloc[i] = lower_band.iloc[i - 1]
            if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i - 1]:
                upper_band.iloc[i] = upper_band.iloc[i - 1]

            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]

        return supertrend

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def calculate_equity_curve(
        self, trades: List[Dict], initial_capital: float = 1000.0
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate equity curve from trades"""
        if not trades:
            return pd.Series(), pd.Series()

        # Sort trades by exit time
        sorted_trades = sorted(trades, key=lambda x: x.get("exit_time", ""))

        equity_values = [initial_capital]
        equity_times = [
            pd.Timestamp(sorted_trades[0].get("entry_time", "")) - pd.Timedelta(days=1)
        ]

        current_equity = initial_capital

        for trade in sorted_trades:
            if trade.get("net_pnl") is not None:
                current_equity += trade["net_pnl"]
                equity_values.append(current_equity)
                equity_times.append(pd.Timestamp(trade["exit_time"]))

        return pd.Series(equity_values, index=equity_times), pd.Series(
            equity_values, index=equity_times
        )

    def create_plot(
        self,
        df: pd.DataFrame,
        indicators: Dict,
        trades: List[Dict],
        equity_curve: pd.Series,
        json_file: str,
        result_data: Dict,
    ) -> None:
        """Create and save the plot"""

        # Get indicators to plot
        strategy_indicators = self.get_indicators_for_strategy(result_data)
        subplot_layout = self.get_subplot_layout(strategy_indicators)

        # Calculate number of subplots
        n_subplots = 1  # Price subplot
        for indicator in strategy_indicators:
            if subplot_layout.get(indicator) == "separate":
                n_subplots += 1
        if self.vis_settings.get("show_equity_curve", True):
            n_subplots += 1

        # Create figure
        plot_size = self.vis_settings.get("plot_size", [15, 10])
        fig, axes = plt.subplots(
            n_subplots,
            1,
            figsize=plot_size,
            gridspec_kw={"height_ratios": [3] + [1] * (n_subplots - 1)},
        )
        if n_subplots == 1:
            axes = [axes]

        # Set title
        strategy_name = f"{result_data.get('best_params', {}).get('entry_logic', {}).get('name', 'Unknown')} + {result_data.get('best_params', {}).get('exit_logic', {}).get('name', 'Unknown')}"
        fig.suptitle(
            f"Strategy: {strategy_name}\n{os.path.basename(json_file)}",
            fontsize=self.vis_settings.get("font_size", 12),
        )

        # Plot price and overlay indicators
        ax_price = axes[0]
        ax_price.plot(
            df.index, df["close"], label="Close Price", linewidth=1, color="black"
        )

        # Plot overlay indicators on price chart
        for indicator_name, indicator_data in indicators.items():
            if subplot_layout.get(indicator_name) == "overlay":
                if indicator_name == "bollinger_bands":
                    ax_price.plot(
                        df.index,
                        indicator_data["upper"],
                        "--",
                        label="BB Upper",
                        alpha=0.7,
                    )
                    ax_price.plot(
                        df.index,
                        indicator_data["middle"],
                        "-",
                        label="BB Middle",
                        alpha=0.7,
                    )
                    ax_price.plot(
                        df.index,
                        indicator_data["lower"],
                        "--",
                        label="BB Lower",
                        alpha=0.7,
                    )
                elif indicator_name == "ichimoku":
                    ax_price.plot(
                        df.index, indicator_data["tenkan"], label="Tenkan", alpha=0.7
                    )
                    ax_price.plot(
                        df.index, indicator_data["kijun"], label="Kijun", alpha=0.7
                    )
                    ax_price.plot(
                        df.index,
                        indicator_data["senkou_span_a"],
                        label="Senkou Span A",
                        alpha=0.7,
                    )
                    ax_price.plot(
                        df.index,
                        indicator_data["senkou_span_b"],
                        label="Senkou Span B",
                        alpha=0.7,
                    )
                elif indicator_name == "supertrend":
                    ax_price.plot(
                        df.index, indicator_data, label="SuperTrend", alpha=0.7
                    )

        # Plot trades
        self._plot_trades(ax_price, trades)

        ax_price.set_ylabel("Price")
        ax_price.legend()
        ax_price.grid(self.vis_settings.get("show_grid", True))

        # Plot separate indicators
        subplot_idx = 1
        for indicator_name, indicator_data in indicators.items():
            if subplot_layout.get(indicator_name) == "separate":
                if subplot_idx < len(axes):
                    ax = axes[subplot_idx]

                    if indicator_name == "rsi":
                        ax.plot(df.index, indicator_data, label="RSI", color="purple")
                        ax.axhline(y=70, color="r", linestyle="--", alpha=0.5)
                        ax.axhline(y=30, color="g", linestyle="--", alpha=0.5)
                        ax.set_ylabel("RSI")
                        ax.set_ylim(0, 100)

                    elif indicator_name == "volume":
                        ax.bar(
                            df.index,
                            indicator_data,
                            label="Volume",
                            alpha=0.7,
                            color="blue",
                        )
                        ax.set_ylabel("Volume")

                    elif indicator_name == "atr":
                        ax.plot(df.index, indicator_data, label="ATR", color="orange")
                        ax.set_ylabel("ATR")

                    ax.legend()
                    ax.grid(self.vis_settings.get("show_grid", True))
                    subplot_idx += 1

        # Plot equity curve
        if self.vis_settings.get("show_equity_curve", True) and not equity_curve.empty:
            ax_equity = axes[-1]
            ax_equity.plot(
                equity_curve.index,
                equity_curve.values,
                label="Equity Curve",
                color="green",
                linewidth=2,
            )
            ax_equity.set_ylabel("Equity")
            ax_equity.legend()
            ax_equity.grid(self.vis_settings.get("show_grid", True))

        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        # Save plot
        output_format = self.vis_settings.get("plot_format", "png")
        output_file = json_file.replace(".json", f".{output_format}")
        plt.savefig(
            output_file, dpi=self.vis_settings.get("plot_dpi", 300), bbox_inches="tight"
        )
        plt.close()

        _logger.info("Plot saved: %s", output_file)

    def _plot_trades(self, ax, trades: List[Dict]) -> None:
        """Plot trades on the price chart"""
        for trade in trades:
            if trade.get("entry_time") and trade.get("exit_time"):
                entry_time = pd.Timestamp(trade["entry_time"])
                exit_time = pd.Timestamp(trade["exit_time"])
                entry_price = trade["entry_price"]
                exit_price = trade["exit_price"]

                # Plot buy marker (green triangle)
                ax.scatter(
                    entry_time,
                    entry_price,
                    color="green",
                    marker="^",
                    s=100,
                    label="Buy" if trade == trades[0] else "",
                    zorder=5,
                )

                # Plot sell marker (red triangle)
                ax.scatter(
                    exit_time,
                    exit_price,
                    color="red",
                    marker="v",
                    s=100,
                    label="Sell" if trade == trades[0] else "",
                    zorder=5,
                )

    def process_json_file(self, json_file: str) -> None:
        """Process a single JSON file and create plot"""
        try:
            _logger.info("Processing %s", json_file)

            # Load result data
            result_data = self.load_result_data(json_file)
            if not result_data:
                return

            # Get data file name
            data_file = result_data.get("data_file", "")
            if not data_file:
                _logger.warning("No data_file found in %s", json_file)
                return

            # Load price data
            df = self.load_price_data(data_file)
            if df.empty:
                _logger.warning("Could not load price data for %s", data_file)
                return

            # Get strategy parameters
            strategy_params = result_data.get("best_params", {})

            # Get indicators to calculate
            indicators_to_calculate = self.get_indicators_for_strategy(result_data)

            # Calculate indicators
            indicators = self.calculate_indicators(
                df, indicators_to_calculate, strategy_params
            )

            # Get trades
            trades = result_data.get("trades", [])

            # Calculate equity curve
            initial_capital = self.config.get("optimizer_settings", {}).get(
                "initial_capital", 1000.0
            )
            equity_curve, _ = self.calculate_equity_curve(trades, initial_capital)

            # Create plot
            self.create_plot(
                df, indicators, trades, equity_curve, json_file, result_data
            )

        except Exception:
            _logger.exception("Error processing %s: %s")

    def run(self, results_dir: str = "results") -> None:
        """Main method to process all JSON files"""
        _logger.info("Starting plot generation...")

        # Get all JSON files
        json_files = self.get_json_files(results_dir)

        if not json_files:
            _logger.warning("No JSON files found in %s", results_dir)
            return

        # Process each file
        for json_file in json_files:
            self.process_json_file(json_file)

        _logger.info("Plot generation completed!")


def main():
    """Main function to run the plotter"""
    plotter = ResultPlotter()
    plotter.run()


if __name__ == "__main__":
    main()
