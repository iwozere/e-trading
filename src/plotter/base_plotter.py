"""
Base Plotter Module

This module provides the base plotting functionality for strategy backtest results.
It handles:
1. Creating and managing plots
2. Plotting price data
3. Plotting indicators
4. Plotting trades
5. Plotting equity curve
"""

from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from src.notification.logger import setup_logger
from src.plotter.indicators.bollinger_bands_plotter import \
    BollingerBandsPlotter
from src.plotter.indicators.ichimoku_plotter import IchimokuPlotter
from src.plotter.indicators.rsi_plotter import RSIPlotter
from src.plotter.indicators.supertrend_plotter import SuperTrendPlotter
from src.plotter.indicators.volume_plotter import VolumePlotter

_logger = setup_logger(__name__)


class BasePlotter:
    def __init__(self, data, trades, strategy, vis_settings):
        """
        Initialize the plotter

        Args:
            data: Backtrader data feed
            trades: List of trade dictionaries
            strategy: Strategy instance containing indicators
            vis_settings: Visualization settings from optimizer config
        """
        self.data = data
        self.trades = trades
        self.strategy = strategy
        self.vis_settings = vis_settings
        self.fig = None
        self.axes = None

        # Initialize indicator plotters
        self.indicator_plotters = self._create_indicator_plotters()

    def _create_indicator_plotters(self):
        """Create appropriate indicator plotters based on strategy"""
        plotters = []
        entry_mixin = self.strategy.entry_mixin

        # Add plotters based on available indicators
        if hasattr(entry_mixin, "indicators"):
            indicators = entry_mixin.indicators
            _logger.info(f"Found indicators: {list(indicators.keys())}")

            # RSI
            if "rsi" in indicators:
                _logger.debug("Creating RSI plotter")
                try:
                    rsi_data = indicators["rsi"]
                    if not hasattr(rsi_data, "array") and not hasattr(
                        rsi_data, "lines"
                    ):
                        _logger.warning("RSI indicator has invalid data structure")
                    else:
                        plotters.append(
                            RSIPlotter(self.data, indicators, self.vis_settings)
                        )
                        _logger.debug("RSI plotter created successfully")
                except Exception as e:
                    _logger.error(f"Error creating RSI plotter: {str(e)}")

            # Ichimoku
            if all(
                k in indicators
                for k in ["tenkan", "kijun", "senkou_span_a", "senkou_span_b"]
            ):
                _logger.debug("Creating Ichimoku plotter")
                try:
                    # Validate Ichimoku data structure
                    valid = all(
                        hasattr(indicators[k], "array")
                        or hasattr(indicators[k], "lines")
                        for k in ["tenkan", "kijun", "senkou_span_a", "senkou_span_b"]
                    )
                    if not valid:
                        _logger.warning(
                            "Ichimoku indicators have invalid data structure"
                        )
                    else:
                        plotters.append(
                            IchimokuPlotter(self.data, indicators, self.vis_settings)
                        )
                        _logger.debug("Ichimoku plotter created successfully")
                except Exception as e:
                    _logger.error(f"Error creating Ichimoku plotter: {str(e)}")

            # Bollinger Bands
            if "bb" in indicators:
                _logger.debug("Creating Bollinger Bands plotter")
                try:
                    bb_data = indicators["bb"]
                    if not hasattr(bb_data, "lines") or len(bb_data.lines) < 3:
                        _logger.warning(
                            "Bollinger Bands indicator has invalid data structure"
                        )
                    else:
                        plotters.append(
                            BollingerBandsPlotter(
                                self.data, indicators, self.vis_settings
                            )
                        )
                        _logger.debug("Bollinger Bands plotter created successfully")
                except Exception as e:
                    _logger.error(f"Error creating Bollinger Bands plotter: {str(e)}")

            # Volume
            if "volume" in indicators:
                _logger.debug("Creating Volume plotter")
                try:
                    volume_data = indicators["volume"]
                    if not hasattr(volume_data, "array") and not hasattr(
                        volume_data, "lines"
                    ):
                        _logger.warning("Volume indicator has invalid data structure")
                    else:
                        plotters.append(
                            VolumePlotter(self.data, indicators, self.vis_settings)
                        )
                        _logger.debug("Volume plotter created successfully")
                except Exception as e:
                    _logger.error(f"Error creating Volume plotter: {str(e)}")

            # SuperTrend
            if "supertrend" in indicators:
                _logger.debug("Creating SuperTrend plotter")
                try:
                    supertrend_data = indicators["supertrend"]
                    if not hasattr(supertrend_data, "array") and not hasattr(
                        supertrend_data, "lines"
                    ):
                        _logger.warning(
                            "SuperTrend indicator has invalid data structure"
                        )
                    else:
                        plotters.append(
                            SuperTrendPlotter(self.data, indicators, self.vis_settings)
                        )
                        _logger.debug("SuperTrend plotter created successfully")
                except Exception as e:
                    _logger.error(f"Error creating SuperTrend plotter: {str(e)}")

            _logger.info(f"Created {len(plotters)} indicator plotters")
        else:
            _logger.warning("No indicators found in entry mixin")

        return plotters

    def plot(self, output_path):
        """Create and save the plot"""
        self._create_figure()
        self._plot_price()
        self._plot_indicators()
        self._plot_trades()
        if self.vis_settings.get("show_equity_curve", True):
            self._plot_equity()
        self._save_plot(output_path)

    def _create_figure(self):
        """Create figure with subplots"""
        # Set plot style
        plt.style.use(self.vis_settings.get("plot_style", "default"))

        # Calculate number of subplots needed
        n_subplots = 1  # Price subplot
        for plotter in self.indicator_plotters:
            if plotter.subplot_type == "separate":
                n_subplots += 1
        if self.vis_settings.get("show_equity_curve", True):
            n_subplots += 1

        # Create figure with appropriate size
        plot_size = self.vis_settings.get("plot_size", [15, 10])
        self.fig = plt.figure(figsize=plot_size)

        # Create subplots with appropriate ratios
        ratios = [3] + [1] * (n_subplots - 1)  # Price plot is larger
        self.axes = self.fig.subplots(n_subplots, 1, height_ratios=ratios)
        if n_subplots == 1:
            self.axes = [self.axes]

        # Set title with timestamp
        self.fig.suptitle(
            f'Strategy Backtest Results - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            fontsize=self.vis_settings.get("font_size", 10),
        )

    def _plot_price(self):
        """Plot price data"""
        ax = self.axes[0]

        # Get datetime and price data from Backtrader data feed
        dates = []
        prices = []

        # Ensure we have data to plot
        try:
            data_len = len(self.data)
            if data_len == 0:
                _logger.warning("No data available for plotting")
                return

            # Get data within valid range
            for i in range(data_len):
                try:
                    # Check if we can access the data at this index
                    if i >= len(self.data.datetime) or i >= len(self.data.close):
                        break

                    dates.append(self.data.datetime.datetime(i))
                    prices.append(self.data.close[i])
                except (IndexError, AttributeError) as e:
                    _logger.warning(
                        f"Error accessing data at index {i}: {str(e)}", exc_info=False
                    )
                    break  # Stop if we hit an error
        except Exception as e:
            _logger.error(f"Error accessing data: {str(e)}", exc_info=False)
            return

        # Ensure we have data to plot
        if not dates or not prices:
            _logger.warning("No valid data points collected for plotting")
            return

        # Convert to pandas datetime
        dates = pd.to_datetime(dates)

        # Plot price data
        ax.plot(dates, prices, label="Price", color="black", alpha=0.7)
        ax.set_ylabel("Price")

        # Configure grid
        if self.vis_settings.get("show_grid", True):
            ax.grid(True, alpha=0.3)

        # Configure legend
        ax.legend(loc=self.vis_settings.get("legend_loc", "upper left"))

        # Set font size
        ax.tick_params(
            axis="both", which="major", labelsize=self.vis_settings.get("font_size", 10)
        )

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        # Log data points for debugging
        _logger.debug(f"Plotted {len(dates)} data points")
        _logger.debug(f"Date range: {dates[0]} to {dates[-1]}")
        _logger.debug(f"Price range: {min(prices)} to {max(prices)}")

    def _plot_indicators(self):
        """Plot all indicators using their respective plotters"""
        current_ax = 0
        for plotter in self.indicator_plotters:
            try:
                if plotter.subplot_type == "price":
                    _logger.debug(
                        f"Plotting {plotter.__class__.__name__} on price axis"
                    )
                    plotter.plot(self.axes[0])
                else:
                    current_ax += 1
                    _logger.debug(
                        f"Plotting {plotter.__class__.__name__} on separate axis {current_ax}"
                    )
                    plotter.plot(self.axes[current_ax])
            except Exception as e:
                _logger.error(f"Error plotting {plotter.__class__.__name__}: {str(e)}")
                continue

    def _plot_trades(self):
        """Plot trade markers"""
        ax = self.axes[0]
        for trade in self.trades:
            try:
                # Plot entry
                if "entry_time" in trade:
                    # Convert Backtrader datetime to pandas datetime
                    entry_date = pd.to_datetime(trade["entry_time"])
                    if entry_date.year < 2000:  # Skip invalid dates
                        _logger.warning(f"Invalid entry date: {entry_date}")
                        continue
                    ax.scatter(
                        entry_date,
                        trade["entry_price"],
                        marker="^",
                        color="green",
                        s=100,
                        label="Buy" if trade == self.trades[0] else "",
                    )

                # Plot exit
                if "exit_time" in trade:
                    # Convert Backtrader datetime to pandas datetime
                    exit_date = pd.to_datetime(trade["exit_time"])
                    if exit_date.year < 2000:  # Skip invalid dates
                        _logger.warning(f"Invalid exit date: {exit_date}")
                        continue
                    ax.scatter(
                        exit_date,
                        trade["exit_price"],
                        marker="v",
                        color="red",
                        s=100,
                        label="Sell" if trade == self.trades[0] else "",
                    )
            except Exception as e:
                _logger.error(f"Error plotting trade: {str(e)}")
                continue

    def _plot_equity(self):
        """Plot equity curve"""
        if len(self.axes) > 1:  # Only plot if we have a second subplot
            ax = self.axes[-1]  # Always last subplot

            # Get equity data from strategy
            dates = pd.to_datetime(self.strategy.equity_dates)
            equity = self.strategy.equity_curve

            # Ensure data is properly aligned
            if len(dates) != len(equity):
                min_len = min(len(dates), len(equity))
                dates = dates[:min_len]
                equity = equity[:min_len]

            ax.plot(dates, equity, label="Equity", color="blue")
            ax.set_ylabel("Equity")

            # Configure grid
            if self.vis_settings.get("show_grid", True):
                ax.grid(True, alpha=0.3)

            # Configure legend
            ax.legend(loc=self.vis_settings.get("legend_loc", "upper left"))

            # Set font size
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=self.vis_settings.get("font_size", 10),
            )

    def _save_plot(self, output_path):
        """Save the plot to file"""
        plt.tight_layout()

        # Save plot with specified DPI
        plt.savefig(
            output_path,
            dpi=self.vis_settings.get("plot_dpi", 300),
            format=self.vis_settings.get("plot_format", "png"),
        )

        # Show plot if configured
        if self.vis_settings.get("show_plot", False):
            plt.show()

        plt.close()
