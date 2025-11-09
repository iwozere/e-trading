from src.backtester.plotter.indicators.base_indicator_plotter import BaseIndicatorPlotter

from src.notification.logger import setup_logger
_logger = setup_logger(__name__)

class RSIPlotter(BaseIndicatorPlotter):
    def plot(self, ax):
        """Plot RSI indicator"""
        try:
            # Try to plot entry RSI first
            if "rsi" in self.indicators:
                rsi = self.indicators["rsi"]
                if hasattr(rsi, "array"):
                    rsi_data = rsi.array
                elif hasattr(rsi, "lines"):
                    rsi_data = rsi.lines[0].array
                else:
                    _logger.warning("RSI indicator has invalid data structure")
                    return

                # Get dates from data feed, skipping buffer period
                dates = []
                for i in range(len(self.data)):
                    if i < self.data.buflen():
                        continue
                    dates.append(self.data.datetime.datetime(i))

                # Ensure data lengths match
                min_len = min(len(dates), len(rsi_data))
                dates = dates[:min_len]
                rsi_data = rsi_data[:min_len]

                # Plot RSI
                ax.plot(dates, rsi_data, label="RSI (Entry)", color="blue", alpha=0.7)
                _logger.debug("Plotted entry RSI with %d points", len(rsi_data))

            # Then try to plot exit RSI if it exists
            if "exit_rsi" in self.indicators:
                rsi = self.indicators["exit_rsi"]
                if hasattr(rsi, "array"):
                    rsi_data = rsi.array
                elif hasattr(rsi, "lines"):
                    rsi_data = rsi.lines[0].array
                else:
                    _logger.warning("Exit RSI indicator has invalid data structure")
                    return

                # Get dates from data feed, skipping buffer period
                dates = []
                for i in range(len(self.data)):
                    if i < self.data.buflen():
                        continue
                    dates.append(self.data.datetime.datetime(i))

                # Ensure data lengths match
                min_len = min(len(dates), len(rsi_data))
                dates = dates[:min_len]
                rsi_data = rsi_data[:min_len]

                # Plot RSI
                ax.plot(dates, rsi_data, label="RSI (Exit)", color="red", alpha=0.7)
                _logger.debug("Plotted exit RSI with %d points", len(rsi_data))

            # Add overbought/oversold lines
            ax.axhline(y=70, color="r", linestyle="--", alpha=0.3)
            ax.axhline(y=30, color="g", linestyle="--", alpha=0.3)

            ax.set_ylabel("RSI")
            self._apply_style(ax)
        except Exception:
            _logger.exception("Error plotting RSI: ")
            _logger.error("Indicator data: %s", self.indicators.get('rsi', 'Not found'))
            _logger.error("Data feed length: %d", len(self.data))

    @property
    def subplot_type(self):
        return "separate"
