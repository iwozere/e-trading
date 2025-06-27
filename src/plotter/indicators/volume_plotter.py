import numpy as np
from src.plotter.indicators.base_indicator_plotter import BaseIndicatorPlotter


class VolumePlotter(BaseIndicatorPlotter):
    def plot(self, ax):
        """Plot Volume indicator"""
        try:
            volume = self.indicators["volume"]
            dates = [self.data.datetime.datetime(i) for i in range(len(self.data))]

            # Plot volume bars
            ax.bar(dates, volume, label="Volume", color="gray", alpha=0.5)
            ax.set_ylabel("Volume")

            # Plot volume MA if available
            if "vol_ma" in self.indicators:
                vol_ma = self.indicators["vol_ma"]
                ax.plot(dates, vol_ma, label="Volume MA", color="blue", alpha=0.7)

            self._apply_style(ax)
        except Exception as e:
            self.logger.error(f"Error plotting Volume: {str(e)}")

    @property
    def subplot_type(self):
        return "separate"
