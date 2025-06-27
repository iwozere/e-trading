from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from src.notification.logger import setup_logger


class BaseIndicatorPlotter(ABC):
    def __init__(self, data, indicators, vis_settings):
        self.data = data
        self.indicators = indicators
        self.vis_settings = vis_settings
        self.logger = setup_logger()

    @abstractmethod
    def plot(self, ax):
        """Plot the indicator on the given axis"""
        pass

    @property
    @abstractmethod
    def subplot_type(self):
        """Return the type of subplot needed for this indicator:
        - 'price': plot on the same subplot as price
        - 'separate': plot on a separate subplot
        """
        pass

    def _apply_style(self, ax):
        """Apply common styling to the axis"""
        try:
            if self.vis_settings.get("show_grid", True):
                ax.grid(True, alpha=0.3)
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=self.vis_settings.get("font_size", 10),
            )
            ax.legend(loc=self.vis_settings.get("legend_loc", "upper left"))
        except Exception as e:
            self.logger.error(f"Error applying style to axis: {str(e)}")
