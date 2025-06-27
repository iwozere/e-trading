from src.plotter.indicators.base_indicator_plotter import BaseIndicatorPlotter


class IchimokuPlotter(BaseIndicatorPlotter):
    def plot(self, ax):
        """Plot Ichimoku Cloud indicator"""
        try:
            dates = [self.data.datetime.datetime(i) for i in range(len(self.data))]

            # Plot the cloud components
            ax.plot(
                dates,
                self.indicators["tenkan"],
                label="Tenkan-sen",
                color="blue",
                alpha=0.7,
            )
            ax.plot(
                dates,
                self.indicators["kijun"],
                label="Kijun-sen",
                color="red",
                alpha=0.7,
            )

            # Plot the cloud
            ax.fill_between(
                dates,
                self.indicators["senkou_span_a"],
                self.indicators["senkou_span_b"],
                where=self.indicators["senkou_span_a"]
                >= self.indicators["senkou_span_b"],
                color="green",
                alpha=0.2,
                label="Bullish Cloud",
            )
            ax.fill_between(
                dates,
                self.indicators["senkou_span_a"],
                self.indicators["senkou_span_b"],
                where=self.indicators["senkou_span_a"]
                < self.indicators["senkou_span_b"],
                color="red",
                alpha=0.2,
                label="Bearish Cloud",
            )

            self._apply_style(ax)
        except Exception as e:
            self.logger.error(f"Error plotting Ichimoku Cloud: {str(e)}")

    @property
    def subplot_type(self):
        return "price"
