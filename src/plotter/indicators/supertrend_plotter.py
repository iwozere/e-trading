from src.plotter.indicators.base_indicator_plotter import BaseIndicatorPlotter


class SuperTrendPlotter(BaseIndicatorPlotter):
    def plot(self, ax):
        """Plot SuperTrend indicator"""
        try:
            dates = [self.data.datetime.datetime(i) for i in range(len(self.data))]

            # Plot SuperTrend line
            ax.plot(
                dates,
                self.indicators["supertrend"],
                label="SuperTrend",
                color="purple",
                alpha=0.7,
            )

            # Plot direction changes
            direction = self.indicators["direction"]
            for i in range(1, len(direction)):
                if direction[i] != direction[i - 1]:
                    color = "green" if direction[i] > 0 else "red"
                    ax.scatter(
                        dates[i],
                        self.indicators["supertrend"][i],
                        color=color,
                        marker="o",
                        s=50,
                    )

            self._apply_style(ax)
        except Exception as e:
            self.logger.error(f"Error plotting SuperTrend: {str(e)}")

    @property
    def subplot_type(self):
        return "price"

    # Plot ATR if available
    def plot_atr(self, ax):
        if "atr" in self.indicators:
            ax.plot(
                self.data.datetime.array,
                self.indicators["atr"].array,
                label="ATR",
                color="orange",
                alpha=0.5,
                linestyle="--",
            )

    # Fill between price and SuperTrend to show trend direction
    def fill_between(self, ax):
        ax.fill_between(
            self.data.datetime.array,
            self.data.close.array,
            self.indicators["supertrend"].array,
            where=self.data.close.array > self.indicators["supertrend"].array,
            color="green",
            alpha=0.1,
            label="Bullish",
        )
        ax.fill_between(
            self.data.datetime.array,
            self.data.close.array,
            self.indicators["supertrend"].array,
            where=self.data.close.array <= self.indicators["supertrend"].array,
            color="red",
            alpha=0.1,
            label="Bearish",
        )
