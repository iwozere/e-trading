from src.plotter.indicators.base_indicator_plotter import BaseIndicatorPlotter


class BollingerBandsPlotter(BaseIndicatorPlotter):
    def plot(self, ax):
        """Plot Bollinger Bands on the price axis"""
        try:
            # Try to plot entry BB first
            if "bb" in self.indicators:
                bb = self.indicators["bb"]
                if not hasattr(bb, "lines"):
                    self.logger.warning(
                        "Bollinger Bands indicator has invalid data structure"
                    )
                    return

                dates = [self.data.datetime.datetime(i) for i in range(len(self.data))]

                # Plot the bands
                ax.plot(
                    dates,
                    bb.lines[0].array,
                    label="BB Upper (Entry)",
                    color="red",
                    alpha=0.5,
                )
                ax.plot(
                    dates,
                    bb.lines[1].array,
                    label="BB Middle (Entry)",
                    color="blue",
                    alpha=0.5,
                )
                ax.plot(
                    dates,
                    bb.lines[2].array,
                    label="BB Lower (Entry)",
                    color="green",
                    alpha=0.5,
                )

                # Fill between bands
                ax.fill_between(
                    dates,
                    bb.lines[0].array,
                    bb.lines[2].array,
                    color="gray",
                    alpha=0.1,
                    label="BB Range (Entry)",
                )

            # Then try to plot exit BB if it exists
            if "exit_bb" in self.indicators:
                bb = self.indicators["exit_bb"]
                if not hasattr(bb, "lines"):
                    self.logger.warning(
                        "Exit Bollinger Bands indicator has invalid data structure"
                    )
                    return

                dates = [self.data.datetime.datetime(i) for i in range(len(self.data))]

                # Plot the bands
                ax.plot(
                    dates,
                    bb.lines[0].array,
                    label="BB Upper (Exit)",
                    color="red",
                    alpha=0.5,
                    linestyle="--",
                )
                ax.plot(
                    dates,
                    bb.lines[1].array,
                    label="BB Middle (Exit)",
                    color="blue",
                    alpha=0.5,
                    linestyle="--",
                )
                ax.plot(
                    dates,
                    bb.lines[2].array,
                    label="BB Lower (Exit)",
                    color="green",
                    alpha=0.5,
                    linestyle="--",
                )

                # Fill between bands
                ax.fill_between(
                    dates,
                    bb.lines[0].array,
                    bb.lines[2].array,
                    color="gray",
                    alpha=0.05,
                    label="BB Range (Exit)",
                )

            self._apply_style(ax)
        except Exception as e:
            self.logger.error(f"Error plotting Bollinger Bands: {str(e)}")

    @property
    def subplot_type(self):
        return "price"
