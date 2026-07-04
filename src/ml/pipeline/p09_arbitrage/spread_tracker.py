
import pandas as pd

from src.ml.pipeline.p09_arbitrage.config import ArbitrageConfig


class SpreadTracker:
    """
    Calculates and tracks the rolling spread and its statistical properties.
    """

    def __init__(self, config: ArbitrageConfig):
        self.config = config

    def calculate_rolling_stats(self, series_a: pd.Series, series_b: pd.Series, beta: float) -> pd.DataFrame:
        """
        Computes rolling spread, mean, standard deviation, and Z-score.
        Spread = SeriesA - (Beta * SeriesB)
        """
        df = pd.DataFrame({"price_a": series_a, "price_b": series_b}).dropna()

        # Calculate raw spread
        df["spread"] = df["price_a"] - (beta * df["price_b"])

        # Rolling stats
        window = self.config.lookback_window
        df["rolling_mean"] = df["spread"].rolling(window=window).mean()
        df["rolling_std"] = df["spread"].rolling(window=window).std()

        # Z-score
        df["zscore"] = (df["spread"] - df["rolling_mean"]) / df["rolling_std"]

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates entry/exit signals based on Config thresholds.
        1: Long Spread (A bought, B sold)
        -1: Short Spread (A sold, B bought)
        0: Neutral / Exit
        """
        entry = self.config.zscore_entry_threshold
        exit = self.config.zscore_exit_threshold

        df["signal"] = 0

        # Simple logical states (can be improved with state machine logic if needed)
        # For now, rule-based approach:

        # Short spread if zscore > entry (expecting mean reversion down)
        df.loc[df["zscore"] > entry, "signal"] = -1

        # Long spread if zscore < -entry (expecting mean reversion up)
        df.loc[df["zscore"] < -entry, "signal"] = 1

        # Exit logic is tougher with simple loc.
        # Typically we use a loop or a custom shift logic to hold signal until exit threshold is hit.

        # State-aware signal generation
        current_signal = 0
        signals = []

        for z in df["zscore"]:
            if current_signal == 0:
                if z > entry:
                    current_signal = -1
                elif z < -entry:
                    current_signal = 1
            elif current_signal == 1:
                if z >= -exit:
                    current_signal = 0
            elif current_signal == -1:
                if z <= exit:
                    current_signal = 0

            signals.append(current_signal)

        df["signal"] = signals
        return df

    def calculate_signals(
        self, df_a: pd.DataFrame, df_b: pd.DataFrame, beta: float, config: ArbitrageConfig | None = None
    ) -> pd.DataFrame:
        """
        Convenience method to calculate stats and signals in one go.
        """
        # Ensure we use the right config if passed overrides, else use self.config
        # Note: self.config is already set in __init__
        stats_df = self.calculate_rolling_stats(df_a["close"], df_b["close"], beta)
        return self.generate_signals(stats_df)
