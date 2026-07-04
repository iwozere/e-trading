import pandas as pd

from src.ml.pipeline.p12_order_flow.config import OrderFlowConfig
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


class MicrostructureAnalyzer:
    """
    Analyzes unified market data to generate microstructure features and signals.
    """

    def __init__(self, config: OrderFlowConfig):
        self.config = config

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers features and generates rule-based signals.
        """
        if df.empty:
            return df

        data = df.copy()

        # --- 1. Feature Engineering ---

        # OI Features
        data["oi_change"] = data["oi_base"].diff()
        data["oi_pct_change"] = data["oi_base"].pct_change()

        # Rolling Z-score for OI Change to detect anomalies
        window = 24  # 24h lookback
        data["oi_change_mean"] = data["oi_pct_change"].rolling(window=window).mean()
        data["oi_change_std"] = data["oi_pct_change"].rolling(window=window).std()
        data["oi_zscore"] = (data["oi_pct_change"] - data["oi_change_mean"]) / data["oi_change_std"]

        # Funding Features
        data["funding_ma"] = data["funding_rate"].rolling(window=3).mean()  # 3 * 8h = 24h

        # LS Ratio Features
        data["ls_ratio_ma"] = data["ls_ratio"].rolling(window=self.config.ls_ratio_ma_window).mean()
        data["ls_ratio_diff"] = data["ls_ratio"] - data["ls_ratio_ma"]

        # Price Features
        data["returns"] = data["close"].pct_change()

        # --- 2. Signal Generation (Rules) ---

        # Rule A: The Liquidation Flush (Long Flush)
        # Price drops significantly AND OI drops (forced liquidations)
        # Parameters from config
        data["long_flush"] = (
            (data["returns"] < self.config.liq_flush_price_drop)
            & (data["oi_pct_change"] < self.config.liq_flush_oi_drop)
        ).astype(int)

        # Rule B: Short Squeeze Divergence
        # Negative funding (shorts paying longs) AND Price rising AND OI dropping (short covering)
        # Or Price rising + LS ratio dropping (retail selling/shorts being squeezed)
        data["short_squeeze"] = (
            (data["funding_rate"] < -self.config.extreme_funding_threshold)
            & (data["returns"] > 0.01)
            & (data["oi_pct_change"] < 0)
        ).astype(int)

        # Rule C: Extreme Crowding (Topping/Bottoming Heuristic)
        # Extreme positive funding + High LS Ratio = Crowded Longs (Risk of Long Squeeze)
        data["crowded_longs"] = (
            (data["funding_rate"] > self.config.extreme_funding_threshold)
            & (data["ls_ratio"] > data["ls_ratio_ma"] * 1.2)
        ).astype(int)

        data["crowded_shorts"] = (
            (data["funding_rate"] < -self.config.extreme_funding_threshold)
            & (data["ls_ratio"] < data["ls_ratio_ma"] * 0.8)
        ).astype(int)

        # Cleanup intermediate columns
        to_drop = ["oi_change_mean", "oi_change_std"]
        data = data.drop(columns=[c for c in to_drop if c in data.columns])

        _logger.info("Microstructure analysis complete. Engineered %d features.", len(data.columns))
        return data
