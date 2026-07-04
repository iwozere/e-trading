from dataclasses import dataclass


@dataclass
class ArbitrageConfig:
    """
    Configuration for the P09 Arbitrage Pipeline.
    """

    # Universe selection
    min_cointegration_p_value: float = 0.05

    # Signal thresholds
    zscore_entry_threshold: float = 2.5
    zscore_exit_threshold: float = 0.5

    # Indicators
    lookback_window: int = 100  # Window for rolling spread mean/std

    # Strategy parameters
    hedge_ratio_recalculation_freq: int = 7  # Days
    max_holding_period: int = 14  # Days

    # Output path
    result_root: str = "results/p09_arbitrage"

    # Asset constraints (optional)
    data_dir: str = "data"
    default_timeframe: str = "1h"
