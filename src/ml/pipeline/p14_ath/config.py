"""
ATH Pipeline Configuration

Configuration dataclasses for the Sequential ATH & Drawdown Analysis pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ATHPipelineConfig:
    """
    Complete ATH pipeline configuration.
    """

    # Analysis parameters
    lookback_years: int = 15
    tickers: List[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "SPY", "QQQ"]
    )

    # Output settings
    results_dir: Path = field(default_factory=lambda: Path("results/p14_ath"))
    output_csv: str = "ath_drawdown_analysis.csv"

    # Visualization settings
    generate_plots: bool = True
    log_scale: bool = True
    plot_markers: bool = True
    initial_equity_usd: float = 1000.0
    # Log y on equity panel helps long horizons (linear scale squashes early years).
    equity_log_scale: bool = True

    # Data parameters
    interval: str = "1d"

    @classmethod
    def create_default(cls) -> "ATHPipelineConfig":
        """Create pipeline config with default settings."""
        return cls()
