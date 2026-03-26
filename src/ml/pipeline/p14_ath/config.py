"""
ATH Pipeline Configuration

Configuration dataclasses for the Sequential ATH & Drawdown Analysis pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

@dataclass
class ATHPipelineConfig:
    """
    Complete ATH pipeline configuration.
    """
    
    # Analysis parameters
    lookback_years: int = 10
    tickers: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "SPY", "QQQ"])
    
    # Output settings
    results_dir: Path = field(default_factory=lambda: Path("results/p14_ath"))
    output_csv: str = "ath_drawdown_analysis.csv"
    
    # Visualization settings
    generate_plots: bool = True
    log_scale: bool = True
    plot_markers: bool = True
    
    # Data parameters
    interval: str = "1d"
    
    @classmethod
    def create_default(cls) -> "ATHPipelineConfig":
        """Create pipeline config with default settings."""
        return cls()
