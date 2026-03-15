from dataclasses import dataclass, field
from typing import List, Dict, Any
from datetime import datetime, timedelta

@dataclass
class OrderFlowConfig:
    """
    Configuration for the P12 Order Flow Pipeline.
    Initializes with default static parameters as per the specification.
    """
    # Asset parameters
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    timeframe: str = "1h"
    lookback_days: int = 30
    
    # Derivative parameters
    funding_interval: str = "8h"
    oi_interval: str = "1h"
    ls_ratio_interval: str = "1h"
    
    # Microstructure Thresholds (Static Parameters)
    # 1. Extreme Funding Threshold (e.g., 0.01% is standard, above that is extreme)
    extreme_funding_threshold: float = 0.01 
    
    # 2. Open Interest Growth (Z-Score threshold for anomalies)
    oi_zscore_threshold: float = 2.0
    
    # 3. Long/Short Ratio Smoothing
    ls_ratio_ma_window: int = 24  # 24 hours if timeframe is 1h
    
    # 4. Liquidation Flush Thresholds
    # Price drop % accompanied by massive OI drop %
    liq_flush_price_drop: float = -0.03 # -3%
    liq_flush_oi_drop: float = -0.05    # -5%
    
    # Output parameters
    result_root: str = "results/p12_order_flow"
    
    def get_start_date(self) -> datetime:
        """Calculate start date based on lookback."""
        return datetime.now() - timedelta(days=self.lookback_days)

    def get_end_date(self) -> datetime:
        """Get current time as end date."""
        return datetime.now()
