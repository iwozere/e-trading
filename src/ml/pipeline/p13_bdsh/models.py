from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class P13Config:
    entry_tiers: Dict[str, Dict[str, float]]
    exit_z_threshold: float = 0.0
    vix_lookback: int = 30
    initial_capital: float = 100000.0
    slippage_pct: float = 0.001
    stop_loss_pct: float = 0.10
    atr_period: int = 14
    atr_multiplier: float = 2.0
    vix_symbol: str = "^VIX"
    resolution: str = "1d"

    @classmethod
    def from_module(cls, config_module: Any):
        """Creates a P13Config instance from the config module's constants."""
        return cls(
            entry_tiers=config_module.ENTRY_TIERS,
            exit_z_threshold=config_module.EXIT_Z_THRESHOLD,
            vix_lookback=config_module.VIX_LOOKBACK,
            initial_capital=config_module.INITIAL_CAPITAL,
            slippage_pct=config_module.SLIPPAGE_PCT,
            stop_loss_pct=config_module.STOP_LOSS_PCT,
            atr_period=config_module.ATR_PERIOD,
            atr_multiplier=config_module.ATR_MULTIPLIER,
            vix_symbol=config_module.VIX_SYMBOL,
            resolution=config_module.RESOLUTION,
        )
