"""
Portfolio PnL Alert configuration.

Dataclasses and YAML loader for the daily PnL alert pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


DEFAULT_CONFIG_PATH = "src/portfolio/pnl_alert/config/pnl_alert.yaml"


@dataclass
class PnLAlertConfig:
    """
    Runtime configuration for the portfolio PnL alert pipeline.

    Attributes:
        threshold_pct: Minimum PnL fraction to include a holding in the alert.
            0.10 means +10%.
        channels: Notification channels (subset of "telegram", "email").
        cron: Cron expression (UTC) for the daily scheduled run.
        watchlist_path: Path, relative to project root, of the user watchlist.
        include_ibkr: Whether to pull live IBKR positions.
        ibkr_stk_only: Restrict IBKR positions to STK sec-types.
        recipient_id: User ID whose email and Telegram are used for delivery.
    """

    threshold_pct: float = 0.10
    channels: List[str] = field(default_factory=lambda: ["telegram", "email"])
    cron: str = "30 21 * * 1-5"
    watchlist_path: str = "src/portfolio/pnl_alert/config/watchlist.yaml"
    include_ibkr: bool = True
    ibkr_stk_only: bool = True
    recipient_id: Optional[int] = None

    def validate(self) -> None:
        """
        Validate field values.

        Raises:
            ValueError: If a field has an invalid value.
        """
        if not isinstance(self.threshold_pct, (int, float)):
            raise ValueError("threshold_pct must be a number")
        if self.threshold_pct <= 0:
            raise ValueError("threshold_pct must be > 0")

        if not self.channels:
            raise ValueError("At least one channel must be configured")
        allowed_channels = {"telegram", "email"}
        unknown = set(self.channels) - allowed_channels
        if unknown:
            raise ValueError(
                f"Unsupported channels: {sorted(unknown)}. "
                f"Supported: {sorted(allowed_channels)}"
            )

        cron_parts = self.cron.strip().split()
        if len(cron_parts) != 5:
            raise ValueError(
                f"cron must have exactly 5 fields, got {len(cron_parts)}: {self.cron!r}"
            )


def load_config(path: Optional[str] = None) -> PnLAlertConfig:
    """
    Load PnL alert configuration from a YAML file.

    Args:
        path: Path to the YAML file, relative to the current working directory
            or absolute. Defaults to `DEFAULT_CONFIG_PATH`.

    Returns:
        Validated `PnLAlertConfig` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML is invalid or fails validation.
    """
    cfg_path = Path(path) if path else Path(DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        raise FileNotFoundError(f"PnL alert config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"PnL alert config must be a mapping, got {type(raw).__name__}")

    raw_recipient = raw.get("recipient_id")
    config = PnLAlertConfig(
        threshold_pct=float(raw.get("threshold_pct", 0.10)),
        channels=list(raw.get("channels", ["telegram", "email"])),
        cron=str(raw.get("cron", "30 21 * * 1-5")),
        watchlist_path=str(raw.get("watchlist_path", "src/portfolio/pnl_alert/config/watchlist.yaml")),
        include_ibkr=bool(raw.get("include_ibkr", True)),
        ibkr_stk_only=bool(raw.get("ibkr_stk_only", True)),
        recipient_id=int(raw_recipient) if raw_recipient is not None else None,
    )
    config.validate()

    _logger.info(
        "Loaded PnL alert config from %s (threshold=%.2f%%, channels=%s, include_ibkr=%s)",
        cfg_path,
        config.threshold_pct * 100,
        config.channels,
        config.include_ibkr,
    )
    return config
