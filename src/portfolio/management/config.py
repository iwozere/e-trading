"""
Portfolio Management (earnings-triggered stop-loss reminder) configuration.

Dataclasses and YAML loader for the pipeline described in
``docs/brainstorm.md`` and ``docs/Design.md``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


DEFAULT_CONFIG_PATH = "src/portfolio/management/config_data/management.yaml"

# IBKR working-order types treated as "protective" coverage for a long
# position (a resting SELL order of one of these types).
PROTECTIVE_ORDER_TYPES = frozenset({"STP", "STP LMT", "TRAIL", "TRAIL LIMIT"})

# Live Gateway port (vs. 4002 for paper — see src/trading/broker/ibkr_utils.py
# trading_mode port mapping). This module only ever connects to live: real
# stop orders only exist on the live account.
DEFAULT_LIVE_PORT = 4001

# Next free clientId after P19's 19 (intraday loop) / 20 (gapper scanner).
DEFAULT_LIVE_CLIENT_ID = 21


@dataclass
class ManagementConfig:
    """
    Runtime configuration for the earnings-triggered stop-loss reminder.

    Attributes:
        earnings_window_days: How many days ahead to look for earnings dates
            when resolving events for held tickers.
        trigger_window_minutes: Half-width of the match window (minutes)
            around each T-1day/T-1hour trigger moment. Must be >= half the
            polling cadence (`cron`) so no trigger moment falls between two
            consecutive runs.
        channels: Notification channels (subset of "telegram", "email").
        cron: Cron expression (UTC) for the polling schedule. Default covers
            ~12:00-22:00 UTC, which brackets both T-1day and T-1hour of
            08:30/09:30/15:00/16:00 ET across DST.
        ibkr_xml_path: Path (or glob pattern) to an IBKR Flex Query Open
            Positions XML export — reuses `pnl_alert`'s holdings pipeline.
        ibkr_stk_only: Restrict holdings to STK sec-types.
        ibkr_live_host: Live Gateway host (defaults to `IBKR_HOST`).
        ibkr_live_port: Live Gateway port (defaults to `IBKR_PORT` env var,
            else `DEFAULT_LIVE_PORT`).
        ibkr_live_client_id: Live Gateway clientId (defaults to
            `IBKR_LIVE_STOP_GUARD_CLIENT_ID` env var, else
            `DEFAULT_LIVE_CLIENT_ID`). Deliberately distinct from
            `IBKR_CLIENT_ID` to avoid colliding with a live trading bot
            session using that id.
        recipient_id: User ID whose email and Telegram are used for delivery.
    """

    earnings_window_days: int = 21
    trigger_window_minutes: int = 15
    channels: List[str] = field(default_factory=lambda: ["telegram", "email"])
    cron: str = "*/15 12-22 * * 1-5"
    ibkr_xml_path: str = "data/portfolio/pnl_alert/Open_Positions.xml"
    ibkr_stk_only: bool = True
    ibkr_live_host: str = ""
    ibkr_live_port: int = DEFAULT_LIVE_PORT
    ibkr_live_client_id: int = DEFAULT_LIVE_CLIENT_ID
    recipient_id: int | None = None

    def validate(self) -> None:
        """
        Validate field values.

        Raises:
            ValueError: If a field has an invalid value.
        """
        if self.earnings_window_days <= 0:
            raise ValueError("earnings_window_days must be > 0")
        if self.trigger_window_minutes <= 0:
            raise ValueError("trigger_window_minutes must be > 0")

        if not self.channels:
            raise ValueError("At least one channel must be configured")
        allowed_channels = {"telegram", "email"}
        unknown = set(self.channels) - allowed_channels
        if unknown:
            raise ValueError(f"Unsupported channels: {sorted(unknown)}. Supported: {sorted(allowed_channels)}")

        cron_parts = self.cron.strip().split()
        if len(cron_parts) != 5:
            raise ValueError(f"cron must have exactly 5 fields, got {len(cron_parts)}: {self.cron!r}")

        if self.ibkr_live_port <= 0:
            raise ValueError("ibkr_live_port must be > 0")
        if self.ibkr_live_client_id < 0:
            raise ValueError("ibkr_live_client_id must be >= 0")


def load_config(path: str | None = None) -> ManagementConfig:
    """
    Load pipeline configuration from a YAML file.

    `ibkr_live_host` / `ibkr_live_port` / `ibkr_live_client_id` fall back to
    `IBKR_HOST` / `IBKR_PORT` / `IBKR_LIVE_STOP_GUARD_CLIENT_ID` (see
    `config.donotshare.donotshare`) when not set in the YAML, so credentials
    stay out of version control the same way `pnl_alert` handles them.

    Args:
        path: Path to the YAML file, relative to the current working directory
            or absolute. Defaults to `DEFAULT_CONFIG_PATH`.

    Returns:
        Validated `ManagementConfig` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML is invalid or fails validation.
    """
    cfg_path = Path(path) if path else Path(DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Portfolio management config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Portfolio management config must be a mapping, got {type(raw).__name__}")

    try:
        from config.donotshare.donotshare import IBKR_HOST, IBKR_LIVE_STOP_GUARD_CLIENT_ID, IBKR_PORT
    except ImportError:
        IBKR_HOST = IBKR_PORT = IBKR_LIVE_STOP_GUARD_CLIENT_ID = None

    raw_recipient = raw.get("recipient_id")
    config = ManagementConfig(
        earnings_window_days=int(raw.get("earnings_window_days", 21)),
        trigger_window_minutes=int(raw.get("trigger_window_minutes", 15)),
        channels=list(raw.get("channels", ["telegram", "email"])),
        cron=str(raw.get("cron", "*/15 12-22 * * 1-5")),
        ibkr_xml_path=str(raw.get("ibkr_xml_path", "data/portfolio/pnl_alert/Open_Positions.xml")),
        ibkr_stk_only=bool(raw.get("ibkr_stk_only", True)),
        ibkr_live_host=str(raw.get("ibkr_live_host") or IBKR_HOST or ""),
        ibkr_live_port=int(raw.get("ibkr_live_port") or IBKR_PORT or DEFAULT_LIVE_PORT),
        ibkr_live_client_id=int(raw.get("ibkr_live_client_id") or IBKR_LIVE_STOP_GUARD_CLIENT_ID or DEFAULT_LIVE_CLIENT_ID),
        recipient_id=int(raw_recipient) if raw_recipient is not None else None,
    )
    config.validate()

    _logger.info(
        "Loaded portfolio management config from %s (window=%dd, trigger_window=%dmin, channels=%s)",
        cfg_path,
        config.earnings_window_days,
        config.trigger_window_minutes,
        config.channels,
    )
    return config
