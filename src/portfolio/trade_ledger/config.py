"""
Trade ledger configuration.

Dataclass and YAML loader for the monthly trade-ledger report. Secrets (Flex
token, query id) are read from `config.donotshare.donotshare` at run time and
are not part of this config.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

DEFAULT_CONFIG_PATH = "src/portfolio/trade_ledger/config/trade_ledger.yaml"
SUPPORTED_CHANNELS = frozenset({"email"})


@dataclass
class TradeLedgerConfig:
    """
    Runtime configuration of the trade-ledger report.

    Attributes:
        recipient_id: User ID whose email address receives the report.
        channels: Delivery channels (only "email").
        cron: Cron expression (UTC) for the scheduled monthly run.
        download: Download the Flex statement before building the report.
        raw_dir: Raw Flex XML archive root (monthly downloads in `<raw_dir>/monthly`).
        reports_dir: Output folder for the XLSX and HTML copies.
    """

    recipient_id: Optional[int] = None
    channels: List[str] = field(default_factory=lambda: ["email"])
    cron: str = "0 6 2 * *"
    download: bool = True
    raw_dir: str = "data/portfolio/trade_ledger/raw"
    reports_dir: str = "data/portfolio/trade_ledger/reports"

    @property
    def monthly_dir(self) -> Path:
        """Folder holding the archived monthly downloads."""
        return Path(self.raw_dir) / "monthly"

    def validate(self) -> None:
        """
        Validate field values.

        Raises:
            ValueError: If a field has an invalid value.
        """
        if not self.channels:
            raise ValueError("At least one channel must be configured")
        unknown = set(self.channels) - SUPPORTED_CHANNELS
        if unknown:
            raise ValueError(f"Unsupported channels: {sorted(unknown)}. Supported: {sorted(SUPPORTED_CHANNELS)}")
        if len(self.cron.strip().split()) != 5:
            raise ValueError(f"cron must have exactly 5 fields: {self.cron!r}")
        if not self.raw_dir or not self.reports_dir:
            raise ValueError("raw_dir and reports_dir must be set")


def config_from_dict(raw: Dict[str, Any]) -> TradeLedgerConfig:
    """
    Build and validate a config from a parsed YAML mapping.

    Args:
        raw: Mapping as loaded from YAML; missing keys take the defaults.

    Returns:
        Validated `TradeLedgerConfig`.

    Raises:
        ValueError: If a value is invalid.
    """
    defaults = TradeLedgerConfig()
    recipient = raw.get("recipient_id")
    config = TradeLedgerConfig(
        recipient_id=int(recipient) if recipient is not None else None,
        channels=[str(c) for c in raw.get("channels", defaults.channels)],
        cron=str(raw.get("cron", defaults.cron)),
        download=bool(raw.get("download", defaults.download)),
        raw_dir=str(raw.get("raw_dir", defaults.raw_dir)),
        reports_dir=str(raw.get("reports_dir", defaults.reports_dir)),
    )
    config.validate()
    return config


def load_config(path: Optional[str] = None) -> TradeLedgerConfig:
    """
    Load the trade-ledger configuration from YAML.

    Args:
        path: YAML path (relative to the working directory or absolute).
            Defaults to `DEFAULT_CONFIG_PATH`.

    Returns:
        Validated `TradeLedgerConfig`.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML is not a mapping or fails validation.
    """
    cfg_path = Path(path) if path else Path(DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Trade ledger config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Trade ledger config must be a mapping, got {type(raw).__name__}")
    config = config_from_dict(raw)
    _logger.info("Loaded trade ledger config from %s (recipient_id=%s, download=%s)", cfg_path, config.recipient_id, config.download)
    return config
