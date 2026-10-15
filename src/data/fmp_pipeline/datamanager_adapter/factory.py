"""
Build the ``DataManager``'s FMP adapter from ``config/fmp/fmp.yaml`` (``fmp.adapter``), or ``None``.

``None`` means "behave exactly as before": the adapter is disabled (the default), or it is enabled but cannot be
opened (no active vintage, unreadable config). The second case is logged as an error, never raised, so a broken
FMP install can not take the ``DataManager`` (and its ~150 callers) down with it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from src.data.fmp_pipeline.datamanager_adapter.adapter import FmpOhlcvAdapter
from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore
from src.data.fmp_pipeline.increments.store import IncrementStore
from src.data.fmp_pipeline.planner.config import load_config
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)


def default_data_root() -> Path:
    """``DATA_CACHE_DIR/fmp-data`` (owner decision Q2)."""
    try:
        from config.donotshare.donotshare import DATA_CACHE_DIR
    except ImportError:
        DATA_CACHE_DIR = os.getenv("DATA_CACHE_DIR", "c:/data-cache")
    return Path(DATA_CACHE_DIR) / "fmp-data"


def load_adapter(config_path: Optional[Path] = None) -> Optional[FmpOhlcvAdapter]:
    """
    Return the configured adapter, or ``None`` when it is disabled or cannot be opened.

    Args:
        config_path: Pipeline config; default ``config/fmp/fmp.yaml``.
    """
    try:
        settings = load_config(config_path).adapter
    except (OSError, ValueError) as exc:
        _logger.error("FMP adapter config unreadable, adapter off: %s", exc)
        return None
    if not settings.enabled:
        return None
    data_root = Path(settings.data_root) if settings.data_root else default_data_root()
    try:
        store = FmpStore(data_root)
    except (OSError, ValueError) as exc:
        _logger.error("FMP adapter enabled but no usable vintage under %s, adapter off: %s", data_root, exc)
        return None
    # Always attached: an empty or not-yet-created increment layer simply returns no bars.
    increments = IncrementStore(data_root)
    _logger.info("FMP adapter on: vintage %s, timeframes %s, increments %s", store.vintage,
                 ", ".join(settings.timeframes), "present" if increments.root.is_dir() else "none yet")
    return FmpOhlcvAdapter(store, increments, timeframes=settings.timeframes, cache_entries=settings.cache_entries)
