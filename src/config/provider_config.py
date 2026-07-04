"""
Provider Configuration Service
================================

Central, module-level store for all external data-provider API keys.

Keys are loaded **once at import time** using a two-step priority:
  1. Environment variable (upper-cased key name)
  2. ``config.donotshare.donotshare`` module attribute

**Import-time contract:** ``_load()`` executes as soon as this module is first
imported.  Any env var set *after* import (e.g. by a late ``dotenv`` load or a
test fixture) will not be picked up for the pre-seeded keys.  Call
``reload()`` explicitly after changing env vars in tests or at runtime.

Unknown keys (not in ``_KNOWN_KEYS``) are resolved live on first access and
then cached.  A ``None`` result is also cached — call ``reload()`` if the key
is added later.

All downloader constructors should call ``get_api_key(key_name)`` through
``BaseDataDownloader._get_config_value``, which delegates here.

Usage::

    from src.config.provider_config import get_api_key, reload
    key = get_api_key("FMP_API_KEY")
"""

from __future__ import annotations

import logging
import os
from typing import Dict

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known provider config keys – add new ones here, documented.
# ---------------------------------------------------------------------------
_KNOWN_KEYS = [
    "FINNHUB_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "POLYGON_API_KEY",
    "TWELVE_DATA_API_KEY",
    "FMP_API_KEY",
    "TIINGO_API_KEY",
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_BASE_URL",
    "BINANCE_API_KEY",
    "BINANCE_SECRET",
    "EODHD_API_KEY",
    "TRADIER_API",
    "SANTIMENT_API_KEY",
    "NEWSAPI_API_KEY",
    "FINRA_API_CLIENT",
    "FINRA_API_SECRET",
    "FRED_API_KEY",
    "ANTHROPIC_API_KEY",
]

# ---------------------------------------------------------------------------
# Internal cache filled at import time.
# ---------------------------------------------------------------------------
_cache: Dict[str, str | None] = {}


def _load() -> None:
    """Populate the cache from env vars and then the donotshare module."""
    # Try to import the donotshare module once
    donotshare = None
    try:
        import importlib

        donotshare = importlib.import_module("config.donotshare.donotshare")
    except ImportError:
        _logger.debug("config.donotshare.donotshare not available; relying on env vars only")

    for key in _KNOWN_KEYS:
        # 1. Environment variable takes precedence
        value = os.getenv(key)
        if not value and donotshare:
            # 2. Fall back to the module attribute
            value = getattr(donotshare, key, None) or None
        _cache[key] = value or None


_load()


def reload() -> None:
    """
    Re-read all known keys from env and donotshare and reset the live-lookup
    cache.  Call this in tests after patching env vars, or after a runtime
    key rotation when env is updated externally.
    """
    _cache.clear()
    _load()


def get_api_key(key_name: str) -> str | None:
    """
    Return the API key/token for the given config key name.

    Checks the cache (pre-loaded at import time from env + donotshare).
    If the key was not in ``_KNOWN_KEYS`` it falls back to a live lookup
    so that callers with custom keys still work.

    Args:
        key_name: Config key (e.g. ``"FMP_API_KEY"``).

    Returns:
        The key value as a string, or ``None`` if not configured.
    """
    if key_name in _cache:
        return _cache[key_name]

    # Live lookup for keys not in the pre-seeded list
    value = os.getenv(key_name)
    if not value:
        try:
            import importlib

            mod = importlib.import_module("config.donotshare.donotshare")
            value = getattr(mod, key_name, None) or None
        except ImportError:
            pass

    # Cache for next time
    _cache[key_name] = value
    return value
