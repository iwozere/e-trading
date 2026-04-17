#!/usr/bin/env python3
"""
Probe fundamentals for one symbol across registered downloaders.

Use this to trace why ``market_cap`` / ``avg_volume`` are missing: compare
raw ``get_fundamentals`` payloads per provider vs ``DataManager.get_fundamentals``.

Example:
    python tests/tmp_probe_fundamentals_symbol.py AAPL
    python tests/tmp_probe_fundamentals_symbol.py AACI --providers yahoo,finnhub
    python tests/tmp_probe_fundamentals_symbol.py BND --force-refresh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_manager import DataManager  # noqa: E402
from src.data.downloader.data_downloader_factory import DataDownloaderFactory  # noqa: E402


def _summarize_fundamentals(obj: Any) -> Dict[str, Any]:
    """Return a small dict of liquidity fields whether input is dict or dataclass-like."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        d = obj
    elif hasattr(obj, "__dict__"):
        d = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    elif hasattr(obj, "_asdict"):
        d = obj._asdict()
    else:
        return {"repr": repr(obj)[:200]}

    profile = d.get("profile") if isinstance(d.get("profile"), dict) else {}
    return {
        "market_cap": d.get("market_cap"),
        "avg_volume": d.get("avg_volume"),
        "average_volume": d.get("average_volume"),
        "current_price": d.get("current_price"),
        "profile_marketCap": profile.get("marketCap"),
        "profile_averageVolume": profile.get("averageVolume"),
        "profile_volume": profile.get("volume"),
        "data_source": d.get("data_source"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe fundamentals per provider + DataManager.")
    parser.add_argument("symbol", help="Ticker symbol, e.g. AAPL")
    parser.add_argument(
        "--providers",
        type=str,
        default="yahoo,finnhub,fmp,alpha_vantage,twelvedata",
        help="Comma-separated provider codes (must support get_fundamentals)",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Bypass cache in DataManager")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional DataManager cache dir")
    args = parser.parse_args()

    symbol = args.symbol.strip().upper()
    providers: List[str] = [p.strip().lower() for p in args.providers.split(",") if p.strip()]

    print("=== Per-provider get_fundamentals (raw) ===")
    for name in providers:
        canonical = DataDownloaderFactory.get_provider_by_code(name)
        if not canonical:
            print("%s: unknown provider code" % name)
            continue
        downloader = DataDownloaderFactory.create_downloader(canonical)
        if downloader is None or not hasattr(downloader, "get_fundamentals"):
            print("%s (%s): skip (no get_fundamentals)" % (name, canonical))
            continue
        try:
            raw = downloader.get_fundamentals(symbol)
            summary = _summarize_fundamentals(raw)
            print("%s (%s): %s" % (name, canonical, json.dumps(summary, default=str)))
        except Exception as exc:
            print("%s (%s): ERROR %s" % (name, canonical, exc))

    print("\n=== DataManager.get_fundamentals (combined, data_type=general) ===")
    dm = DataManager(args.cache_dir) if args.cache_dir else DataManager()
    combined = dm.get_fundamentals(
        symbol,
        data_type="general",
        force_refresh=args.force_refresh,
        max_age_days=14,
    )
    print(json.dumps(_summarize_fundamentals(combined), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
