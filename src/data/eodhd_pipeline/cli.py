"""
Command line of the EODHD pipeline::

    python -m src.data.eodhd_pipeline.cli quota                 # plan and calls left today (free of charge)
    python -m src.data.eodhd_pipeline.cli universe              # 2 calls: live + delisted symbol lists
    python -m src.data.eodhd_pipeline.cli download              # 3 calls per stock, resumable
    python -m src.data.eodhd_pipeline.cli build
    python -m src.data.eodhd_pipeline.cli crosscheck
    python -m src.data.eodhd_pipeline.cli status

Every command takes ``--data-root`` (default ``DATA_CACHE_DIR/eodhd-data``). See docs/Runbook.md.

Exit codes: 0 done, 2 the key's plan cuts the history (free plan; the data is not usable), 3 stopped early (budget or
quota; run ``download`` again tomorrow).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eodhd_pipeline import config  # noqa: E402
from src.data.eodhd_pipeline.build import build_store  # noqa: E402
from src.data.eodhd_pipeline.client import EodhdClient  # noqa: E402
from src.data.eodhd_pipeline.crosscheck import run_crosscheck  # noqa: E402
from src.data.eodhd_pipeline.download import download_histories, load_progress  # noqa: E402
from src.data.eodhd_pipeline.store import EodhdStore  # noqa: E402
from src.data.eodhd_pipeline.universe import build_universe, load_universe  # noqa: E402
from src.notification.logger import setup_logger  # noqa: E402

_logger = setup_logger(__name__)

EXIT_PLAN_LIMITED = 2
EXIT_STOPPED_EARLY = 3


def _root(args: argparse.Namespace) -> Path:
    return Path(args.data_root) if args.data_root else config.default_data_root()


def _client() -> EodhdClient:
    key = config.api_key()
    if not key:
        raise SystemExit(f"{config.API_KEY_NAME} is not configured")
    return EodhdClient(key, calls_per_minute=config.CALLS_PER_MINUTE)


def _open_fmp(fmp_root: Optional[str]) -> Any:
    """Open the FMP vintage, or return None when there is none."""
    from src.data.fmp_pipeline.datamanager_adapter.fmp_store import FmpStore

    root = Path(fmp_root) if fmp_root else config.default_data_root().parent / "fmp-data"
    try:
        return FmpStore(root, None)
    except FileNotFoundError:
        _logger.warning("No FMP vintage under %s; continuing without it", root)
        return None


def _fmp_securities(fmp: Any) -> Any:
    """Read the columns of the FMP vintage securities table that the mapping needs."""
    import pyarrow.parquet as pq

    from src.data.fmp_pipeline.builders.validate import SECURITIES

    columns = ["security_id", "symbol", "name", "instrument_type", "first_price_date", "last_price_date"]
    return pq.read_table(fmp.dir / SECURITIES, columns=columns).to_pandas()


def _quota(args: argparse.Namespace) -> int:
    client = _client()
    try:
        record = client.user()
    finally:
        client.close()
    left = int(record.get("dailyRateLimit") or 0) + int(record.get("extraLimit") or 0) - int(record.get("apiRequests") or 0)
    _logger.info("EODHD plan %s (%s): %s of %s daily calls used on %s, %s extra; %d left", record.get("subscriptionType"),
                 record.get("subscriptionMode"), record.get("apiRequests"), record.get("dailyRateLimit"), record.get("apiRequestsDate"),
                 record.get("extraLimit"), left)
    return 0


def _universe(args: argparse.Namespace) -> int:
    client = _client()
    try:
        build_universe(client, _root(args), args.types, None if args.all_exchanges else config.MAIN_EXCHANGES)
    finally:
        client.close()
    return _status(args)


def _download(args: argparse.Namespace) -> int:
    root = _root(args)
    universe = load_universe(root)
    client = _client()
    try:
        summary = download_histories(client, root, universe, args.max_calls, args.workers, args.allow_limited, args.force,
                                     args.codes, args.limit)
    finally:
        client.close()
    for error in summary.errors[:20]:
        _logger.warning("  %s", error)
    if summary.plan_limited:
        _logger.error("The key's plan cuts the history or lacks splits/dividends: buy the EOD plan before the real download")
        return EXIT_PLAN_LIMITED
    if summary.stop_reason in ("budget", "quota"):
        _logger.warning("Stopped early (%s); run download again to continue", summary.stop_reason)
        return EXIT_STOPPED_EARLY
    return 0


def _build(args: argparse.Namespace) -> int:
    fmp = None if args.no_fmp else _open_fmp(args.fmp_root)
    info = build_store(_root(args), _fmp_securities(fmp) if fmp is not None else None)
    _logger.info("Build summary: %s", json.dumps(info, default=str))
    return 0


def _crosscheck(args: argparse.Namespace) -> int:
    summary = run_crosscheck(EodhdStore(_root(args)), None if args.no_fmp else _open_fmp(args.fmp_root), args.limit)
    _logger.info("Cross-check summary: %s", json.dumps(summary, indent=2, default=str))
    return 0


def _status(args: argparse.Namespace) -> int:
    root = _root(args)
    try:
        universe = load_universe(root)
    except FileNotFoundError:
        _logger.info("Data root %s: no universe yet (run universe)", root)
        return 0
    progress = load_progress(root)
    codes = set(universe["eodhd_code"])
    states = [record.get("status") for code, record in progress.items() if code in codes]
    ok, missing, errors = states.count("ok"), states.count("missing"), states.count("error")
    pending = len(universe) - ok - missing
    _logger.info("Data root %s: universe %d codes (%d delisted, %d live)", root, len(universe), int((~universe["listed"]).sum()),
                 int(universe["listed"].sum()))
    _logger.info("Downloaded %d, not in EODHD %d, failed last time %d, still to do %d (about %d calls)", ok, missing, errors, pending,
                 pending * config.CALLS_PER_STOCK)
    limited = sum(1 for record in progress.values() if record.get("plan_limited") or record.get("events_restricted"))
    if limited:
        _logger.warning("%d downloads came from a limited plan (dry run); download again with --force on the paid key", limited)
    info_path = root / config.STORE_DIR / config.BUILD_INFO
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        _logger.info("Store built %s: %s securities with bars, %s price rows, %s mapped to FMP", info.get("built_at"),
                     info.get("securities_with_bars"), info.get("price_rows"), info.get("mapped_to_fmp"))
        for row in info.get("coverage_by_last_trade_year", []):
            _logger.info("  delisted %-10s %6d with bars, %6d also in FMP, %6d new", row["bucket"], row["delisted_with_bars"],
                         row["also_in_fmp"], row["new_vs_fmp"])
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Run the command line; returns the process exit code."""
    parser = argparse.ArgumentParser(prog="eodhd", description="EODHD delisted-history pipeline")
    parser.add_argument("--data-root", help="EODHD data root (default DATA_CACHE_DIR/eodhd-data)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("quota", help="Plan and calls left today (costs no call)").set_defaults(func=_quota)

    universe = sub.add_parser("universe", help="Download the symbol lists and assign ids (2 calls)")
    universe.add_argument("--types", nargs="*", default=list(config.DEFAULT_TYPES), help="EODHD Type values to keep")
    universe.add_argument("--all-exchanges", action="store_true", help="Keep OTC venues too")
    universe.set_defaults(func=_universe)

    download = sub.add_parser("download", help="Download bars, splits and dividends (3 calls per stock, resumable)")
    download.add_argument("--max-calls", type=int, help="Spend at most this many calls in this run")
    download.add_argument("--workers", type=int, default=4, help="Parallel downloads")
    download.add_argument("--codes", nargs="*", help="Only these EODHD codes")
    download.add_argument("--limit", type=int, help="Only the first N planned codes")
    download.add_argument("--force", action="store_true", help="Download finished codes again")
    download.add_argument("--allow-limited", action="store_true", help="Accept a free plan's cut history (mechanics dry run only)")
    download.set_defaults(func=_download)

    build = sub.add_parser("build", help="Build the Parquet store from the raw files")
    build.add_argument("--fmp-root", help="FMP data root (default DATA_CACHE_DIR/fmp-data)")
    build.add_argument("--no-fmp", action="store_true", help="Do not map securities to the FMP vintage")
    build.set_defaults(func=_build)

    check = sub.add_parser("crosscheck", help="Check the store against EODHD's adjusted close and the FMP vintage")
    check.add_argument("--fmp-root")
    check.add_argument("--no-fmp", action="store_true")
    check.add_argument("--limit", type=int, help="Vendor check: only this many mapped securities")
    check.set_defaults(func=_crosscheck)

    sub.add_parser("status", help="Universe, download progress and store summary").set_defaults(func=_status)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
