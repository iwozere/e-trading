"""
Command line of the Sharadar pipeline::

    python -m src.data.sharadar_pipeline.cli status
    python -m src.data.sharadar_pipeline.cli download --sample      # free key, a few large caps
    python -m src.data.sharadar_pipeline.cli bulk-status            # paid key: file sizes, no download
    python -m src.data.sharadar_pipeline.cli download               # paid key: all bulk files
    python -m src.data.sharadar_pipeline.cli build
    python -m src.data.sharadar_pipeline.cli crosscheck
    python -m src.data.sharadar_pipeline.cli purge --confirm
    python -m src.data.sharadar_pipeline.cli eodhd-check            # paid EODHD key: depth of its delisted history

Every command takes ``--data-root`` (default ``DATA_CACHE_DIR/sharadar-data``). See docs/Runbook.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sharadar_pipeline import config  # noqa: E402
from src.data.sharadar_pipeline.build import build_store, read_raw  # noqa: E402
from src.data.sharadar_pipeline.client import SharadarClient  # noqa: E402
from src.data.sharadar_pipeline.convert import securities_frame  # noqa: E402
from src.data.sharadar_pipeline.crosscheck import run_crosscheck  # noqa: E402
from src.data.sharadar_pipeline.download import bulk_status, download_bulk, download_sample  # noqa: E402
from src.data.sharadar_pipeline.eodhd_check import EodhdClient, run_check, summarize  # noqa: E402
from src.data.sharadar_pipeline.manifest import Manifest  # noqa: E402
from src.data.sharadar_pipeline.mapping import map_to_fmp  # noqa: E402
from src.data.sharadar_pipeline.purge import inventory, purge  # noqa: E402
from src.data.sharadar_pipeline.store import SharadarStore  # noqa: E402
from src.notification.logger import setup_logger  # noqa: E402

_logger = setup_logger(__name__)


def _root(args: argparse.Namespace) -> Path:
    return Path(args.data_root) if args.data_root else config.default_data_root()


def _paid_client() -> SharadarClient:
    key = config.api_key()
    if not key:
        raise SystemExit(f"{config.API_KEY_ENV} is not set; add it to .env (use --sample for the free key)")
    return SharadarClient(key)


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

    columns = ["security_id", "symbol", "cik", "instrument_type", "first_price_date", "last_price_date"]
    return pq.read_table(fmp.dir / SECURITIES, columns=columns).to_pandas()


def _fmp_close_reader(fmp: Any, tickers: Any) -> Any:
    """Return ``(sharadar_security_id, start, end) -> FMP raw closes`` for securities mapped by CIK."""
    securities = securities_frame(tickers)
    mapped = dict(zip(securities["security_id"], map_to_fmp(securities, _fmp_securities(fmp))))

    def closes(security_id: int, start: Any, end: Any) -> Any:
        fmp_id = mapped.get(security_id)
        if fmp_id is None or fmp_id is pd.NA:
            return None
        bars = fmp.bulk_daily([int(fmp_id)], start, end).get(int(fmp_id))
        return None if bars is None else bars["close"]

    return closes


def _eodhd_check(args: argparse.Namespace) -> int:
    from src.config.provider_config import get_api_key

    root = _root(args)
    key = get_api_key("EODHD_API_KEY")
    if not key:
        raise SystemExit("EODHD_API_KEY is not configured")
    tickers = read_raw(root, "tickers")  # from `download --sample`
    fmp = None if args.no_fmp else _open_fmp(args.fmp_root)
    client = EodhdClient(key)
    try:
        result = run_check(client, tickers, None if args.all else args.per_bucket, args.seed, args.max_calls,
                           _fmp_close_reader(fmp, tickers) if fmp is not None else None)
    finally:
        client.close()
    summary = summarize(result)
    reports = root / config.REPORTS_DIR
    reports.mkdir(parents=True, exist_ok=True)
    result.to_csv(reports / "eodhd_check.csv", index=False)  # holds Sharadar ticker rows: stays in the data root
    kept = root.parent / "eodhd-check"
    kept.mkdir(parents=True, exist_ok=True)
    summary.to_csv(kept / "summary.csv", index=False)
    _logger.info("EODHD check: %d stocks, %d API calls. Detail %s, summary %s", len(result), client.calls,
                 reports / "eodhd_check.csv", kept / "summary.csv")
    for line in summary.to_string(index=False).splitlines():
        _logger.info("%s", line)
    return 2 if result.attrs.get("plan_limited") else 0


def _status(args: argparse.Namespace) -> int:
    root = _root(args)
    manifest = Manifest.load(root)
    files = inventory(root)
    _logger.info("Data root %s: %d files, %.1f MB, mode %s", root, len(files), sum(f["size"] for f in files) / 1e6, manifest.mode)
    for table, entry in sorted(manifest.files.items()):
        _logger.info("  %-18s %12d bytes  rows %s  downloaded %s", table, entry.size, entry.rows, entry.downloaded_at)
    deadline = manifest.deletion_deadline()
    if deadline is not None:
        left = manifest.days_left()
        level = _logger.error if left is not None and left < 0 else _logger.warning if left is not None and left <= 7 else _logger.info
        level("Deletion deadline %s (%s days left). Subscription start %s, end %s", deadline, left,
              manifest.subscription_start, manifest.subscription_end or "not recorded (one month assumed)")
    return 0


def _download(args: argparse.Namespace) -> int:
    root = _root(args)
    client = SharadarClient(config.SAMPLE_KEY) if args.sample else _paid_client()
    try:
        if args.sample:
            download_sample(client, root)
        else:
            download_bulk(client, root, args.tables or config.BULK_TABLES, args.force, args.subscription_start)
    finally:
        client.close()
    return _status(args)


def _bulk_status(args: argparse.Namespace) -> int:
    client = _paid_client()
    try:
        for table, status in bulk_status(client, args.tables or config.BULK_TABLES).items():
            _logger.info("%-18s %s", table, json.dumps(status))
    finally:
        client.close()
    return 0


def _build(args: argparse.Namespace) -> int:
    fmp_securities = None
    fmp = None if args.no_fmp else _open_fmp(args.fmp_root)
    if fmp is not None:
        fmp_securities = _fmp_securities(fmp)
    info = build_store(_root(args), fmp_securities)
    _logger.info("Build summary: %s", json.dumps(info, default=str))
    return 0


def _crosscheck(args: argparse.Namespace) -> int:
    store = SharadarStore(_root(args))
    summary = run_crosscheck(store, None if args.no_fmp else _open_fmp(args.fmp_root), args.limit)
    _logger.info("Cross-check summary: %s", json.dumps(summary, indent=2, default=str))
    return 0


def _set_end(args: argparse.Namespace) -> int:
    root = _root(args)
    manifest = Manifest.load(root)
    manifest.subscription_end = args.date
    manifest.save(root)
    return _status(args)


def _purge(args: argparse.Namespace) -> int:
    root = _root(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    report_path = Path(args.report) if args.report else root.parent / f"sharadar-purge-report-{stamp}.json"
    report = purge(root, confirm=args.confirm, report_path=report_path)
    if report["deleted"]:
        _logger.info("Deleted %d files (%.1f MB). Report: %s", report["files"], report["bytes"] / 1e6, report_path)
        _logger.info("Still to do by hand: the Pi copy, backup disks, and %s in .env", config.API_KEY_ENV)
    else:
        _logger.info("Dry run: %d files, %.1f MB would be deleted from %s. Add --confirm to delete.", report["files"],
                     report["bytes"] / 1e6, root)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Run the command line; returns the process exit code."""
    parser = argparse.ArgumentParser(prog="sharadar", description="Sharadar research-window pipeline")
    parser.add_argument("--data-root", help="Sharadar data root (default DATA_CACHE_DIR/sharadar-data)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Files, manifest and deletion deadline").set_defaults(func=_status)

    download = sub.add_parser("download", help="Download bulk files (paid key) or the free sample")
    download.add_argument("--sample", action="store_true", help="Use the public sample key")
    download.add_argument("--tables", nargs="*", help="Only these tables")
    download.add_argument("--force", action="store_true", help="Download again even if present")
    download.add_argument("--subscription-start", help="ISO date the subscription began (default: today)")
    download.set_defaults(func=_download)

    status = sub.add_parser("bulk-status", help="Bulk file sizes without downloading (paid key)")
    status.add_argument("--tables", nargs="*")
    status.set_defaults(func=_bulk_status)

    build = sub.add_parser("build", help="Build the Parquet store from raw files")
    build.add_argument("--fmp-root", help="FMP data root (default DATA_CACHE_DIR/fmp-data)")
    build.add_argument("--no-fmp", action="store_true", help="Do not map securities to the FMP vintage")
    build.set_defaults(func=_build)

    check = sub.add_parser("crosscheck", help="Check the store against its own adjusted close and the FMP vintage")
    check.add_argument("--fmp-root")
    check.add_argument("--no-fmp", action="store_true")
    check.add_argument("--limit", type=int, help="Vendor check: only this many mapped securities")
    check.set_defaults(func=_crosscheck)

    end = sub.add_parser("set-end", help="Record the day the subscription ended (sets the deletion deadline)")
    end.add_argument("date", help="ISO date")
    end.set_defaults(func=_set_end)

    purge_cmd = sub.add_parser("purge", help="Delete the data root (dry run without --confirm)")
    purge_cmd.add_argument("--confirm", action="store_true")
    purge_cmd.add_argument("--report", help="Report path (default: next to the data root)")
    purge_cmd.set_defaults(func=_purge)

    eodhd = sub.add_parser("eodhd-check", help="Depth of EODHD's delisted history against Sharadar's ticker list (paid EODHD key)")
    eodhd.add_argument("--per-bucket", type=int, default=100, help="Stocks sampled per last-price-year bucket")
    eodhd.add_argument("--all", action="store_true", help="Check every delisted common stock (about 13,000 calls)")
    eodhd.add_argument("--seed", type=int, default=7)
    eodhd.add_argument("--max-calls", type=int, help="Stop after this many EODHD API calls")
    eodhd.add_argument("--fmp-root")
    eodhd.add_argument("--no-fmp", action="store_true", help="Skip the price-level comparison with the FMP vintage")
    eodhd.set_defaults(func=_eodhd_check)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
