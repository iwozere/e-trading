"""Pipeline checks (spec §11): V-COMPLETE, V-RAW-ARCHIVE, V-REPRO, V-ADJ-UNIT."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.fmp_pipeline.archive.archive import SUFFIX
from src.data.fmp_pipeline.archive.lists import latest_records
from src.data.fmp_pipeline.archive.reader import ArchiveCorrupt, iter_records
from src.data.fmp_pipeline.archive.symbol_path import encode_symbol, endpoint_dir
from src.data.fmp_pipeline.builders.events import DIVIDENDS, SPLITS
from src.data.fmp_pipeline.builders.identity import IdentityResult, SecurityRecord
from src.data.fmp_pipeline.builders.parquet_io import ParquetSettings, file_sha256
from src.data.fmp_pipeline.builders.prices import DAILY_ENDPOINT, PriceBuilder, PriceStats, bucket_path
from src.data.fmp_pipeline.planner.config import PROJECT_ROOT, BuildConfig, build_config_from_dict
from src.data.fmp_pipeline.validators.checksums import ARCHIVE_FILE, checksum_dir, read_summary, verify_checksums
from src.data.fmp_pipeline.validators.framework import FAIL, PASS, CheckResult, ValidationContext, check

OPEN_STATUSES = ("pending", "running", "failed")
REPRO_SECURITIES = 5
UNIT_TESTS = ("src/data/fmp_pipeline/tests/test_adjust.py", "src/data/fmp_pipeline/tests/test_yfinance_to_raw.py")


@check("V-COMPLETE", FAIL, "Manifest: no pending, running or failed tasks; restricted and empty listed")
def complete(ctx: ValidationContext) -> CheckResult:
    """Every task in a terminal state; ``skipped`` tasks are the plan's explicit skips."""
    result = CheckResult("V-COMPLETE", PASS)
    manifest = ctx.manifest()
    if manifest is None:
        result.status, result.message = FAIL, f"no manifest at {ctx.manifest_path}"
        return result
    with manifest:
        rows = manifest.task_rows()
    result.examined = len(rows)
    per_endpoint: dict[str, Counter[str]] = defaultdict(Counter)
    skipped: Counter[str] = Counter()
    for row in rows:
        per_endpoint[row["endpoint"]][row["status"]] += 1
        if row["status"] in OPEN_STATUSES:
            result.violations += 1
            result.add_example(f"{row['status']}: {row['endpoint']} {row['symbol'] or ''} {row['params_json']}")
        elif row["status"] == "skipped":
            skipped[row["error"] or "skipped"] += 1
    result.details = {
        "plan_restricted": {e: c["plan_restricted"] for e, c in sorted(per_endpoint.items()) if c["plan_restricted"]},
        "ok_empty": {e: c["ok_empty"] for e, c in sorted(per_endpoint.items()) if c["ok_empty"]},
        "skipped_by_plan": dict(skipped),
    }
    result.status = FAIL if result.violations else PASS
    return result


@check("V-RAW-ARCHIVE", FAIL, "Archive readable end to end, every response matches the manifest checksum")
def raw_archive(ctx: ValidationContext) -> CheckResult:
    """Decompress every member, match every finished task's sha256, and re-check ``ARCHIVE.sha256`` if present."""
    result = CheckResult("V-RAW-ARCHIVE", PASS)
    archived: set[tuple[str, str]] = set()
    for path in sorted(ctx.dump_dir.rglob("*" + SUFFIX)) if ctx.dump_dir.is_dir() else []:
        result.examined += 1
        try:
            for record in iter_records(path, strict=True):
                archived.add((record["task_id"], record["sha256"]))
        except (ArchiveCorrupt, ValueError) as exc:
            result.violations += 1
            result.add_example(f"unreadable: {exc}")
    manifest = ctx.manifest()
    if manifest is not None:
        with manifest:
            for row in manifest.task_rows():
                if row["sha256"] and (row["task_id"], row["sha256"]) not in archived:
                    result.violations += 1
                    result.add_example(f"no archived response for {row['endpoint']} {row['symbol'] or ''} "
                                       f"(sha256 {row['sha256'][:12]})")
    directory = checksum_dir(ctx.data_root, ctx.store.vintage)
    if (directory / ARCHIVE_FILE).exists():
        for error in verify_checksums(ctx.dump_dir, directory / ARCHIVE_FILE, read_summary(directory).get("archive")):
            result.violations += 1
            result.add_example(f"{ARCHIVE_FILE}: {error}")
    result.status = FAIL if result.violations else PASS
    result.message = f"{result.examined:,} archive files, {len(archived):,} responses"
    return result


def _rebuild_config(ctx: ValidationContext) -> tuple[BuildConfig, ParquetSettings]:
    info = ctx.store.build_info
    config = build_config_from_dict(info["build_config"]) if "build_config" in info else BuildConfig()
    parquet = info.get("parquet", {})
    return config, ParquetSettings(int(parquet.get("compression_level", 6)), int(parquet.get("row_group_rows", 100_000)))


@check("V-REPRO", FAIL, "Rebuilding a random subset of daily files from the archive reproduces their hashes")
def repro(ctx: ValidationContext) -> CheckResult:
    """Rebuild a few securities' daily files into a temp dir and compare sha256 with ``build_info.json``."""
    result = CheckResult("V-REPRO", PASS)
    outputs = ctx.store.build_info.get("outputs", {})
    securities = ctx.scan.securities
    candidates = [sid for sid in ctx.sample if bucket_path("1d_raw", sid) in outputs][:REPRO_SECURITIES]
    if not candidates:
        result.message = "no daily files to rebuild"
        return result
    config, settings = _rebuild_config(ctx)
    splits = pq.read_table(ctx.vintage_dir / SPLITS.path).to_pandas()
    dividends = pq.read_table(ctx.vintage_dir / DIVIDENDS.path).to_pandas()
    with tempfile.TemporaryDirectory(prefix="fmp-repro-") as tmp:
        for sid in candidates:
            symbol = ctx.store.security(sid).symbol
            rows = securities[securities["symbol"] == symbol]
            records = [SecurityRecord(key=str(r["security_id"]), symbol=symbol, security_id=int(r["security_id"]),
                                      valid_from=None if pd.isna(r["valid_from"]) else r["valid_from"],
                                      valid_to=None if pd.isna(r["valid_to"]) else r["valid_to"])
                       for _, r in rows.iterrows()]
            ids = {r.security_id for r in records}
            split_dates = _dates_by_security(splits[splits["ratio"].notna()], "date", ids)
            dividend_dates = _dates_by_security(dividends, "ex_date", ids)
            builder = PriceBuilder(ctx.dump_dir, Path(tmp), IdentityResult(records, by_symbol={symbol: records}),
                                   config, ctx.store.vintage, settings, split_dates, dividend_dates)
            archive = ctx.dump_dir / endpoint_dir(DAILY_ENDPOINT) / (encode_symbol(symbol) + SUFFIX)
            builder.build_daily_symbol(symbol, latest_records(archive), PriceStats())
            rel = bucket_path("1d_raw", sid)
            result.examined += 1
            rebuilt = Path(tmp) / rel
            if not rebuilt.exists() or file_sha256(rebuilt) != outputs[rel]["sha256"]:
                result.violations += 1
                result.add_example(f"{symbol}: rebuilt {rel} differs from build_info.json")
    result.status = FAIL if result.violations else PASS
    return result


def _dates_by_security(frame: pd.DataFrame, column: str, ids: set[int]) -> dict[int, np.ndarray]:
    out = {}
    for sid in ids:
        dates = frame.loc[(frame["security_id"] == sid) & frame[column].notna(), column]
        if len(dates):
            out[sid] = np.array(sorted(dates), dtype="datetime64[D]")
    return out


@check("V-ADJ-UNIT", FAIL, "Unit tests of adjust() and yfinance_to_raw() pass")
def adj_unit(ctx: ValidationContext) -> CheckResult:
    """Run the WP-7 test modules in a subprocess."""
    tests = [t for t in UNIT_TESTS if (PROJECT_ROOT / t).exists()]
    result = CheckResult("V-ADJ-UNIT", PASS, examined=len(tests))
    if not UNIT_TESTS or len(tests) != len(UNIT_TESTS):  # never call pytest without paths: it would collect all
        result.status, result.message = FAIL, "unit test modules missing"
        return result
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests], cwd=PROJECT_ROOT,
                         capture_output=True, text=True, timeout=600, check=False)
    tail = (run.stdout.strip().splitlines() or [""])[-1]
    result.message = tail
    if run.returncode != 0:
        result.status, result.violations = FAIL, 1
        result.add_example(tail)
    return result
