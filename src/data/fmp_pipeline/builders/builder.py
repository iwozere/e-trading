"""
``fmp build`` (spec §7.4, plan WP-6): turn one dump's raw archive into a Parquet vintage.

Steps, all reading **only** the archive (never the network):

1. Universe from the archived bootstrap lists and profiles (same rules as ``fmp plan``), then identities and
   ``security_id`` allocation (``identity.py``).
2. Events (``events.py``), then raw prices one security at a time with flags (``prices.py``), then securities and
   symbol history, then every other endpoint as a generic record table (``records.py``, ``tables.py``).
3. Structural validation (``validate.py``). Only if it passes: the id registry is saved, the temp directory
   replaces ``vintages/<dump_id>/``, and ``current.txt`` (a text file, never a symlink) is switched.

Everything is built in ``vintages/.<dump_id>.building/``; a failed build leaves the previous vintage and
``current.txt`` untouched. ``_meta/build_info.json`` records code version, inputs (a digest of every response
used), settings, and per output file its row count and sha256 (V-REPRO). The same archive and code produce the
same files.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa

from src.data.fmp_pipeline.archive.archive import SUFFIX
from src.data.fmp_pipeline.archive.lists import symbol_files
from src.data.fmp_pipeline.builders.events import build_events
from src.data.fmp_pipeline.builders.identity import IdentityBuilder, IdentityResult, IdRegistry
from src.data.fmp_pipeline.builders.parquet_io import ParquetSettings, file_sha256, write_table
from src.data.fmp_pipeline.builders.prices import DAILY_ENDPOINT, PriceBuilder, PriceStats
from src.data.fmp_pipeline.builders.records import build_tables
from src.data.fmp_pipeline.builders.tables import INTRADAY_PREFIX, TABLES, covered_endpoints
from src.data.fmp_pipeline.builders.validate import SECURITIES, validate_structure
from src.data.fmp_pipeline.fsutil import atomic_write_bytes, replace_with_retry
from src.data.fmp_pipeline.planner.config import PROJECT_ROOT, PlanConfig
from src.data.fmp_pipeline.planner.planner import read_profiles, read_sources
from src.data.fmp_pipeline.planner.universe import assemble_universe
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

BUILDER_VERSION = 1
CURRENT_FILE = "current.txt"
REGISTRY_FILE = "security_ids.json"
HISTORY = "reference/symbol_history.parquet"
SECURITIES_SCHEMA = pa.schema([
    pa.field("security_id", pa.int64()), pa.field("symbol", pa.string()), pa.field("yf_symbol", pa.string()),
    pa.field("exchange", pa.string()), pa.field("name", pa.string()), pa.field("instrument_type", pa.string()),
    pa.field("cik", pa.string()), pa.field("isin", pa.string()), pa.field("cusip", pa.string()),
    pa.field("ipo_date", pa.date32()), pa.field("delisted_date", pa.date32()),
    pa.field("is_actively_trading", pa.bool_()), pa.field("first_price_date", pa.date32()),
    pa.field("last_price_date", pa.date32()), pa.field("source_flags", pa.string()), pa.field("tier", pa.string()),
    pa.field("valid_from", pa.date32()), pa.field("valid_to", pa.date32()),
    pa.field("identity_uncertain", pa.bool_()), pa.field("uncertain_reason", pa.string()),
])
HISTORY_SCHEMA = pa.schema([pa.field("security_id", pa.int64()), pa.field("symbol", pa.string()),
                            pa.field("valid_from", pa.date32()), pa.field("valid_to", pa.date32()),
                            pa.field("reason", pa.string())])


class BuildFailed(Exception):
    """Structural validation failed; the vintage was not switched."""

    def __init__(self, errors: list[str], kept: Path) -> None:
        """
        Args:
            errors: Validation messages.
            kept: Where the rejected build was left for inspection.
        """
        super().__init__(f"{len(errors)} structural errors; rejected build kept at {kept}: " + "; ".join(errors[:5]))
        self.errors = errors
        self.kept = kept


@dataclass
class BuildResult:
    """What ``VintageBuilder.build`` produced."""

    vintage_dir: Path
    switched: bool
    securities: int
    outputs: dict[str, int]
    daily: PriceStats
    intraday: dict[str, PriceStats] = field(default_factory=dict)
    unbuilt: list[str] = field(default_factory=list)
    uncertain: int = 0
    info: dict[str, Any] = field(default_factory=dict)


def code_version() -> str:
    """Return the git commit of the code (``unknown`` outside a checkout)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                             timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def archived_endpoints(dump_dir: Path) -> list[str]:
    """Return every endpoint with at least one archive file, as ``a/b`` paths, sorted."""
    found = {path.parent.relative_to(dump_dir).as_posix() for path in dump_dir.rglob("*" + SUFFIX)}
    return sorted(found)


def read_current(data_root: Path) -> Optional[str]:
    """Return the dump id in ``current.txt``, or ``None`` if there is none."""
    path = data_root / CURRENT_FILE
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or None


class VintageBuilder:
    """Builds the vintage of one dump."""

    def __init__(self, data_root: Path, config: PlanConfig, as_of: date) -> None:
        """
        Args:
            data_root: Pipeline data root (``raw/``, ``vintages/``, ``reports/``, ``current.txt``).
            config: Pipeline configuration; ``config.dump_id`` selects the dump.
            as_of: The dump's date (from its manifest): decides which delistings have happened.
        """
        self.data_root = data_root
        self.config = config
        self.as_of = as_of
        self.dump_id = config.dump_id
        self.dump_dir = data_root / "raw" / config.dump_id
        self.vintages = data_root / "vintages"
        self.report_dir = data_root / "reports" / config.dump_id
        build = config.build
        self.settings = ParquetSettings(build.zstd_level, build.row_group_rows)

    def build(self, switch: bool = True) -> BuildResult:
        """
        Build, validate and (if ``switch``) activate the vintage.

        Args:
            switch: Point ``current.txt`` at the new vintage after it passes validation.

        Returns:
            Counts, statistics and paths.

        Raises:
            FileNotFoundError: The dump has no archive.
            BuildFailed: Structural validation failed (nothing was switched).
        """
        if not self.dump_dir.is_dir():
            raise FileNotFoundError(f"no raw archive for dump {self.dump_id} at {self.dump_dir}")
        tmp = self.vintages / f".{self.dump_id}.building"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)

        registry = IdRegistry(self.data_root / REGISTRY_FILE)
        identity = self._identity(registry)
        _logger.info("Build %s: %d securities (%d new ids)", self.dump_id, len(identity.securities),
                     len(registry.added))
        events = build_events(self.dump_dir, tmp, identity.resolve, self.settings)
        prices = PriceBuilder(self.dump_dir, tmp, identity, self.config.build, self.dump_id, self.settings,
                              events.split_dates, events.dividend_dates)
        daily = prices.build_daily()
        _logger.info("Daily raw: %d securities, %d bars, %d flagged", daily.securities, daily.bars, daily.flagged_bars)
        intraday: dict[str, PriceStats] = {}
        for interval in self._intervals():
            stats = prices.build_intraday(interval)
            if stats is not None:
                intraday[interval] = stats
        outputs: dict[str, int] = {f.path: f.rows for f in prices.files}
        outputs.update({path: s.rows for path, s in events.stats.items()})
        outputs[SECURITIES] = write_table(tmp / SECURITIES, self._securities_table(identity), self.settings)
        outputs[HISTORY] = write_table(tmp / HISTORY, self._history_table(identity), self.settings)
        tables = build_tables(TABLES, self.dump_dir, tmp, identity.resolve, self.settings,
                              self.config.build.batch_rows, self.config.build.schema_sample_files)
        outputs.update({path: s.rows for path, s in tables.items()})

        errors = validate_structure(tmp, outputs)
        if errors:
            rejected = self.vintages / f".{self.dump_id}.rejected"
            if rejected.exists():
                shutil.rmtree(rejected)
            replace_with_retry(tmp, rejected)
            raise BuildFailed(errors, rejected)

        unbuilt = self._unbuilt()
        info = self._build_info(tmp, outputs, identity, daily, intraday, events.invalid_ratios,
                                {**events.stats, **tables}, unbuilt)
        atomic_write_bytes(tmp / "_meta" / "build_info.json", json.dumps(info, indent=1, sort_keys=True).encode())
        registry.save()
        final = self._activate(tmp)
        self._reports(prices.missing_splits_csv(), identity)
        if switch:
            atomic_write_bytes(self.data_root / CURRENT_FILE, f"{self.dump_id}\n".encode("utf-8"))
            _logger.info("current.txt now points to %s", self.dump_id)
        uncertain = sum(1 for s in identity.securities if s.identity_uncertain)
        return BuildResult(final, switch, len(identity.securities), outputs, daily, intraday, unbuilt, uncertain, info)

    # --- steps ---------------------------------------------------------------------------------------------

    def _identity(self, registry: IdRegistry) -> IdentityResult:
        sources, _ = read_sources(self.dump_dir)
        profiles = read_profiles(self.dump_dir)
        universe = assemble_universe(sources, profiles, self.config.universe, self.as_of)
        identity_builder = IdentityBuilder(registry, self.config.build.segment_reused_tickers)
        identity = identity_builder.build(universe, sources, profiles)
        with_daily = {f.symbol for f in symbol_files(self.dump_dir, DAILY_ENDPOINT)}
        # A symbol known only as the old side of a rename, with no price file of its own, is history, not a security.
        orphans = {s.security_id for s in identity.securities if s.rename_only and s.symbol not in with_daily}
        if orphans:
            identity.drop(orphans)
        return identity

    def _intervals(self) -> list[str]:
        folder = self.dump_dir / "historical-chart"
        archived = sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []
        return sorted(set(archived) | {self.config.intraday.interval})

    def _unbuilt(self) -> list[str]:
        covered = covered_endpoints()
        return [e for e in archived_endpoints(self.dump_dir) if e not in covered and not e.startswith(INTRADAY_PREFIX)]

    @staticmethod
    def _securities_table(identity: IdentityResult) -> pa.Table:
        rows = sorted(identity.securities, key=lambda s: s.security_id)
        columns = {name: [getattr(s, name) for s in rows] for name in SECURITIES_SCHEMA.names}
        return pa.table(columns, schema=SECURITIES_SCHEMA)

    @staticmethod
    def _history_table(identity: IdentityResult) -> pa.Table:
        rows = identity.history
        return pa.table({name: [getattr(h, name) for h in rows] for name in HISTORY_SCHEMA.names},
                        schema=HISTORY_SCHEMA)

    def _build_info(self, tmp: Path, outputs: dict[str, int], identity: IdentityResult, daily: PriceStats,
                    intraday: dict[str, PriceStats], invalid_ratios: int, tables: dict[str, Any],
                    unbuilt: list[str]) -> dict[str, Any]:
        def price_stats(stats: PriceStats) -> dict[str, Any]:
            data = asdict(stats)
            data["flags"] = dict(sorted(stats.flags.items()))
            return data

        return {
            "dump_id": self.dump_id,
            "as_of": self.as_of.isoformat(),
            "builder_version": BUILDER_VERSION,
            "code_version": code_version(),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "parquet": self.settings.as_dict(),
            "build_config": asdict(self.config.build),
            "securities": len(identity.securities),
            "identity_uncertain": sum(1 for s in identity.securities if s.identity_uncertain),
            "segmented_symbols": sorted({s.symbol for s in identity.securities if s.valid_to or s.valid_from}),
            "prices": {"1d_raw": price_stats(daily), **{k: price_stats(v) for k, v in intraday.items()}},
            "invalid_split_ratios": invalid_ratios,
            "inputs": {path: {"responses": s.responses, "digest": s.input_digest, "out_of_window": s.out_of_window,
                              "duplicates": s.duplicates, "unresolved_rows": s.unresolved}
                       for path, s in sorted(tables.items())},
            "unbuilt_endpoints": unbuilt,
            "outputs": {rel: {"rows": rows, "sha256": file_sha256(tmp / rel)} for rel, rows in sorted(outputs.items())},
        }

    def _activate(self, tmp: Path) -> Path:
        """Replace ``vintages/<dump_id>/`` with the validated build."""
        final = self.vintages / self.dump_id
        old = self.vintages / f".{self.dump_id}.old"
        if old.exists():
            shutil.rmtree(old)
        if final.exists():
            replace_with_retry(final, old)
        replace_with_retry(tmp, final)
        if old.exists():
            shutil.rmtree(old)
        return final

    def _reports(self, missing_splits: bytes, identity: IdentityResult) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(self.report_dir / "missing_splits_candidates.csv", missing_splits)
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["symbol", "security_id", "reason"])
        writer.writerows([s.symbol, s.security_id, s.uncertain_reason or ""]
                         for s in sorted(identity.securities, key=lambda s: (s.symbol, s.security_id))
                         if s.identity_uncertain)
        atomic_write_bytes(self.report_dir / "identity_uncertain.csv", buffer.getvalue().encode("utf-8"))
