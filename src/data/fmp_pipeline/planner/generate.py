"""
Task generation in the strict order of spec §6.3.5, with a size estimate per task.

``generate`` yields every task the plan implies, each with estimated rows and bytes. Tasks that cannot be fixed yet
are yielded with ``deferred`` set and a reason; they count in the projection but are not written to the manifest
(D36):

- per-symbol tasks (other than ``profile``) for candidates whose exchange is still unknown;
- intraday windows of a symbol whose daily raw history is not finished (the windows depend on its first/last bar,
  spec §6.5.5);
- all intraday windows while the row cap is unknown (``intraday.max_rows`` is set from the paid probe).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional

from src.data.fmp_pipeline.manifest.manifest import TaskSpec
from src.data.fmp_pipeline.planner.catalog import EndpointSpec, priority_of, selected
from src.data.fmp_pipeline.planner.config import PlanConfig
from src.data.fmp_pipeline.planner.universe import TIERS, Security, Universe
from src.data.fmp_pipeline.planner.windowing import (
    intraday_window_days,
    intraday_windows,
    trading_days,
    year_windows,
)

BOOTSTRAP_PRIORITY = 0


@dataclass(frozen=True)
class PlannedTask:
    """One planned request and its size estimate."""

    spec: TaskSpec
    block: int
    rows: float
    bytes_wire: float
    bytes_decoded: float
    measured: bool
    deferred: Optional[str] = None


def bootstrap_specs(config: PlanConfig) -> list[TaskSpec]:
    """Return the first request of every bootstrap list (page 0 for paged ones), at the lowest priority value."""
    specs = []
    for entry in selected(config.priorities, config.intraday.interval):
        if entry.bootstrap:
            params = {"page": 0, "limit": config.page_size} if entry.kind == "paged" else {}
            specs.append(TaskSpec(entry.endpoint, None, params, BOOTSTRAP_PRIORITY))
    return specs


class _Gen:
    def __init__(self, universe: Universe, config: PlanConfig, bounds: dict[str, tuple[date, date]],
                 daily_done: set[str]) -> None:
        self.u = universe
        self.c = config
        self.as_of = universe.as_of
        self.bounds = bounds
        self.daily_done = daily_done
        self.sample = set(universe.sample(config.reconciliation_fixed, config.reconciliation_size))

    def task(self, entry: EndpointSpec, symbol: Optional[str], params: dict[str, Any], sub: int,
             tier: Optional[str], rows: float, deferred: Optional[str] = None) -> PlannedTask:
        spec = TaskSpec(entry.endpoint, symbol, params, priority_of(entry.block, sub), tier)
        return PlannedTask(spec, entry.block, rows, rows * entry.wire_per_row, rows * entry.decoded_per_row,
                           entry.measured, deferred)

    def symbols(self, entry: EndpointSpec) -> list[Security]:
        """Securities an endpoint covers; ``profile`` also covers unresolved candidates."""
        everyone = sorted(self.u.securities.values(), key=lambda s: s.symbol)
        if entry.endpoint == "profile":
            return everyone
        if entry.scope == "universe":
            return everyone
        if entry.scope == "stocks":
            return [s for s in everyone if s.is_stock]
        if entry.scope == "statements":
            listed = self.u.statement_symbols
            return [s for s in everyone if s.is_stock and (listed is None or s.symbol in listed or not s.resolved)]
        if entry.scope == "sample":
            return [s for s in everyone if s.symbol in self.sample]
        return [s for s in everyone if s.is_etf]

    def sub(self, sec: Security) -> int:
        return 1 if sec.is_delisted(self.as_of) else 2

    def listed_range(self, sec: Security, lo: date, hi: date) -> tuple[date, date]:
        """Narrow ``lo`` … ``hi`` to what the security's IPO and delisting dates suggest (estimates only)."""
        if sec.ipo_date is not None:
            lo = max(lo, sec.ipo_date)
        if sec.delisted_date is not None and sec.is_delisted(self.as_of):
            hi = min(hi, sec.delisted_date)
        return lo, hi

    def run(self) -> Iterator[PlannedTask]:
        for entry in selected(self.c.priorities, self.c.intraday.interval):
            if entry.bootstrap:
                continue
            yield from getattr(self, f"_{entry.kind}")(entry)

    # --- one method per kind ---------------------------------------------------------------------------------

    def _list(self, entry: EndpointSpec) -> Iterator[PlannedTask]:
        yield self.task(entry, None, {}, 2, None, entry.rows)

    def _paged(self, entry: EndpointSpec) -> Iterator[PlannedTask]:
        yield self.task(entry, None, {"page": 0, "limit": self.c.page_size}, 2, None, entry.rows)

    def _exchange(self, entry: EndpointSpec) -> Iterator[PlannedTask]:
        for exchange in self.c.exchanges:
            yield self.task(entry, None, {"exchange": exchange}, 2, None, entry.rows)

    def _indicator(self, entry: EndpointSpec) -> Iterator[PlannedTask]:
        for name in self.c.economic_indicators:
            yield self.task(entry, None, {"name": name}, 2, None, entry.rows)

    def _calendar(self, entry: EndpointSpec) -> Iterator[PlannedTask]:
        step = timedelta(days=self.c.calendar_window_days)
        lo = self.c.daily_from
        while lo <= self.as_of:
            hi = min(lo + step - timedelta(days=1), self.as_of)
            share = ((hi - lo).days + 1) / self.c.calendar_window_days
            yield self.task(entry, None, {"from": lo.isoformat(), "to": hi.isoformat()}, 2, None, entry.rows * share)
            lo = hi + timedelta(days=1)

    def _symbol(self, entry: EndpointSpec) -> Iterator[PlannedTask]:
        for sec in self.symbols(entry):
            deferred = None if sec.resolved or entry.endpoint == "profile" else "exchange unknown until profile"
            yield self.task(entry, sec.symbol, {}, self.sub(sec), None, entry.rows, deferred)

    def _statement(self, entry: EndpointSpec) -> Iterator[PlannedTask]:
        for sec in self.symbols(entry):
            deferred = None if sec.resolved else "exchange unknown until profile"
            for period in self.c.statement_periods:
                rows = entry.rows * (4 if period == "quarter" else 1)
                params = {"period": period, "limit": self.c.statement_limit}
                yield self.task(entry, sec.symbol, params, self.sub(sec), None, rows, deferred)

    def _daily(self, entry: EndpointSpec) -> Iterator[PlannedTask]:
        windows = year_windows(self.c.daily_from, self.as_of, self.c.daily_window_years)
        margin = timedelta(days=self.c.intraday.delisted_margin_days)
        for sec in self.symbols(entry):
            deferred = None if sec.resolved else "exchange unknown until profile"
            for lo, hi in windows:
                if sec.is_delisted(self.as_of) and sec.delisted_date is not None and lo > sec.delisted_date + margin:
                    continue
                rows = trading_days(*self.listed_range(sec, lo, hi))
                params = {"from": lo.isoformat(), "to": hi.isoformat()}
                yield self.task(entry, sec.symbol, params, self.sub(sec), None, rows, deferred)

    def _intraday(self, entry: EndpointSpec) -> Iterator[PlannedTask]:
        cfg = self.c.intraday
        days = intraday_window_days(cfg.max_rows or cfg.assumed_max_rows, cfg.bars_per_day, cfg.safety_factor)
        entry = EndpointSpec(entry.endpoint, entry.block, entry.priority, entry.kind, entry.scope,
                             cfg.wire_bytes_per_row, cfg.decoded_bytes_per_row, measured=False)
        earliest = cfg.earliest or self.c.daily_from
        margin = timedelta(days=cfg.delisted_margin_days)
        for sec in self.symbols(entry):
            if sec.tier not in cfg.tiers:
                continue
            sub = 1 + TIERS.index(sec.tier) * 2 + (0 if sec.is_delisted(self.as_of) else 1)
            known = sec.symbol in self.daily_done
            if known and sec.symbol not in self.bounds:
                continue  # daily history finished with no bars: nothing to window
            if known:
                first, last = self.bounds[sec.symbol]
                first = max(first, cfg.earliest) if cfg.earliest else first
            else:
                first, last = self.listed_range(sec, earliest, self.as_of)
            if sec.is_delisted(self.as_of):
                last = last + margin
            deferred = None
            if cfg.max_rows is None:
                deferred = "row cap unknown (paid probe)"
            elif not sec.resolved:
                deferred = "exchange unknown until profile"
            elif not known:
                deferred = "awaiting daily raw"
            for lo, hi in intraday_windows(first, last, self.as_of, days):
                rows = trading_days(max(lo, first), min(hi, last)) * cfg.bars_per_day
                params = {"from": lo.isoformat(), "to": hi.isoformat()}
                yield self.task(entry, sec.symbol, params, sub, sec.tier, rows, deferred)


def generate(universe: Universe, config: PlanConfig, bounds: Optional[dict[str, tuple[date, date]]] = None,
             daily_done: Optional[set[str]] = None) -> Iterator[PlannedTask]:
    """
    Yield every non-bootstrap task the plan implies, in catalogue (block) order.

    Args:
        universe: The assembled universe.
        config: Planner settings.
        bounds: ``{symbol: (first daily bar, last daily bar)}`` from the manifest.
        daily_done: Symbols whose daily raw tasks are all finished.

    Yields:
        Planned tasks with estimates; ``deferred`` ones must not be written to the manifest.
    """
    return _Gen(universe, config, bounds or {}, daily_done or set()).run()
