"""
Reviewed corrections of price-basis errors in a vintage, applied when ``FmpStore`` reads (D11: the built files are
never altered).

Some ``1d_raw`` stretches are not on the raw basis: before a split FMP returned bars that are already
split-adjusted, so the raw series shows no jump on the ex-date and ``adjust()`` divides those bars a second time
(``V-SPLIT-CONSISTENCY`` lists them in ``split_without_jump_candidates.csv``). Sometimes the split row is missing
too, and nothing in the vintage shows the error at all. Price evidence alone is not enough to repair: rebasing
every flat split of vintage 2026-10 fixed two P22 companies and broke five whose split rows had the wrong date or
belonged to another listing. So nothing is corrected automatically. Each row of the override file is a reviewed
decision backed by outside evidence (SEC share counts, a second price source, the filing announcing the split).

File ``<data_root>/overrides/price_basis.csv``:

=============  ===================================================================================================
Column         Meaning
=============  ===================================================================================================
vintage        Dump id the row was reviewed against. Rows for other vintages are ignored: a rebuilt vintage may
               have new data, so its rows are reviewed again.
security_id    Security in that vintage.
symbol         Its symbol, checked against the vintage (guards against an id copied from the wrong row).
ex_date        Split effective date. Bars before it are restored to the raw basis.
ratio          Split ratio, numerator / denominator, as in ``events/splits`` (0.04 for 1-for-25).
from_date      Optional first bar of the adjusted stretch (default: the first bar).
add_split      ``true`` when the vintage lacks the split row itself; ``events`` then gets it.
evidence       Required: where the split and the price basis were confirmed.
=============  ===================================================================================================

Restoring multiplies the prices of bars in ``[from_date, ex_date)`` by ``ratio`` and divides their volume by it,
the inverse of split adjustment. Only daily bars are corrected: the intraday endpoints have not shown the error.
"""

from __future__ import annotations

import csv
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

OVERRIDES_FILE = "overrides/price_basis.csv"
COLUMNS = ("vintage", "security_id", "symbol", "ex_date", "ratio", "from_date", "add_split", "evidence")
PRICE_COLUMNS = ("open", "high", "low", "close")
TRUE_VALUES = {"true", "yes", "1"}
FALSE_VALUES = {"false", "no", "0", ""}


@dataclass(frozen=True)
class PriceBasisOverride:
    """One reviewed correction: bars of ``security_id`` in ``[from_date, ex_date)`` are split-adjusted by ``ratio``."""

    security_id: int
    symbol: str
    ex_date: date
    ratio: float
    from_date: Optional[date]
    add_split: bool
    evidence: str


@dataclass
class OverrideSet:
    """The rows that apply to one vintage, plus what was skipped and why."""

    rows: dict[int, list[PriceBasisOverride]] = field(default_factory=dict)
    other_vintages: int = 0
    problems: list[str] = field(default_factory=list)
    digest: str = ""

    def for_security(self, security_id: int) -> list[PriceBasisOverride]:
        """The overrides of one security (empty when it has none)."""
        return self.rows.get(security_id, [])

    def __len__(self) -> int:
        return sum(len(v) for v in self.rows.values())


def _parse(row: dict[str, str], symbols: Mapping[int, str]) -> PriceBasisOverride:
    """One CSV row of the active vintage; ``ValueError`` names what is wrong with it."""
    missing = [c for c in COLUMNS if c not in row]
    if missing:
        raise ValueError(f"missing columns {missing}")
    security_id = int(row["security_id"])
    symbol = row["symbol"].strip()
    if symbols.get(security_id) != symbol:
        raise ValueError(f"security {security_id} is {symbols.get(security_id)!r} in this vintage, not {symbol!r}")
    ratio = float(row["ratio"])
    if not math.isfinite(ratio) or ratio <= 0 or math.isclose(ratio, 1.0):
        raise ValueError(f"ratio {row['ratio']!r} is not a split ratio")
    ex_date = date.fromisoformat(row["ex_date"].strip())
    from_text = (row["from_date"] or "").strip()
    from_date = date.fromisoformat(from_text) if from_text else None
    if from_date is not None and from_date >= ex_date:
        raise ValueError(f"from_date {from_date} is not before ex_date {ex_date}")
    add_split = (row["add_split"] or "").strip().lower()
    if add_split not in TRUE_VALUES | FALSE_VALUES:
        raise ValueError(f"add_split {row['add_split']!r} is not true/false")
    evidence = (row["evidence"] or "").strip()
    if not evidence:
        raise ValueError("no evidence")
    return PriceBasisOverride(security_id, symbol, ex_date, ratio, from_date, add_split in TRUE_VALUES, evidence)


def load_overrides(path: Path, vintage: str, symbols: Mapping[int, str]) -> OverrideSet:
    """
    Read the override file and keep the valid rows reviewed against ``vintage``.

    A bad row is skipped and reported in ``problems``, never guessed at; one typo must not stop every price read.

    Args:
        path: ``<data_root>/overrides/price_basis.csv``; a missing file means no overrides.
        vintage: Dump id of the store.
        symbols: ``security_id`` to symbol in that vintage.

    Returns:
        The applicable rows per security, the count of rows for other vintages, the problems, and a digest of the
        applied rows (part of the adjusted-cache key).
    """
    result = OverrideSet()
    if not path.exists():
        return result
    applied: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            if (row.get("vintage") or "").strip() != vintage:
                result.other_vintages += 1
                continue
            try:
                override = _parse(row, symbols)
            except (TypeError, ValueError) as exc:
                result.problems.append(f"{path.name} line {line}: {exc}")
                continue
            result.rows.setdefault(override.security_id, []).append(override)
            applied.append(repr(override))
    result.digest = hashlib.sha256("\n".join(applied).encode("utf-8")).hexdigest()[:16] if applied else ""
    return result


def apply_price_overrides(frame: pd.DataFrame, overrides: list[PriceBasisOverride]) -> pd.DataFrame:
    """
    Restore the raw basis of the overridden stretches of one security's daily bars.

    Args:
        frame: Daily bars on a ``DatetimeIndex`` (the ``FmpStore.prices`` shape).
        overrides: That security's overrides.

    Returns:
        A corrected copy, or ``frame`` itself when nothing applies.
    """
    if frame.empty or not overrides:
        return frame
    out = frame.copy()
    index = pd.DatetimeIndex(out.index)
    volume = out["volume"].astype(float) if "volume" in out else None
    for override in overrides:
        mask = index < pd.Timestamp(override.ex_date)
        if override.from_date is not None:
            mask &= index >= pd.Timestamp(override.from_date)
        if not mask.any():
            continue
        for column in PRICE_COLUMNS:
            out.loc[mask, column] = out.loc[mask, column].astype(float) * override.ratio
        if volume is not None:
            volume[mask] = np.round(volume[mask] / override.ratio)
    if volume is not None:
        integer = pd.api.types.is_integer_dtype(frame["volume"]) and volume.notna().all()
        out["volume"] = volume.astype(frame["volume"].dtype) if integer else volume
    return out


def add_override_splits(splits: pd.DataFrame, overrides: list[PriceBasisOverride]) -> pd.DataFrame:
    """
    Add the split rows the vintage lacks (``add_split`` overrides) to one security's splits.

    A split already on the same date is kept as it is: the vintage row wins and the override only rebases prices.

    Args:
        splits: ``date``, ``numerator``, ``denominator`` (the ``FmpStore.events`` shape).
        overrides: That security's overrides.

    Returns:
        The splits, extended and sorted by date when anything was added.
    """
    have = {pd.Timestamp(d).date() for d in splits["date"]}
    new = [{"date": o.ex_date, "numerator": o.ratio, "denominator": 1.0}
           for o in overrides if o.add_split and o.ex_date not in have]
    if not new:
        return splits
    added = pd.DataFrame(new)
    merged = pd.concat([splits, added], ignore_index=True) if not splits.empty else added
    return merged.sort_values("date", key=lambda s: pd.to_datetime(s)).reset_index(drop=True)
