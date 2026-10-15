"""
Structural validation of a freshly built vintage (spec §7.4): the gate before ``current.txt`` is switched.

"Structural" means the vintage is well-formed, not that the data is right (that is ``fmp validate``, WP-A):
every recorded file exists, opens, and has the recorded row count; price files have the expected schema, one
security each, matching their file name, with strictly increasing non-null time keys; ``security_id`` is unique
in ``securities`` and every price file belongs to a known security.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.data.fmp_pipeline.builders.prices import DAILY_SCHEMA, INTRADAY_SCHEMA

SECURITIES = "reference/securities.parquet"
MAX_ERRORS = 50


def _price_errors(path: Path, rel: str, known: set[int]) -> list[str]:
    daily = rel.startswith("prices/1d_raw/")
    expected = DAILY_SCHEMA if daily else INTRADAY_SCHEMA
    key = "date" if daily else "ts_utc"
    table = pq.read_table(path, columns=["security_id", key])
    errors = []
    if not pq.read_schema(path).equals(expected):
        errors.append(f"{rel}: unexpected schema")
    ids = table.column("security_id").to_numpy(zero_copy_only=False)
    sid = int(Path(rel).stem)
    if len(ids) and (ids != sid).any():
        errors.append(f"{rel}: rows of another security")
    if sid not in known:
        errors.append(f"{rel}: security {sid} not in securities")
    column = table.column(key)
    if column.null_count:
        errors.append(f"{rel}: {column.null_count} null {key}")
    else:
        values = column.to_numpy(zero_copy_only=False)
        if len(values) > 1 and not (values[1:] > values[:-1]).all():
            errors.append(f"{rel}: {key} not strictly increasing")
    return errors


def validate_structure(out_dir: Path, outputs: dict[str, int]) -> list[str]:
    """
    Check a built vintage.

    Args:
        out_dir: The vintage directory.
        outputs: ``{relative path: rows written}`` as recorded by the builder.

    Returns:
        Error messages (at most ``MAX_ERRORS``); empty means the vintage may be switched to.
    """
    errors: list[str] = []
    if SECURITIES not in outputs:
        return [f"{SECURITIES} missing"]
    ids = pq.read_table(out_dir / SECURITIES, columns=["security_id"]).column("security_id")
    values = ids.to_numpy(zero_copy_only=False)
    if ids.null_count or len(np.unique(values)) != len(values):
        errors.append("securities: security_id null or not unique")
    known = set(int(v) for v in values) if not ids.null_count else set()
    for rel, rows in sorted(outputs.items()):
        if len(errors) >= MAX_ERRORS:
            errors.append("… more errors not shown")
            break
        path = out_dir / rel
        try:
            actual = pq.read_metadata(path).num_rows
        except (OSError, ValueError) as exc:
            errors.append(f"{rel}: unreadable ({exc})")
            continue
        if actual != rows:
            errors.append(f"{rel}: {actual} rows, recorded {rows}")
        if rel.startswith("prices/") and not rel.startswith("prices/recon/"):
            errors += _price_errors(path, rel, known)
    return errors
