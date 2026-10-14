"""
Schema inference over FMP JSON bodies and synthetic fixture generation (spec §4.2.1, D5).

``infer_schema`` records field names, JSON types, null rates and string formats — never values — so the schema
can be committed. ``synthesize`` turns a schema back into fake records that have the same shape, for the
fixtures the mock server and the builders are tested against.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

_FORMATS = (
    ("date", re.compile(r"^\d{4}-\d{2}-\d{2}$")),
    ("datetime", re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?$")),
    ("datetime_tz", re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})$")),
    ("year", re.compile(r"^\d{4}$")),
    ("numeric_string", re.compile(r"^-?\d+(\.\d+)?$")),
)
_SYNTHETIC_STRINGS = {
    "date": "2001-02-05",
    "datetime": "2001-02-05 09:30:00",
    "datetime_tz": "2001-02-05T09:30:00.000Z",
    "year": "2001",
    "numeric_string": "1.5",
}


@dataclass
class FieldSchema:
    """Observed shape of one field across all records."""

    name: str
    types: Counter[str] = field(default_factory=Counter)
    formats: Counter[str] = field(default_factory=Counter)
    present: int = 0

    def null_rate(self, records: int) -> float:
        """Fraction of records where the field is missing or null."""
        return 1.0 - (self.present - self.types.get("null", 0)) / records if records else 0.0


@dataclass
class Schema:
    """Shape of a JSON body: its container kind, record count and per-field schemas."""

    container: str
    records: int
    fields: dict[str, FieldSchema]
    wrapper_key: str = ""


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    return "object"


def _string_format(value: str) -> str:
    for name, pattern in _FORMATS:
        if pattern.match(value):
            return name
    return "text"


def infer_schema(body: Any) -> Schema:
    """
    Infer the schema of a parsed JSON body.

    Args:
        body: A list of records, an object, or an object wrapping a list under ``historical``/``data``/``results``.

    Returns:
        The inferred schema; ``container`` is ``list``, ``object``, ``wrapped_list`` or ``scalar``.
    """
    wrapper = ""
    if isinstance(body, dict):
        for key in ("historical", "data", "results"):
            if isinstance(body.get(key), list):
                wrapper, body = key, body[key]
                break
    if isinstance(body, list):
        records = [r for r in body if isinstance(r, dict)]
        container = "wrapped_list" if wrapper else "list"
    elif isinstance(body, dict):
        records, container = [body], "object"
    else:
        return Schema("scalar", 0, {})

    fields: dict[str, FieldSchema] = {}
    for record in records:
        for name, value in record.items():
            schema = fields.setdefault(name, FieldSchema(name))
            schema.present += 1
            kind = _json_type(value)
            schema.types[kind] += 1
            if kind == "str":
                schema.formats[_string_format(value)] += 1
    return Schema(container, len(records), fields, wrapper)


def _synthetic_value(schema: FieldSchema, seq: int) -> Any:
    non_null = [t for t, _ in schema.types.most_common() if t != "null"]
    kind = non_null[0] if non_null else "null"
    if kind == "str":
        fmt = schema.formats.most_common(1)[0][0] if schema.formats else "text"
        return _SYNTHETIC_STRINGS.get(fmt, f"SYNTH{seq}")
    if kind == "int":
        return 100 + seq
    if kind == "float":
        return 10.5 + seq
    if kind == "bool":
        return seq % 2 == 0
    if kind == "list":
        return []
    if kind == "object":
        return {}
    return None


def synthesize(schema: Schema, count: int = 2) -> Any:
    """
    Build a synthetic body with the same container and fields as ``schema``; every value is fake.

    Args:
        schema: Output of ``infer_schema``.
        count: Number of records for list containers.

    Returns:
        A JSON-serializable body.
    """
    def record(seq: int) -> dict[str, Any]:
        return {name: _synthetic_value(f, seq) for name, f in schema.fields.items()}

    if schema.container == "object":
        return record(0)
    if schema.container == "scalar" or not schema.fields:
        return []
    records = [record(i) for i in range(count)]
    return {schema.wrapper_key: records} if schema.container == "wrapped_list" else records
