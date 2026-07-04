"""
P19 shadow store — append-only SQLite log of every intraday poll.

This is the calibration dataset (spec §12): one row per watchlist name per poll,
plus end-of-day O/H/L/C backfilled after the close. It is **p19-specific** and lives
at ``results/p19_penny_intraday/shadow.sqlite`` — separate from the app DB and from
the shared ``DATA_CACHE_DIR`` OHLCV cache.

Single-writer append from the market-hours loop; SQLite is ideal at this volume.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.ml.pipeline.p19_penny_intraday.models.intraday_signal import IntradaySignal
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

DEFAULT_DB_PATH = "results/p19_penny_intraday/shadow.sqlite"

# Columns persisted per poll. Mirrors IntradaySignal.to_dict() + a `date` partition.
_COLUMNS = [
    "date",
    "ts",
    "ticker",
    "source",
    "tier",
    "price",
    "day_open",
    "day_high",
    "day_low",
    "prev_close",
    "pct_from_open",
    "pct_from_prev_close",
    "day_volume",
    "avg_volume_30d",
    "rvol_so_far",
    "dollar_volume_so_far",
    "volume_is_delayed",
    "fresh_catalyst",
    "catalyst_signals",
    "short_squeeze_score",
    "dilution_penalty",
    "sentiment",
    "severity",
    "trigger_reason",
    "eod_open",
    "eod_high",
    "eod_low",
    "eod_close",
]


class ShadowStore:
    """Append-only SQLite store for intraday shadow rows."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cols = ",\n  ".join(f"{c} {self._col_type(c)}" for c in _COLUMNS)
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS shadow_log (\n  id INTEGER PRIMARY KEY AUTOINCREMENT,\n  {cols}\n)"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS ix_shadow_date_ticker ON shadow_log(date, ticker)")
        self._conn.commit()

    @staticmethod
    def _col_type(col: str) -> str:
        if col in ("date", "ts", "ticker", "source", "tier", "trigger_reason", "catalyst_signals", "sentiment"):
            return "TEXT"
        if col in ("volume_is_delayed", "fresh_catalyst"):
            return "INTEGER"
        return "REAL"

    # ── Writes ─────────────────────────────────────────────────────────────

    def append(self, date: str, signal: IntradaySignal) -> None:
        self.append_many(date, [signal])

    def append_many(self, date: str, signals: Iterable[IntradaySignal]) -> int:
        rows = [self._row(date, s) for s in signals]
        if not rows:
            return 0
        placeholders = ",".join("?" for _ in _COLUMNS)
        self._conn.executemany(
            f"INSERT INTO shadow_log ({','.join(_COLUMNS)}) VALUES ({placeholders})",
            rows,
        )
        self._conn.commit()
        return len(rows)

    @staticmethod
    def _row(date: str, s: IntradaySignal) -> List[Any]:
        d = s.to_dict()
        d["date"] = date
        out: List[Any] = []
        for c in _COLUMNS:
            v = d.get(c)
            if isinstance(v, bool):
                v = int(v)
            out.append(v)
        return out

    # ── EOD backfill ───────────────────────────────────────────────────────

    def tickers_for_date(self, date: str) -> List[str]:
        cur = self._conn.execute(
            "SELECT DISTINCT ticker FROM shadow_log WHERE date = ? AND eod_close IS NULL",
            (date,),
        )
        return [r[0] for r in cur.fetchall()]

    def update_eod(self, date: str, ticker: str, ohlc: Dict[str, float]) -> int:
        cur = self._conn.execute(
            "UPDATE shadow_log SET eod_open=?, eod_high=?, eod_low=?, eod_close=? WHERE date=? AND ticker=?",
            (ohlc.get("open"), ohlc.get("high"), ohlc.get("low"), ohlc.get("close"), date, ticker),
        )
        self._conn.commit()
        return cur.rowcount

    # ── Misc ───────────────────────────────────────────────────────────────

    def count(self, date: str = "") -> int:
        if date:
            cur = self._conn.execute("SELECT COUNT(*) FROM shadow_log WHERE date=?", (date,))
        else:
            cur = self._conn.execute("SELECT COUNT(*) FROM shadow_log")
        return int(cur.fetchone()[0])

    def close(self) -> None:
        self._conn.close()
