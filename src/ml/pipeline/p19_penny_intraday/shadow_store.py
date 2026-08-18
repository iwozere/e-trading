"""
P19 shadow store — append-only SQLite log of every intraday poll.

This is the calibration dataset (spec §12): one row per watchlist name per poll,
plus end-of-day O/H/L/C and outcome labels backfilled after the close / at T+10.
It is **p19-specific** and lives at ``results/p19_penny_intraday/shadow.sqlite`` —
separate from the app DB and from the shared ``DATA_CACHE_DIR`` OHLCV cache.

Single-writer append from the market-hours loop; SQLite is ideal at this volume
(design-v2.md decision #2 — the v2 spec proposed DuckDB, but no DuckDB
infrastructure exists anywhere in the codebase and the dataset is small; this
stays SQLite and revisits only if calibration queries actually prove slow).

**Schema v2** (design-v2.md §6): additive columns for the structural axis
(denormalised from Layer 0, spec §12.1), momentum_score/momentum_tier (replacing
the unused v1 `severity`), and outcome labels (spec §12.2). Migration is an
idempotent ``ALTER TABLE ADD COLUMN`` pass — safe against both a fresh table and
one that already has some v2 columns from a prior partial run. Confirmed against
the real Pi shadow store (1,741 rows / 3 days as of 2026-08-18) before shipping.
"""

import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.ml.pipeline.p19_penny_intraday.models.intraday_signal import IntradaySignal
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

DEFAULT_DB_PATH = "results/p19_penny_intraday/shadow.sqlite"

# Columns persisted per poll. Mirrors IntradaySignal.to_dict() + a `date` partition.
# v1 columns (unchanged, existing rows always have these):
_V1_COLUMNS = [
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
# v2 additive columns (design-v2.md §6). NULL on any row written before this
# migration ran — never backfilled retroactively (structural features cannot be
# reliably reconstructed after the fact, spec §0.1).
_V2_COLUMNS = [
    "fresh_dilution_filing",
    "structural_grade",
    "dilution_urgency",
    "insider_conviction",
    "runway_quarters",
    "disqualifiers",
    "structural_coverage",
    "is_fpi",
    "momentum_score",
    "momentum_tier",
    "high_time",
    "close_retention",
    "mae_from_alert",
    "mfe_from_alert",
    "ret_t1",
    "ret_t3",
    "ret_t5",
    "ret_t10",
    "dilution_event_within_5d",
    "reverse_split_within_180d",
]
_COLUMNS = _V1_COLUMNS + _V2_COLUMNS


def _as_int_or_none(v: Any) -> Any:
    """bool -> 0/1 for SQLite storage; None passes through; anything else unchanged."""
    return int(v) if isinstance(v, bool) else v


class ShadowStore:
    """Append-only SQLite store for intraday shadow rows."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cols = ",\n  ".join(f"{c} {self._col_type(c)}" for c in _V1_COLUMNS)
        self._conn.execute(
            f"CREATE TABLE IF NOT EXISTS shadow_log (\n  id INTEGER PRIMARY KEY AUTOINCREMENT,\n  {cols}\n)"
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS ix_shadow_date_ticker ON shadow_log(date, ticker)")
        self._migrate_v2_columns()
        self._conn.commit()

    def _migrate_v2_columns(self) -> None:
        """Idempotent additive migration — adds any v2 column missing from an
        existing table. No-op on a table created fresh (already has them via
        _ensure_schema's CREATE TABLE, since _V1_COLUMNS + this pass converge
        on the same final schema either way)."""
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(shadow_log)").fetchall()}
        for col in _V2_COLUMNS:
            if col not in existing:
                self._conn.execute(f"ALTER TABLE shadow_log ADD COLUMN {col} {self._col_type(col)}")
                _logger.info("Shadow store schema v2: added column %s", col)

    @staticmethod
    def _col_type(col: str) -> str:
        _TEXT_COLS = {
            "date",
            "ts",
            "ticker",
            "source",
            "tier",
            "trigger_reason",
            "catalyst_signals",
            "sentiment",
            "structural_grade",
            "momentum_tier",
            "disqualifiers",
            "high_time",
        }
        _INT_COLS = {
            "volume_is_delayed",
            "fresh_catalyst",
            "fresh_dilution_filing",
            "dilution_event_within_5d",
            "reverse_split_within_180d",
            "is_fpi",
        }
        if col in _TEXT_COLS:
            return "TEXT"
        if col in _INT_COLS:
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

    def get_eod(self, date: str, ticker: str) -> Optional[Dict[str, float]]:
        """The first EOD OHLC row for one name/day, or None if not yet backfilled."""
        row = self._conn.execute(
            "SELECT eod_open, eod_high, eod_low, eod_close FROM shadow_log "
            "WHERE date=? AND ticker=? AND eod_close IS NOT NULL LIMIT 1",
            (date, ticker),
        ).fetchone()
        if row is None:
            return None
        return {"open": row[0], "high": row[1], "low": row[2], "close": row[3]}

    # ── Outcome labels (v2 spec §12.2) ──────────────────────────────────────

    def polls_for_date_ticker(self, date: str, ticker: str) -> List[Dict[str, Any]]:
        """All poll rows for one name/day, ascending by timestamp — the raw
        material eod_backfill/label_backfill derive same-day labels from."""
        cols = ["ts", "price", "day_high", "day_low", "momentum_tier"]
        cur = self._conn.execute(
            f"SELECT {','.join(cols)} FROM shadow_log WHERE date=? AND ticker=? ORDER BY ts", (date, ticker)
        )
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def tickers_for_date_needing_labels(self, date: str) -> List[str]:
        """Names with an EOD close but no `close_retention` yet — the
        eod-backfill-time label pass (same-day labels only; ret_t*/dilution
        labels need T+10 and are label_backfill's job)."""
        cur = self._conn.execute(
            "SELECT DISTINCT ticker FROM shadow_log WHERE date=? AND eod_close IS NOT NULL AND close_retention IS NULL",
            (date,),
        )
        return [r[0] for r in cur.fetchall()]

    def update_same_day_labels(self, date: str, ticker: str, labels: Dict[str, Any]) -> int:
        """Write high_time/close_retention/mae_from_alert/mfe_from_alert for
        every poll row of one name/day (same value across the day's rows —
        these are day-level labels, denormalised like the structural snapshot)."""
        cur = self._conn.execute(
            "UPDATE shadow_log SET high_time=?, close_retention=?, mae_from_alert=?, mfe_from_alert=? "
            "WHERE date=? AND ticker=?",
            (
                labels.get("high_time"),
                labels.get("close_retention"),
                labels.get("mae_from_alert"),
                labels.get("mfe_from_alert"),
                date,
                ticker,
            ),
        )
        self._conn.commit()
        return cur.rowcount

    def dates_needing_label_backfill(self, min_age_sessions: int = 10) -> List[str]:
        """Dates with EOD data whose forward-return labels (ret_t10 needs T+10
        sessions) haven't been filled yet. Age is measured in calendar days as
        an upper-bound proxy for trading sessions — label_backfill itself
        checks per-ticker whether T+10 close data actually exists before
        writing, so an early call here is a safe no-op, not a bad label."""
        del min_age_sessions  # session-vs-calendar-day distinction handled by the caller
        cur = self._conn.execute(
            "SELECT DISTINCT date FROM shadow_log WHERE eod_close IS NOT NULL AND ret_t10 IS NULL ORDER BY date"
        )
        return [r[0] for r in cur.fetchall()]

    def tickers_needing_label_backfill(self, date: str) -> List[str]:
        cur = self._conn.execute(
            "SELECT DISTINCT ticker FROM shadow_log WHERE date=? AND eod_close IS NOT NULL AND ret_t10 IS NULL",
            (date,),
        )
        return [r[0] for r in cur.fetchall()]

    def update_forward_labels(self, date: str, ticker: str, labels: Dict[str, Any]) -> int:
        cur = self._conn.execute(
            "UPDATE shadow_log SET ret_t1=?, ret_t3=?, ret_t5=?, ret_t10=?, "
            "dilution_event_within_5d=?, reverse_split_within_180d=? WHERE date=? AND ticker=?",
            (
                labels.get("ret_t1"),
                labels.get("ret_t3"),
                labels.get("ret_t5"),
                labels.get("ret_t10"),
                _as_int_or_none(labels.get("dilution_event_within_5d")),
                _as_int_or_none(labels.get("reverse_split_within_180d")),
                date,
                ticker,
            ),
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
