"""
P20 Kestrel — /pos command parser and position flow handler.

Parses Telegram /pos commands and writes to k20_positions.
Grammar per §9.4:

  /pos add TICKER SLEEVE ENTRY_PX SIZE_PCT [stop=PX] [t1=PX] [t2=PX] [trail=PCT]
  /pos scale TICKER THIRD_N PX
  /pos stop TICKER PX
  /pos close TICKER PX [reason]
  /pos list

Confirmation required before 'add' creates the position (caller must
prompt user and call confirm_add(pending_id)).
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.db.services.kestrel_service import KestrelService as _KestrelService

_kestrel = _KestrelService()
get_open_positions = _kestrel.get_open_positions
insert_position = _kestrel.insert_position
update_position = _kestrel.update_position
upsert_watchlist = _kestrel.upsert_watchlist
from src.notification.logger import setup_logger

_logger = setup_logger(__name__)

# Defaults per spec §9.4
_DEFAULT_STOP_MULT = 0.75
_DEFAULT_T1_MULT = 1.35
_DEFAULT_T2_MULT = 1.60
_DEFAULT_TRAIL_PCT = 20.0

_ADD_RE = re.compile(
    r"^/pos\s+add\s+(?P<ticker>[A-Z]+)\s+(?P<sleeve>[A-C])\s+(?P<entry>[0-9.]+)\s+(?P<size>[0-9.]+)"
    r"(?:\s+stop=(?P<stop>[0-9.]+))?"
    r"(?:\s+t1=(?P<t1>[0-9.]+))?"
    r"(?:\s+t2=(?P<t2>[0-9.]+))?"
    r"(?:\s+trail=(?P<trail>[0-9.]+))?$",
    re.IGNORECASE,
)
_SCALE_RE = re.compile(
    r"^/pos\s+scale\s+(?P<ticker>[A-Z]+)\s+(?P<third>[123])\s+(?P<px>[0-9.]+)$",
    re.IGNORECASE,
)
_STOP_RE = re.compile(
    r"^/pos\s+stop\s+(?P<ticker>[A-Z]+)\s+(?P<px>[0-9.]+)$",
    re.IGNORECASE,
)
_CLOSE_RE = re.compile(
    r"^/pos\s+close\s+(?P<ticker>[A-Z]+)\s+(?P<px>[0-9.]+)(?:\s+(?P<reason>.+))?$",
    re.IGNORECASE,
)
_LIST_RE = re.compile(r"^/pos\s+list$", re.IGNORECASE)


class PosCommandError(ValueError):
    """Raised when a /pos command has invalid syntax or references a missing position."""


def _parse_add(text: str) -> Dict[str, Any]:
    """
    Parse /pos add command.

    Args:
        text: Command text.

    Returns:
        Dict of parsed fields including computed defaults.

    Raises:
        PosCommandError: On parse failure.
    """
    m = _ADD_RE.match(text.strip())
    if not m:
        raise PosCommandError("Usage: /pos add TICKER SLEEVE ENTRY_PX SIZE_PCT [stop=PX] [t1=PX] [t2=PX] [trail=PCT]")

    ticker = m.group("ticker").upper()
    sleeve = m.group("sleeve").upper()
    entry = float(m.group("entry"))
    size = float(m.group("size"))

    stop = float(m.group("stop")) if m.group("stop") else round(entry * _DEFAULT_STOP_MULT, 4)
    t1 = float(m.group("t1")) if m.group("t1") else round(entry * _DEFAULT_T1_MULT, 4)
    t2 = float(m.group("t2")) if m.group("t2") else round(entry * _DEFAULT_T2_MULT, 4)
    trail = float(m.group("trail")) if m.group("trail") else _DEFAULT_TRAIL_PCT

    return {
        "ticker": ticker,
        "sleeve": sleeve,
        "entry_date": date.today(),
        "entry_px": entry,
        "size_pct": size,
        "stop_px": stop,
        "t1_px": t1,
        "t2_px": t2,
        "trail_pct": trail,
        "realized_thirds": 0,
        "status": "pending",
    }


def echo_card(pos: Dict[str, Any]) -> str:
    """
    Format a position echo card for Telegram confirmation.

    Args:
        pos: Position dict (pending or confirmed).

    Returns:
        Formatted string for display.
    """
    return (
        f"📋 Position Preview — {pos['ticker']} (Sleeve {pos['sleeve']})\n"
        f"  Entry:  ${pos['entry_px']:.4f} @ {pos.get('size_pct')}%\n"
        f"  Stop:   ${pos['stop_px']:.4f}\n"
        f"  T1:     ${pos['t1_px']:.4f}\n"
        f"  T2:     ${pos['t2_px']:.4f}\n"
        f"  Trail:  {pos['trail_pct']}%\n"
        f"\nTap ✅ Confirm to open position."
    )


def confirm_add(pending_pos: Dict[str, Any]) -> Dict[str, Any]:
    """
    Confirm and insert a pending /pos add position.

    Updates watchlist state to active_position.

    Args:
        pending_pos: Dict returned from _parse_add().

    Returns:
        Confirmed position dict with id.
    """
    confirmed = dict(pending_pos)
    confirmed["status"] = "open"
    pos_id = insert_position(confirmed)
    confirmed["id"] = pos_id

    # Transition watchlist state
    upsert_watchlist(
        {
            "ticker": confirmed["ticker"],
            "sleeve": confirmed["sleeve"],
            "state": "active_position",
        }
    )
    _logger.info(
        "Position opened: %s Sleeve %s @ $%.4f", confirmed["ticker"], confirmed["sleeve"], confirmed["entry_px"]
    )
    return confirmed


def handle_command(text: str) -> Tuple[str, Dict[str, Any] | None]:
    """
    Dispatch a /pos command.

    Args:
        text: Full command text from Telegram message.

    Returns:
        Tuple of (reply_text, data_dict_or_None).
        For 'add' commands, data_dict is the pending position for confirmation.

    Raises:
        PosCommandError: On unknown or malformed command.
    """
    text = text.strip()

    if _LIST_RE.match(text):
        positions = get_open_positions()
        if not positions:
            return "No open positions.", None
        lines = ["Open positions:"]
        for p in positions:
            entry = p.get("entry_px") or 0
            stop = p.get("stop_px") or 0
            lines.append(f"  {p['ticker']} (Sleeve {p.get('sleeve', '?')}): entry ${entry:.4f} | stop ${stop:.4f}")
        return "\n".join(lines), None

    if text.lower().startswith("/pos add"):
        pending = _parse_add(text)
        return echo_card(pending), pending

    m = _SCALE_RE.match(text)
    if m:
        ticker = m.group("ticker").upper()
        third = int(m.group("third"))
        px = float(m.group("px"))
        positions = [p for p in get_open_positions() if p.get("ticker") == ticker]
        if not positions:
            raise PosCommandError(f"No open position for {ticker}")
        pos = positions[0]
        realized_thirds = int(pos.get("realized_thirds") or 0) + 1
        update_position(
            pos["id"],
            {
                "realized_thirds": realized_thirds,
                "notes": f"Scaled 1/3 #{third} @ ${px:.4f}",
            },
        )
        # Move stop to breakeven after first third
        if third == 1 and pos.get("entry_px"):
            update_position(pos["id"], {"stop_px": float(pos["entry_px"])})
        return f"Scaled 1/3 #{third} for {ticker} @ ${px:.4f}. Realized thirds: {realized_thirds}.", None

    m = _STOP_RE.match(text)
    if m:
        ticker = m.group("ticker").upper()
        px = float(m.group("px"))
        positions = [p for p in get_open_positions() if p.get("ticker") == ticker]
        if not positions:
            raise PosCommandError(f"No open position for {ticker}")
        update_position(positions[0]["id"], {"stop_px": px})
        return f"Stop updated for {ticker} to ${px:.4f}.", None

    m = _CLOSE_RE.match(text)
    if m:
        ticker = m.group("ticker").upper()
        px = float(m.group("px"))
        reason = m.group("reason") or "manual close"
        positions = [p for p in get_open_positions() if p.get("ticker") == ticker]
        if not positions:
            raise PosCommandError(f"No open position for {ticker}")
        pos = positions[0]
        entry_px = float(pos.get("entry_px") or px)
        pnl_pct = (px - entry_px) / entry_px
        update_position(
            pos["id"],
            {
                "status": "closed",
                "notes": f"Closed @ ${px:.4f} ({pnl_pct:+.1%}) — {reason}",
            },
        )
        upsert_watchlist(
            {
                "ticker": ticker,
                "sleeve": pos.get("sleeve", "?"),
                "state": "expired",
            }
        )
        return f"Position {ticker} closed @ ${px:.4f} ({pnl_pct:+.1%}). Reason: {reason}.", None

    raise PosCommandError("Unknown /pos command. Use: add | scale | stop | close | list")
