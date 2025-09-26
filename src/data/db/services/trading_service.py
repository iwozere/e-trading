"""
Trading Service
---------------
Orchestrates trading DB operations (bots, trades, positions, metrics)
via the repository layer. Returns plain dicts/primitives.

All operations use a single Unit-of-Work:
    with _db.uow() as r:
        ... r.<repo>
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.data.db.services.database_service import get_database_service

# Resolve once per module to avoid repeated lookups.
_db = get_database_service()

# ---------- Bots ----------
def upsert_bot(bot: Dict[str, Any]) -> Dict[str, Any]:
    with _db.uow() as r:
        row = r.bots.upsert_bot(bot)
        return _bot_to_dict(row)

def heartbeat(bot_id: str) -> None:
    with _db.uow() as r:
        r.bots.heartbeat(bot_id)


# ---------- Trades ----------
def add_trade(trade: Dict[str, Any]) -> Dict[str, Any]:
    with _db.uow() as r:
        row = r.trades.add(trade)
        return _trade_to_dict(row)

def close_trade(trade_id: str, **fields) -> bool:
    if not fields:
        return False
    with _db.uow() as r:
        r.trades.close_trade(trade_id, **fields)
        return True

def get_open_trades(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    with _db.uow() as r:
        rows = r.trades.open_trades(symbol)
        return [_trade_to_dict(t) for t in rows]

def get_pnl_summary(bot_id: Optional[str] = None) -> Dict[str, Any]:
    with _db.uow() as r:
        agg = r.trades.pnl_summary(bot_id)
        return {
            "net_pnl": float(agg.net_pnl or 0),
            "n_trades": int(agg.n_trades or 0),
        }


# ---------- Positions ----------
def ensure_open_position(
    *,
    bot_id: str,
    trade_type: str,
    symbol: str,
    direction: str,
    opened_at: Optional[datetime] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    with _db.uow() as r:
        row = r.positions.ensure_open(
            bot_id=bot_id,
            trade_type=trade_type,
            symbol=symbol,
            direction=direction,
            opened_at=opened_at,
            metadata=metadata,
        )
        return _position_to_dict(row)

def apply_fill(
    position_id: str,
    *,
    action: str,
    qty: float,
    price: float,
    ts: Optional[datetime] = None,
    close_when_flat: bool = True,
) -> Dict[str, Any]:
    with _db.uow() as r:
        row = r.positions.apply_fill(
            position_id=position_id,
            action=action,
            qty=qty,
            price=price,
            ts=ts,
            close_when_flat=close_when_flat,
        )
        return _position_to_dict(row)

def close_if_flat(position_id: str, ts: Optional[datetime] = None) -> Dict[str, Any]:
    with _db.uow() as r:
        row = r.positions.close_if_flat(position_id=position_id, ts=ts)
        return _position_to_dict(row)

def mark_closed(position_id: str, ts: Optional[datetime] = None) -> Dict[str, Any]:
    with _db.uow() as r:
        row = r.positions.mark_closed(position_id=position_id, ts=ts)
        return _position_to_dict(row)

def get_open_positions(
    *,
    bot_id: Optional[str] = None,
    symbol: Optional[str] = None,
) -> List[Dict[str, Any]]:
    with _db.uow() as r:
        rows = r.positions.open_positions(bot_id=bot_id, symbol=symbol)
        return [_position_to_dict(p) for p in rows]


# ---------- Metrics ----------
def add_metric(metric: Dict[str, Any]) -> Dict[str, Any]:
    with _db.uow() as r:
        row = r.metrics.add(metric)
        return _metric_to_dict(row)

def latest_metrics(bot_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    with _db.uow() as r:
        rows = r.metrics.latest_for_bot(bot_id, limit=limit)
        return [_metric_to_dict(m) for m in rows]


# ---------- DTO helpers ----------
def _bot_to_dict(b) -> Dict[str, Any]:
    return {
        "id": b.id,
        "type": b.type,
        "config_file": b.config_file,
        "status": b.status,
        "started_at": b.started_at,
        "last_heartbeat": b.last_heartbeat,
        "error_count": b.error_count,
        "current_balance": float(b.current_balance) if b.current_balance is not None else None,
        "total_pnl": float(b.total_pnl) if b.total_pnl is not None else None,
        "extra_metadata": b.extra_metadata,
        "created_at": b.created_at,
        "updated_at": b.updated_at,
    }

def _trade_to_dict(t) -> Dict[str, Any]:
    return {
        "id": t.id,
        "bot_id": t.bot_id,
        "trade_type": t.trade_type,
        "strategy_name": t.strategy_name,
        "entry_logic_name": t.entry_logic_name,
        "exit_logic_name": t.exit_logic_name,
        "symbol": t.symbol,
        "interval": t.interval,
        "entry_time": t.entry_time,
        "exit_time": t.exit_time,
        "entry_price": float(t.entry_price) if t.entry_price is not None else None,
        "exit_price": float(t.exit_price) if t.exit_price is not None else None,
        "size": float(t.size) if t.size is not None else None,
        "direction": t.direction,
        "commission": float(t.commission) if t.commission is not None else None,
        "gross_pnl": float(t.gross_pnl) if t.gross_pnl is not None else None,
        "net_pnl": float(t.net_pnl) if t.net_pnl is not None else None,
        "pnl_percentage": float(t.pnl_percentage) if t.pnl_percentage is not None else None,
        "exit_reason": t.exit_reason,
        "status": t.status,
        "extra_metadata": t.extra_metadata,
        "created_at": t.created_at,
        "updated_at": t.updated_at,
        "position_id": t.position_id,
    }

def _position_to_dict(p) -> Dict[str, Any]:
    return {
        "id": p.id,
        "bot_id": p.bot_id,
        "trade_type": p.trade_type,
        "symbol": p.symbol,
        "direction": p.direction,
        "opened_at": p.opened_at,
        "closed_at": p.closed_at,
        "qty_open": float(p.qty_open) if p.qty_open is not None else 0.0,
        "avg_price": float(p.avg_price) if p.avg_price is not None else None,
        "realized_pnl": float(p.realized_pnl) if p.realized_pnl is not None else 0.0,
        "status": p.status,
        "extra_metadata": p.extra_metadata,
    }

def _metric_to_dict(m) -> Dict[str, Any]:
    return {
        "id": m.id,
        "bot_id": m.bot_id,
        "trade_type": m.trade_type,
        "symbol": m.symbol,
        "interval": m.interval,
        "entry_logic_name": m.entry_logic_name,
        "exit_logic_name": m.exit_logic_name,
        "metrics": m.metrics,
        "calculated_at": m.calculated_at,
        "created_at": m.created_at,
    }
