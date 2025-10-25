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
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from src.data.db.services.database_service import get_database_service
from src.trading.services.bot_config_validator import validate_database_bot_record, BotConfigValidator

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

def get_bot_by_id(bot_id: int) -> Optional[Dict[str, Any]]:
    """Get a specific bot by ID."""
    with _db.uow() as r:
        from src.data.db.models.model_trading import BotInstance
        bot = r.s.get(BotInstance, bot_id)
        return _bot_to_dict(bot) if bot else None

def get_enabled_bots(user_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get all enabled bots, optionally filtered by user."""
    with _db.uow() as r:
        from sqlalchemy import select
        from src.data.db.models.model_trading import BotInstance

        query = select(BotInstance).where(BotInstance.status != 'disabled')
        if user_id:
            query = query.where(BotInstance.user_id == user_id)

        bots = r.s.execute(query).scalars().all()
        return [_bot_to_dict(bot) for bot in bots]

def get_bots_by_status(status: str, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get bots by status, optionally filtered by user."""
    with _db.uow() as r:
        from sqlalchemy import select
        from src.data.db.models.model_trading import BotInstance

        query = select(BotInstance).where(BotInstance.status == status)
        if user_id:
            query = query.where(BotInstance.user_id == user_id)

        bots = r.s.execute(query).scalars().all()
        return [_bot_to_dict(bot) for bot in bots]

def update_bot_status(bot_id: int, status: str, error_message: Optional[str] = None) -> bool:
    """Update bot status and optionally increment error count."""
    with _db.uow() as r:
        from sqlalchemy import update
        from src.data.db.models.model_trading import BotInstance

        update_data = {"status": status, "updated_at": datetime.now()}

        if status == "error" and error_message:
            # Increment error count and store error in extra_metadata
            bot = r.s.get(BotInstance, bot_id)
            if bot:
                error_count = (bot.error_count or 0) + 1
                metadata = bot.extra_metadata or {}
                metadata["last_error"] = error_message
                metadata["last_error_time"] = datetime.now().isoformat()

                update_data.update({
                    "error_count": error_count,
                    "extra_metadata": metadata
                })

        result = r.s.execute(
            update(BotInstance)
            .where(BotInstance.id == bot_id)
            .values(**update_data)
        )
        return result.rowcount > 0

def update_bot_performance(bot_id: int, current_balance: Optional[float] = None,
                          total_pnl: Optional[float] = None) -> bool:
    """Update bot performance metrics."""
    with _db.uow() as r:
        from sqlalchemy import update
        from src.data.db.models.model_trading import BotInstance

        update_data = {"updated_at": datetime.now()}

        if current_balance is not None:
            update_data["current_balance"] = current_balance
        if total_pnl is not None:
            update_data["total_pnl"] = total_pnl

        result = r.s.execute(
            update(BotInstance)
            .where(BotInstance.id == bot_id)
            .values(**update_data)
        )
        return result.rowcount > 0


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
        "user_id": b.user_id,
        "type": b.type,
        "config": b.config,
        "description": b.description,
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

# ---------- Configuration Validation ----------
def validate_bot_configuration(bot_id: int) -> Tuple[bool, List[str], List[str]]:
    """
    Validate a bot's configuration from the database.

    Args:
        bot_id: Bot ID to validate

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    with _db.uow() as r:
        from src.data.db.models.model_trading import BotInstance

        bot = r.s.get(BotInstance, bot_id)
        if not bot:
            return False, [f"Bot with ID {bot_id} not found"], []

        bot_dict = _bot_to_dict(bot)
        return validate_database_bot_record(bot_dict)

def validate_all_bot_configurations(user_id: Optional[int] = None) -> Dict[int, Tuple[bool, List[str], List[str]]]:
    """
    Validate configurations for all bots, optionally filtered by user.

    Args:
        user_id: Optional user ID to filter bots

    Returns:
        Dictionary mapping bot_id to (is_valid, errors, warnings)
    """
    bots = get_enabled_bots(user_id)
    results = {}

    for bot in bots:
        bot_id = bot["id"]
        is_valid, errors, warnings = validate_database_bot_record(bot)
        results[bot_id] = (is_valid, errors, warnings)

    return results

def get_bot_configuration_schema() -> Dict[str, Any]:
    """
    Get the expected configuration schema for bot configurations.

    Returns:
        Dictionary describing the expected configuration structure
    """
    return {
        "type": "object",
        "required": ["id", "name", "enabled", "symbol", "broker", "strategy"],
        "properties": {
            "id": {"type": "string", "description": "Unique bot identifier"},
            "name": {"type": "string", "description": "Human-readable bot name"},
            "enabled": {"type": "boolean", "description": "Whether the bot is enabled"},
            "symbol": {"type": "string", "description": "Trading symbol (e.g., BTCUSDT)"},
            "broker": {
                "type": "object",
                "required": ["type", "trading_mode", "name"],
                "properties": {
                    "type": {"type": "string", "enum": ["binance", "paper", "alpaca", "interactive_brokers"]},
                    "trading_mode": {"type": "string", "enum": ["paper", "live"]},
                    "name": {"type": "string", "description": "Broker instance name"},
                    "cash": {"type": "number", "description": "Initial cash for paper trading"}
                }
            },
            "strategy": {
                "type": "object",
                "required": ["type"],
                "properties": {
                    "type": {"type": "string", "description": "Strategy type (e.g., CustomStrategy)"},
                    "parameters": {"type": "object", "description": "Strategy-specific parameters"}
                }
            },
            "data": {
                "type": "object",
                "properties": {
                    "data_source": {"type": "string", "description": "Data source provider"},
                    "interval": {"type": "string", "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]},
                    "lookback_bars": {"type": "integer", "minimum": 1, "maximum": 5000}
                }
            },
            "trading": {
                "type": "object",
                "properties": {
                    "position_size": {"type": "number", "minimum": 0, "maximum": 1},
                    "max_positions": {"type": "integer", "minimum": 1}
                }
            },
            "risk_management": {
                "type": "object",
                "properties": {
                    "max_position_size": {"type": "number", "minimum": 0},
                    "stop_loss_pct": {"type": "number", "minimum": 0},
                    "take_profit_pct": {"type": "number", "minimum": 0},
                    "max_daily_loss": {"type": "number", "minimum": 0},
                    "max_daily_trades": {"type": "integer", "minimum": 1}
                }
            },
            "notifications": {
                "type": "object",
                "properties": {
                    "position_opened": {"type": "boolean"},
                    "position_closed": {"type": "boolean"},
                    "email_enabled": {"type": "boolean"},
                    "telegram_enabled": {"type": "boolean"},
                    "error_notifications": {"type": "boolean"},
                    "performance_summaries": {"type": "string", "enum": ["none", "daily", "weekly", "monthly"]},
                    "risk_alerts": {"type": "boolean"}
                }
            }
        }
    }