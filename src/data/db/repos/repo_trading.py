from __future__ import annotations

import uuid
from decimal import Decimal
from datetime import timezone, datetime
from typing import Sequence
from sqlalchemy import select, func, update
from sqlalchemy.orm import Session

from src.data.db.models.model_trading import BotInstance, PerformanceMetric, Trade, Position

UTC = timezone.utc

def _ensure_bot(s: Session, status: str = "stopped", bot_id: int | None = None):
    if bot_id:
        b = s.get(BotInstance, bot_id)
        if b:
            return b
    b = BotInstance(status=status)
    if bot_id:
        b.id = bot_id
    s.add(b); s.flush()
    return b

def _uuid() -> str:
    return str(uuid.uuid4())

def _dec(x: float | Decimal | None) -> Decimal:
    if x is None:
        return Decimal("0")
    return x if isinstance(x, Decimal) else Decimal(str(x))


class BotsRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def upsert_bot(self, bot: dict) -> BotInstance:
        bot_id = bot.get("id")
        obj = self.s.get(BotInstance, bot_id) if bot_id else None

        if obj:
            for k, v in bot.items():
                if k != "id":
                    setattr(obj, k, v)
            return obj

        obj = BotInstance(**bot)
        self.s.add(obj); self.s.flush()
        return obj

    def heartbeat(self, bot_id: str) -> None:
        self.s.execute(
            update(BotInstance)
            .where(BotInstance.id == bot_id)
            .values(last_heartbeat=datetime.now(UTC))
        )


class MetricsRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def add(self, metric: dict) -> PerformanceMetric:
        m = PerformanceMetric(**metric)
        self.s.add(m); self.s.flush()
        return m

    def latest_for_bot(self, bot_id: str, limit: int = 20) -> Sequence[PerformanceMetric]:
        q = (select(PerformanceMetric)
             .where(PerformanceMetric.bot_id == bot_id)
             .order_by(PerformanceMetric.calculated_at.desc())
             .limit(limit))
        return list(self.s.execute(q).scalars())


class TradesRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def add(self, trade: dict) -> Trade:
        t = Trade(**trade)
        self.s.add(t); self.s.flush()
        return t

    def close_trade(self, trade_id: str, **fields) -> None:
        self.s.execute(update(Trade).where(Trade.id == trade_id).values(**fields))

    def open_trades(self, symbol: str | None = None) -> Sequence[Trade]:
        q = select(Trade).where(Trade.status == "open")
        if symbol:
            q = q.where(Trade.symbol == symbol)
        return list(self.s.execute(q).scalars())

    def pnl_summary(self, bot_id: str | None = None):
        q = select(
            func.sum(Trade.net_pnl).label("net_pnl"),
            func.count(Trade.id).label("n_trades")
        ).where(Trade.status == "closed")
        if bot_id:
            q = q.where(Trade.bot_id == bot_id)
        return self.s.execute(q).one()


class PositionsRepo:
    def __init__(self, s: Session) -> None:
        self.s = s

    def ensure_open(
        self,
        *,
        bot_id: str,
        trade_type: str,
        symbol: str,
        direction: str,
        opened_at: datetime | None = None,
        metadata: dict | None = None,
    ):
        # Query with the injected session; return a *bound* instance.
        p = self.s.execute(
            select(Position).where(
                Position.bot_id == bot_id,
                Position.trade_type == trade_type,
                Position.symbol == symbol,
                Position.status == "open",
            )
        ).scalar_one_or_none()
        if p:
            return p

        p = Position(
            id=_uuid(),
            bot_id=bot_id,
            trade_type=trade_type,
            symbol=symbol,
            direction=direction,
            opened_at=(opened_at or datetime.now(UTC)),
            closed_at=None,
            qty_open=Decimal("0"),
            avg_price=None,
            realized_pnl=Decimal("0"),
            status="open",
            extra_metadata=(metadata or None),
        )
        self.s.add(p)
        self.s.flush()
        return p

    def open_positions(self, *, bot_id: str | None = None, symbol: str | None = None) -> Sequence[Position]:
        q = select(Position).where(Position.status == "open")
        if bot_id:
            q = q.where(Position.bot_id == bot_id)
        if symbol:
            q = q.where(Position.symbol == symbol)
        return list(self.s.execute(q).scalars())

    def apply_fill(self, *, position_id: str, action: str, qty: float, price: float,
                   ts: datetime | None = None, close_when_flat: bool = True) -> Position:
        action = action.lower()
        assert action in ("buy", "sell"), "action must be 'buy' or 'sell'"

        p = self.s.get(Position, position_id)
        if not p:
            raise ValueError(f"Position {position_id} not found")

        qty_d = _dec(qty)
        price_d = _dec(price)
        cur_qty = _dec(p.qty_open)
        avg = _dec(p.avg_price)
        realized = _dec(p.realized_pnl)

        add_side = ("buy" if p.direction == "long" else "sell")
        reduce_side = ("sell" if p.direction == "long" else "buy")

        if action == add_side:
            if cur_qty == 0:
                new_qty = qty_d
                new_avg = price_d
            else:
                new_qty = cur_qty + qty_d
                new_avg = (avg * cur_qty + price_d * qty_d) / new_qty
            p.qty_open = new_qty
            p.avg_price = new_avg

        elif action == reduce_side:
            if qty_d > cur_qty:
                raise ValueError(f"Reduce qty {qty_d} > open qty {cur_qty} (pos {position_id})")
            if p.direction == "long":
                realized += (price_d - avg) * qty_d
            else:
                realized += (avg - price_d) * qty_d
            new_qty = cur_qty - qty_d
            p.qty_open = new_qty
            p.realized_pnl = realized
            if new_qty == 0:
                p.avg_price = None
                if close_when_flat:
                    p.status = "closed"
                    p.closed_at = ts or datetime.now(UTC)

        # DO NOT set p.updated_at here unless your model has that column
        self.s.flush()
        return p

    def close_if_flat(self, *, position_id: str, ts: datetime | None = None) -> Position:
        p = self.s.get(Position, position_id)
        if not p:
            raise ValueError(f"Position {position_id} not found")
        if _dec(p.qty_open) == 0 and p.status != "closed":
            p.status = "closed"
            p.closed_at = ts or datetime.now(UTC)
        self.s.flush()
        return p

    def mark_closed(self, *, position_id: str, ts: datetime | None = None) -> Position:
        p = self.s.get(Position, position_id)
        if not p:
            raise ValueError(f"Position {position_id} not found")
        p.status = "closed"
        p.closed_at = ts or datetime.now(UTC)
        if _dec(p.qty_open) != 0:
            p.qty_open = Decimal("0")
            p.avg_price = None
        self.s.flush()
        return p

    # -------- info helpers (no DB writes) ---------------------------------
    @staticmethod
    def unrealized_pnl(p: Position, *, mark_price: float | Decimal) -> Decimal:
        mark = _dec(mark_price)
        qty = _dec(p.qty_open)
        if qty == 0 or p.avg_price is None:
            return Decimal("0")
        avg = _dec(p.avg_price)
        return (mark - avg) * qty if p.direction == "long" else (avg - mark) * qty

    @staticmethod
    def total_pnl(p: Position, *, mark_price: float | Decimal | None = None) -> Decimal:
        realized = _dec(p.realized_pnl)
        return realized if mark_price is None else realized + PositionsRepo.unrealized_pnl(p, mark_price=mark_price)
