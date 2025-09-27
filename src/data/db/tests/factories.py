# tests/factories.py
from __future__ import annotations
import random
import json
import uuid
from datetime import datetime, timezone, timedelta

from sqlalchemy.orm import Session

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from datetime import datetime, timezone

# import your real models
from src.data.db.models.model_users import User, AuthIdentity  # your files
from src.data.db.models.model_telegram import (
    TelegramAlert, TelegramSchedule, TelegramFeedback,
    TelegramCommandAudit, TelegramSetting, TelegramVerificationCode
)
from src.data.db.models.model_trading import Position  # (above) or your own path
from src.data.db.models.model_trading import BotInstance

UTC = timezone.utc

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def ensure_bot(s, bot_id: str, *, type: str = "paper", status: str = "running"):
    if not s.get(BotInstance, bot_id):
        s.add(BotInstance(id=bot_id, type=type, status=status))
        s.flush()
    return bot_id

class RNG:
    """Deterministic rng holder, set seed in tests for reproducibility."""
    def __init__(self, seed: int = 42) -> None:
        self.r = random.Random(seed)

    def choice(self, seq):
        return self.r.choice(seq)

    def randfloat(self, a: float, b: float) -> float:
        return self.r.uniform(a, b)

    def randint(self, a: int, b: int) -> int:
        return self.r.randint(a, b)

    def uuid(self) -> str:
        return str(uuid.UUID(int=self.r.getrandbits(128)))

def iso_now(offset_sec: int = 0) -> str:
    return (datetime.now(UTC) + timedelta(seconds=offset_sec)).isoformat()

# ----------------------------- USERS & AUTH ----------------------------------

def make_user(s: Session, *, email: str | None = None, role: str = "trader",
              is_active: bool = True) -> User:
    u = User(email=email, role=role, is_active=is_active)  # users table shape from your schema
    s.add(u); s.flush()
    return u

def add_telegram_identity(s, user_id: int, telegram_user_id: str, **meta):
    ai = AuthIdentity(
        user_id=user_id,
        provider="telegram",
        external_id=str(telegram_user_id),
        identity_metadata=meta or None,  # <-- renamed
    )
    s.add(ai); s.flush()
    return ai

# -------------------------------- POSITIONS ----------------------------------

def make_position(s: Session, rng: RNG, *,
                  bot_id: str | None = None,
                  trade_type: str | None = None,
                  symbol: str | None = None,
                  direction: str | None = None,
                  qty_open: float | None = None,
                  avg_price: float | None = None,
                  status: str | None = None) -> Position:
    bot_id = bot_id or f"bot-{rng.randint(100, 999)}"
    trade_type = trade_type or rng.choice(["paper", "live", "optimization"])
    symbol = symbol or rng.choice(["AAPL", "NVDA", "TSLA", "MRNA", "SMCI"])
    direction = direction or rng.choice(["long", "short"])
    qty_open = qty_open if qty_open is not None else round(rng.randfloat(1, 10), 4)
    avg_price = avg_price if avg_price is not None else round(rng.randfloat(10, 400), 2)
    status = status or "open"

    # Ensure FK: positions.bot_id -> bot_instances.id
    ensure_bot(s, bot_id)

    p = Position(
        id=rng.uuid(),
        bot_id=bot_id,
        trade_type=trade_type,
        symbol=symbol,
        direction=direction,
        opened_at=datetime.now(UTC),
        closed_at=None if status == "open" else datetime.now(UTC),
        qty_open=qty_open,
        avg_price=avg_price,
        realized_pnl=None,
        status=status,
        extra_metadata={"note": "factory"}
    )
    s.add(p); s.flush()
    return p

# -------------------------------- TELEGRAM -----------------------------------

def make_alert(
    s: Session,
    rng: RNG,
    *,
    user_id: int,
    ticker: str | None = None,
    condition: str | None = None,
    price: float | None = None,
    timeframe: str | None = None,
    email: bool = False,
    alert_type: str = "price",
    is_armed: bool = True,
    active: bool = True,
) -> TelegramAlert:
    ticker = ticker or rng.choice(["AAPL", "NVDA", "TSLA", "MRNA", "SMCI"])
    condition = condition or rng.choice(["above", "below"])
    timeframe = timeframe or rng.choice(["1h", "4h", "1d"])
    price = price if price is not None else round(rng.randfloat(10, 400), 2)

    # Build new-schema JSON
    rule = {"price_above": price} if condition == "above" else {"price_below": price}
    cfg = {"ticker": ticker, "timeframe": timeframe, "rule": rule}
    row = TelegramAlert(
        user_id=user_id,
        status="ARMED",
        email=email,
        created_at=utcnow(),
        config_json=json.dumps(cfg, separators=(",", ":")),
        re_arm_config=None,
        trigger_count=0,
        last_trigger_condition=None,
        last_triggered_at=None,
    )

    s.add(row); s.flush()
    return row

def make_schedule(s: Session, rng: RNG, *, user_id: int,
                  scheduled_time: str | None = None, interval: str | None = None,
                  active: bool = True) -> TelegramSchedule:
    row = TelegramSchedule(
        ticker=rng.choice(["AAPL", "NVDA", "TSLA"]),
        user_id=user_id,
        #scheduled_time=scheduled_time or "09:00",
        period="daily",
        active=active,
        email=False,
        indicators="rsi,bb,atr",
        interval=interval or "1h",
        provider="alpaca",
        created=utcnow(),
        schedule_type="screener",
        list_type="watchlist",
        config_json=None,
        schedule_config="utc"
    )
    s.add(row); s.flush()
    return row

def make_feedback(s: Session, *, user_id: int, type_: str = "bug",
                  message: str = "it broke", status: str | None = None) -> TelegramFeedback:
    row = TelegramFeedback(user_id=user_id, type=type_, message=message, created=utcnow(), status=status)
    s.add(row); s.flush()
    return row

def log_command(s: Session, *, telegram_user_id: str, command: str,
                success: bool = True, **kw) -> TelegramCommandAudit:
    row = TelegramCommandAudit(
        telegram_user_id=str(telegram_user_id), command=command,
        success=success, created=utcnow(), **kw
    )
    s.add(row); s.flush()
    return row

def set_setting(s: Session, *, key: str, value: str | None) -> TelegramSetting:
    row = s.get(TelegramSetting, key)
    if row:
        row.value = value; s.flush(); return row
    row = TelegramSetting(key=key, value=value)
    s.add(row); s.flush()
    return row

def issue_code(s: Session, *, user_id: int, code: str, sent_time: int) -> TelegramVerificationCode:
    row = TelegramVerificationCode(user_id=user_id, code=code, sent_time=sent_time)
    s.add(row); s.flush()
    return row


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "-rA"]))