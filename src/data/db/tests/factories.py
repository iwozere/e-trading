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
from src.data.db.models.model_users import User, AuthIdentity, VerificationCode  # your files
from src.data.db.models.model_telegram import (
    TelegramFeedback, TelegramCommandAudit, TelegramBroadcastLog, TelegramSetting
)
from src.data.db.models.model_trading import Position  # (above) or your own path
from src.data.db.models.model_trading import BotInstance
from src.data.db.models.model_jobs import Schedule, ScheduleRun, JobType, RunStatus

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
        identity_metadata=meta or None,  # <-- correct attribute name for metadata column
        created_at=utcnow()  # Ensure created_at is set
    )
    s.add(ai); s.flush()
    return ai

def make_verification_code(s: Session, *, user_id: int, code: str | None = None,
                          provider: str = "telegram", sent_time: int | None = None) -> VerificationCode:
    """Create a verification code with all required columns."""
    import time

    code = code or f"CODE{random.randint(100000, 999999)}"
    sent_time = sent_time or int(time.time())

    vc = VerificationCode(
        user_id=user_id,
        code=code,
        sent_time=sent_time,
        provider=provider,  # New column with default 'telegram'
        created_at=utcnow()  # New column with default now()
    )
    s.add(vc); s.flush()
    return vc

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

def make_feedback(s: Session, *, user_id: int, type_: str = "bug",
                  message: str = "it broke", status: str | None = None,
                  created_at: datetime | None = None) -> TelegramFeedback:
    # Use all the corrected TelegramFeedback columns
    created_at = created_at or utcnow()
    status = status or "open"

    row = TelegramFeedback(
        user_id=user_id,
        type=type_,  # Now included in model
        message=message,  # Now included in model
        created_at=created_at,  # Now included in model (renamed from 'created')
        status=status  # Now included in model
    )
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

# Removed issue_code function as TelegramVerificationCode doesn't exist

# -------------------------------- JOBS ------------------------------------

def make_job_schedule(s: Session, rng: RNG, *,
                      user_id: int,
                      name: str | None = None,
                      job_type: str | None = None,
                      target: str | None = None,
                      cron: str | None = None,
                      enabled: bool = True) -> Schedule:
    name = name or f"Schedule-{rng.randint(100, 999)}"
    job_type = job_type or rng.choice(["report", "screener", "alert"])
    target = target or rng.choice(["portfolio", "tech_stocks", "price_alert"])
    cron = cron or rng.choice(["0 9 * * *", "*/5 * * * *", "0 0 1 * *"])

    schedule = Schedule(
        user_id=user_id,
        name=name,
        job_type=job_type,
        target=target,
        task_params={"created_by": "factory"},
        cron=cron,
        enabled=enabled,
        next_run_at=datetime.now(UTC) + timedelta(hours=1)
    )
    s.add(schedule); s.flush()
    return schedule

def make_run(s: Session, rng: RNG, *,
             user_id: int,
             job_type: str | None = None,
             job_id: int | None = None,
             status: str | None = None,
             scheduled_for: datetime | None = None,
             worker_id: str | None = None) -> Run:
    job_type = job_type or rng.choice(["report", "screener", "alert"])
    # Changed to BigInteger - use larger range for PostgreSQL int8
    job_id = job_id if job_id is not None else rng.randint(100000, 999999999)
    status = status or "pending"
    scheduled_for = scheduled_for or datetime.now(UTC)
    worker_id = worker_id or f"worker-{rng.randint(1, 100)}"

    # Import the correct Run model (not ScheduleRun)
    from src.data.db.models.model_jobs import Run

    run = Run(
        job_type=job_type,
        job_id=job_id,  # Now BigInteger
        user_id=user_id,  # Already BigInteger in model
        status=status,
        scheduled_for=scheduled_for,
        worker_id=worker_id,  # Add missing worker_id field
        job_snapshot={"created_by": "factory", "test": True}
    )
    s.add(run); s.flush()
    return run


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "-rA"]))