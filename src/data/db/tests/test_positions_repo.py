
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4] if __file__ else Path.cwd()
sys.path.append(str(PROJECT_ROOT))

from decimal import Decimal
from datetime import datetime, timezone
from src.data.db.models.model_trading import Base as PosBase, Position, BotInstance
from src.data.db.repos.repo_trading import PositionsRepo
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

UTC = timezone.utc

def _ensure_bot(s, bot_id="bot1"):
    if not s.get(BotInstance, bot_id):
        s.add(BotInstance(id=bot_id, type="paper", status="running"))
        s.flush()

def test_open_add_reduce_close():
    eng = create_engine("sqlite+pysqlite:///:memory:", future=True)
    PosBase.metadata.create_all(eng)
    Session = sessionmaker(bind=eng, future=True)

    s = Session()
    try:
        _ensure_bot(s, "bot1")
        repo = PositionsRepo(s)

        # 1) Open a long pos (bound to this session)
        p = repo.ensure_open(bot_id="bot1", trade_type="paper", symbol="AAPL", direction="long")

        # 2) Buy 10 @ 100 → qty=10, avg=100
        p = repo.apply_fill(position_id=p.id, action="buy", qty=10, price=100)
        assert Decimal(p.qty_open) == Decimal("10")
        assert Decimal(p.avg_price) == Decimal("100")

        # 3) Buy 10 @ 110 → qty=20, avg=105
        p = repo.apply_fill(position_id=p.id, action="buy", qty=10, price=110)
        assert Decimal(p.qty_open) == Decimal("20")
        assert Decimal(p.avg_price).quantize(Decimal("0.00000001")) == Decimal("105")

        # 4) Sell 5 @ 120 → realize (120-105)*5 = 75 ; qty=15
        p = repo.apply_fill(position_id=p.id, action="sell", qty=5, price=120)
        assert Decimal(p.realized_pnl) == Decimal("75")
        assert Decimal(p.qty_open) == Decimal("15")

        # 5) Sell 15 @ 90 → realize (90-105)*15 = -225 ; total = -150 ; close
        p = repo.apply_fill(position_id=p.id, action="sell", qty=15, price=90)
        assert Decimal(p.realized_pnl) == Decimal("-150")
        assert p.status == "closed"
        assert p.closed_at is not None

        s.commit()
    finally:
        s.close()

def test_short_math():
    eng = create_engine("sqlite+pysqlite:///:memory:", future=True)
    PosBase.metadata.create_all(eng)
    Session = sessionmaker(bind=eng, future=True)

    s = Session()
    try:
        _ensure_bot(s, "bot1")
        repo = PositionsRepo(s)
        p = repo.ensure_open(bot_id="bot1", trade_type="paper", symbol="TSLA", direction="short")

        # 1) Sell 5 @ 200 → qty=5, avg=200
        p = repo.apply_fill(position_id=p.id, action="sell", qty=5, price=200)
        # 2) Buy 2 @ 180 → realize (200-180)*2 = 40 ; qty=3
        p = repo.apply_fill(position_id=p.id, action="buy", qty=2, price=180)
        # 3) Buy 3 @ 220 → realize (200-220)*3 = -60 ; total = -20 ; close
        p = repo.apply_fill(position_id=p.id, action="buy", qty=3, price=220)
        assert p.status == "closed"

        s.commit()
    finally:
        s.close()

if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "-rA"]))
