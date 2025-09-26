from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import time
from decimal import Decimal
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from src.data.db.models.model_trading import Base as TradingBase, BotInstance
from src.data.db.repos.repo_trading import BotsRepo, TradesRepo, MetricsRepo, PositionsRepo, _ensure_bot


# ---------- in-memory DB (FKs ON) + tables just for trading ----------
@pytest.fixture()
def engine():
    eng = create_engine("sqlite+pysqlite:///:memory:", future=True)
    @event.listens_for(eng, "connect")
    def _fk_on(dbapi_con, _):
        cur = dbapi_con.cursor()
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()
    TradingBase.metadata.create_all(eng)
    return eng

@pytest.fixture()
def dbsess(engine):
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    s = Session()
    try:
        yield s
        s.rollback()
    finally:
        s.close()


# ------------------------------- BotRepo --------------------------------
def test_bot_upsert_and_heartbeat(dbsess):
    bots = BotsRepo(dbsess)

    # insert
    bots.upsert_bot({"id": "bot1", "type": "paper", "status": "running"})
    obj = dbsess.get(BotInstance, "bot1")
    assert obj and obj.status == "running"

    # update path
    bots.upsert_bot({"id": "bot1", "type": "paper", "status": "stopped"})
    obj = dbsess.get(BotInstance, "bot1")
    assert obj.status == "stopped"

    # heartbeat
    before = time.time()
    bots.heartbeat("bot1")
    obj = dbsess.get(BotInstance, "bot1")
    assert obj.last_heartbeat is not None
    assert obj.last_heartbeat.timestamp() >= before - 1


# ------------------------------ TradesRepo ------------------------------
def test_trades_add_close_summary_and_open_query(dbsess):
    _ensure_bot(dbsess, "bot1")
    trades = TradesRepo(dbsess)

    # open trade
    t1 = trades.add({
        "id": "t1",
        "bot_id": "bot1",
        "trade_type": "paper",
        "entry_logic_name": "E",
        "exit_logic_name": "X",
        "symbol": "AAPL",
        "interval": "1h",
        "direction": "long",
        "status": "open",
    })
    # open_trades shows it
    assert any(x.id == "t1" for x in trades.open_trades())

    # close it with PnL
    trades.close_trade("t1", status="closed", net_pnl=Decimal("12.5"))
    assert not any(x.id == "t1" for x in trades.open_trades())

    # another closed trade with negative PnL
    trades.add({
        "id": "t2",
        "bot_id": "bot1",
        "trade_type": "paper",
        "entry_logic_name": "E",
        "exit_logic_name": "X",
        "symbol": "AAPL",
        "interval": "1h",
        "direction": "long",
        "status": "closed",
        "net_pnl": Decimal("-2.5"),
    })

    agg = trades.pnl_summary("bot1")
    # net: 12.5 - 2.5 = 10, n_trades = 2 (only closed count)
    assert agg.net_pnl == Decimal("10.0")
    assert agg.n_trades == 2


def test_trades_check_constraint_violation(dbsess):
    trades = TradesRepo(dbsess)
    # invalid direction should violate CHECK constraint
    with pytest.raises(IntegrityError):
        trades.add({
            "id": "bad1",
            "bot_id": "bot1",
            "trade_type": "paper",
            "entry_logic_name": "E",
            "exit_logic_name": "X",
            "symbol": "AAPL",
            "interval": "1h",
            "direction": "sideways",  # invalid
            "status": "open",
        })


# ------------------------------ MetricsRepo -----------------------------
def test_metrics_latest_for_bot(dbsess):
    _ensure_bot(dbsess, "bot1")
    metrics = MetricsRepo(dbsess)
    dt1 = datetime(2024, 1, 1, 12, 0, 0)
    dt2 = datetime(2024, 1, 2, 12, 0, 0)

    metrics.add({"id": "m1", "bot_id": "bot1", "trade_type": "paper", "metrics": {}, "calculated_at": dt1})
    metrics.add({"id": "m2", "bot_id": "bot1", "trade_type": "paper", "metrics": {}, "calculated_at": dt2})
    latest = metrics.latest_for_bot("bot1", limit=1)
    assert len(latest) == 1 and latest[0].id == "m2"


# ----------------------------- PositionsRepo ----------------------------
def test_positions_integration_and_math(dbsess):
    _ensure_bot(dbsess, "bot1")
    positions = PositionsRepo(dbsess)
    UTC = timezone.utc

    p = positions.ensure_open(bot_id="bot1", trade_type="paper", symbol="MSFT", direction="long")
    # Buy 10 @ 100 => qty=10, avg=100
    positions.apply_fill(position_id=p.id, action="buy", qty=10, price=100.0)
    # Sell 10 @ 110 => realize +100, close
    positions.apply_fill(position_id=p.id, action="sell", qty=10, price=110.0, ts=datetime.now(UTC))
    assert p.status == "closed"
    assert Decimal(p.realized_pnl) == Decimal("100")


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "-rA"]))