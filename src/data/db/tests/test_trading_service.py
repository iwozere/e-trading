from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import time
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

# services under test
from src.data.db.services import database_service as ds
from src.data.db.services.trading_service import (
    upsert_bot, heartbeat,
    add_trade, close_trade, get_open_trades, get_pnl_summary,
    ensure_open_position, apply_fill, close_if_flat, mark_closed,
    get_open_positions, add_metric, latest_metrics,
)

# models for optional direct assertions
from src.data.db.models.model_trading import Base as TradingBase, BotInstance


@pytest.fixture()
def setup_inmemory_db(monkeypatch):
    # Create in-memory engine with FKs ON
    eng = create_engine("sqlite+pysqlite:///:memory:", future=True)
    @event.listens_for(eng, "connect")
    def _fk_on(dbapi_con, _):
        cur = dbapi_con.cursor()
        cur.execute("PRAGMA foreign_keys=ON;")
        cur.close()
    SessionLocal = sessionmaker(bind=eng, autoflush=False, autocommit=False, future=True)

    # Monkeypatch database_service to use this engine/sessionmaker
    monkeypatch.setattr(ds, "engine", eng, raising=False)
    monkeypatch.setattr(ds, "SessionLocal", SessionLocal, raising=False)
    ds._db_service_singleton = None  # reset singleton

    # Create tables via DatabaseService.init_databases()
    svc = ds.get_database_service()
    # only trading base required for these tests
    TradingBase.metadata.create_all(eng)

    yield eng, SessionLocal


def test_bot_and_trades_flow(setup_inmemory_db):
    eng, SessionLocal = setup_inmemory_db

    # Upsert bot and heartbeat
    b = upsert_bot({"id": "bot1", "type": "paper", "status": "running"})
    assert b["id"] == "bot1" and b["status"] == "running"

    upsert_bot({"id": "bot1", "type": "paper", "status": "stopped"})
    heartbeat("bot1")

    # Optional direct assertion via session
    s = SessionLocal()
    try:
        row = s.get(BotInstance, "bot1")
        assert row and row.status == "stopped"
        assert row.last_heartbeat is not None
        assert row.last_heartbeat.timestamp() <= time.time() + 1
    finally:
        s.close()

    # Trades
    t = add_trade({
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
    assert t["id"] == "t1"
    assert any(tr["id"] == "t1" for tr in get_open_trades())

    assert close_trade("t1", status="closed", net_pnl=Decimal("12.5")) is True
    assert not any(tr["id"] == "t1" for tr in get_open_trades())

    # PnL summary after adding another closed trade
    add_trade({
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
    agg = get_pnl_summary("bot1")
    assert agg == {"net_pnl": 10.0, "n_trades": 2}


def test_positions_and_metrics_flow(setup_inmemory_db):
    _, _ = setup_inmemory_db
    UTC = timezone.utc

    # Ensure FK parent exists in this fresh DB
    upsert_bot({"id": "bot1", "type": "paper", "status": "running"})

    # Positions
    p = ensure_open_position(bot_id="bot1", trade_type="paper", symbol="MSFT", direction="long")
    # Buy 10 @ 100 => qty=10, avg=100
    p = apply_fill(p["id"], action="buy", qty=10, price=100.0)
    assert p["qty_open"] == 10.0 and p["avg_price"] == 100.0
    # Sell 10 @ 110 => realize +100, close
    p = apply_fill(p["id"], action="sell", qty=10, price=110.0, ts=datetime.now(UTC))
    assert p["status"] == "closed" and round(p["realized_pnl"], 6) == 100.0

    # Reopen and explicitly close/mark
    p2 = ensure_open_position(bot_id="bot1", trade_type="paper", symbol="NVDA", direction="short")
    p2 = close_if_flat(p2["id"])  # still open (qty 0? ensure_open_position starts with 0)
    # mark closed anyway
    p2 = mark_closed(p2["id"])
    assert p2["status"] == "closed"

    open_pos = get_open_positions(bot_id="bot1")
    # Both MSFT and NVDA positions should be closed now
    assert all(x["status"] == "closed" for x in open_pos) or open_pos == []

    # Metrics
    m1 = add_metric({"id": "m1", "bot_id": "bot1", "trade_type": "paper", "metrics": {}, "calculated_at": datetime(2024, 1, 1, 12, 0, 0)})
    m2 = add_metric({"id": "m2", "bot_id": "bot1", "trade_type": "paper", "metrics": {}, "calculated_at": datetime(2024, 1, 2, 12, 0, 0)})
    latest = latest_metrics("bot1", limit=1)
    assert len(latest) == 1 and latest[0]["id"] == "m2"


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "-rA"]))