from __future__ import annotations
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy.orm import Session

from src.data.db.repos.repo_trading import BotsRepo, MetricsRepo, PositionsRepo
from src.data.db.repos.repo_users import UsersRepo


UTC = timezone.utc


def test_bots_metrics_positions(db_session: Session):
    bots = BotsRepo(db_session)
    metrics = MetricsRepo(db_session)
    positions = PositionsRepo(db_session)
    users = UsersRepo(db_session)

    # create a user for FK
    user = users.ensure_user_for_telegram("bot-user-1", defaults_user={"email": "botuser@example.com"})
    assert user.id > 0

    # upsert bot (integer PK, include required fields and config)
    b = bots.upsert_bot({"id": 1, "user_id": user.id, "type": "paper", "status": "stopped", "config": {}})
    assert b.id == 1

    # heartbeat doesn't raise
    bots.heartbeat(1)

    # metrics add / latest
    m1 = metrics.add({"bot_id": 1, "trade_type": "paper", "calculated_at": datetime.now(UTC), "metrics": {"pnl": {"net": 0}}})
    m2 = metrics.add({"bot_id": 1, "trade_type": "paper", "calculated_at": datetime.now(UTC), "metrics": {"pnl": {"net": 1}}})
    latest = metrics.latest_for_bot(1, limit=1)
    assert len(latest) == 1
    assert latest[0].id in {m1.id, m2.id}

    # positions ensure + apply fills
    p = positions.ensure_open(bot_id=1, trade_type="live", symbol="BTCUSDT", direction="long")
    assert p.status == "open"
    # add
    p = positions.apply_fill(position_id=p.id, action="buy", qty=1, price=100)
    assert p.qty_open == Decimal("1")
    assert p.avg_price == Decimal("100")
    # add again at higher price -> avg moves
    p = positions.apply_fill(position_id=p.id, action="buy", qty=1, price=110)
    assert p.qty_open == Decimal("2")
    assert p.avg_price == Decimal("105")
    # reduce some
    p = positions.apply_fill(position_id=p.id, action="sell", qty=1, price=120)
    assert p.qty_open == Decimal("1")
    assert p.realized_pnl == Decimal("15")
    # flat and close
    p = positions.apply_fill(position_id=p.id, action="sell", qty=1, price=105)
    assert p.status == "closed"
