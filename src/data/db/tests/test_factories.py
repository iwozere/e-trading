from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

import json
from datetime import datetime, timezone
from src.data.db.tests.factories import RNG, make_user, add_telegram_identity, make_position

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def test_make_user_and_identity(dbsess):
    u = make_user(dbsess, email="a@b.com")
    add_telegram_identity(dbsess, u.id, "12345", language="en")
    dbsess.commit()
    assert u.id is not None

def test_make_position(dbsess):
    rng = RNG(1)
    u = make_user(dbsess, email="x@y.com")
    p = make_position(dbsess, rng, bot_id="bot-1", symbol="AAPL", direction="long")
    dbsess.commit()
    assert p.symbol == "AAPL"

if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q", "-rA"]))