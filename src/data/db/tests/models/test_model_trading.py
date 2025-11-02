from sqlalchemy.exc import IntegrityError

from src.data.db.models.model_trading import BotInstance, Trade, Position
from src.data.db.models.model_users import User


def test_botinstance_requires_config(db_session):
    # create a user to satisfy FK
    u = User()
    u.email = "bot_owner@example.com"
    db_session.add(u)
    db_session.flush()

    bot = BotInstance()
    bot.user_id = u.id
    bot.type = "paper"
    bot.status = "running"
    # missing config -> event listener should raise IntegrityError on flush/commit

    db_session.add(bot)
    try:
        db_session.flush()
        raised = False
    except IntegrityError:
        db_session.rollback()
        raised = True

    assert raised is True


def test_trade_position_relationships(db_session):
    u = User()
    u.email = "trade_owner@example.com"
    db_session.add(u)
    db_session.flush()

    bot = BotInstance()
    bot.user_id = u.id
    bot.type = "paper"
    bot.status = "running"
    bot.config = {"strategy": "x"}
    db_session.add(bot)
    db_session.flush()

    pos = Position()
    pos.bot_id = bot.id
    pos.trade_type = "paper"
    pos.symbol = "ABC"
    pos.direction = "long"
    pos.qty_open = 1
    pos.status = "open"
    db_session.add(pos)
    db_session.flush()

    t = Trade()
    t.bot_id = bot.id
    t.trade_type = "paper"
    t.entry_logic_name = "e"
    t.exit_logic_name = "x"
    t.symbol = "ABC"
    t.interval = "1m"
    t.direction = "long"
    t.status = "open"
    t.position_id = pos.id
    db_session.add(t)
    db_session.flush()

    assert t.id is not None
    assert t.position is not None
