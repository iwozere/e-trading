Nice — you’re baselined. From here, Alembic becomes your one source of truth for schema changes. Here’s a tight playbook + a couple of ready-made templates you can copy/paste when you make your first “real” migration.

# What to do next (day-to-day workflow)

1. **Change your models** (add/remove columns, FKs, indexes, constraints).
2. Generate a migration script:

   ```bash
   alembic revision --autogenerate -m "describe the change"
   ```
3. **Open the generated file** in `.../migrations/versions/` and review the `upgrade()` / `downgrade()` (especially on SQLite).
4. Apply it:

   ```bash
   alembic upgrade head
   ```
5. Verify:

   ```bash
   alembic current -v
   ```

> Tip (SQLite): we already enabled `render_as_batch=True` in `env.py`, so Alembic can emulate `ALTER TABLE` safely.

---

# Keep this in mind

* **Don’t call `Base.metadata.create_all()` at runtime** for your app DB anymore. Use it only in unit tests (e.g., for in-memory SQLite), or switch tests to `alembic upgrade head` (fixture example below).
* **Every schema change goes through Alembic**. If you tweak the DB by hand, Alembic won’t know.
* **Name constraints** in the models (you already did) so autogenerate doesn’t choke.

---

# Quick wins you can ship as first real migrations

These are common/handy:

### A) Add a missing index (example: `trades.position_id`)

1. Add to model:

   ```python
   from sqlalchemy import Index
   # in Trade.__table_args__:
   Index("ix_trades_position_id", "position_id"),
   ```
2. Generate & upgrade:

   ```bash
   alembic revision --autogenerate -m "trades: index position_id"
   alembic upgrade head
   ```

### B) Add an `updated_at` column to `positions` (if you want to track writes)

1. Model change:

   ```python
   from sqlalchemy import DateTime
   updated_at: Mapped[str | None] = mapped_column(DateTime)
   ```
2. Autogenerate & upgrade (SQLite will rebuild the table for this — expected).

### C) Add a foreign key later

If you add an FK (e.g., `trades.position_id → positions.id`) to the model, Alembic will produce a batch migration (table copy). Review it carefully, then upgrade.

---

# Data migration template (copy/paste)

Need to move/clean data during an upgrade? Use SQLAlchemy Core in the migration:

```python
# migrations/versions/XXXXXXXX_data_fix.py
from alembic import op
import sqlalchemy as sa

revision = "XXXXXXXX"
down_revision = "YYYYYYYY"
branch_labels = None
depends_on = None

def upgrade():
    conn = op.get_bind()

    # Example: backfill trades.position_id from extra_metadata JSON (SQLite)
    # conn.execute(sa.text("""
    #   UPDATE trades
    #   SET position_id = json_extract(extra_metadata, '$.position_id')
    #   WHERE position_id IS NULL AND extra_metadata IS NOT NULL
    # """))

def downgrade():
    pass  # usually leave data-only downgrades as no-op
```

---

# Test fixtures (run migrations for test DB)

If you want tests to reflect the real schema, upgrade an in-memory DB to `head`:

```python
# tests/conftest.py
import pytest
from sqlalchemy.orm import sessionmaker
from alembic import command
from alembic.config import Config
from src.data.db.core.database import make_sqlite_memory_engine  # add this helper if you haven’t

@pytest.fixture(scope="session")
def engine():
    eng = make_sqlite_memory_engine()
    # run Alembic migrations programmatically
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    return eng

@pytest.fixture()
def dbsess(engine):
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    s = SessionLocal()
    try:
        yield s
        s.rollback()
    finally:
        s.close()
```

> If you don’t have `make_sqlite_memory_engine()` yet, add it to `database.py` (StaticPool, FK=ON, WAL not needed for memory).

---

# Helpful commands (cheat sheet)

* Show current revision: `alembic current -v`
* See history: `alembic history --verbose`
* Upgrade stepwise: `alembic upgrade +1` / Downgrade: `alembic downgrade -1`
* Merge branches (if teammates created parallel heads): `alembic merge -m "merge heads" <revA> <revB>`

---
