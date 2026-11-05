"""legacy_schema_fixes

Revision ID: 015_legacy_schema_fixes
Revises: 014_trade_ledger
Create Date: 2026-11-05

Brings the pre-Alembic tables in line with the ORM models. Everything else that
differed (indexes, defaults, comments, constraint names) was fixed in the models,
which now describe prod exactly; this migration only covers the cases where the
model is right and the database is wrong:

- msg_system_health: drop two leftovers from its old name msg_channel_health:
  UNIQUE (component) -- blocks two systems sharing a component name; the real key
  is the unique index on (system, COALESCE(component, '')) -- and a status CHECK
  without 'UNKNOWN', which rejects the repo's default status. Widen component to
  VARCHAR(100) as the model declares (v_channel_health_summary depends on the
  column, so it is dropped and recreated with its current definition and comment).
- usr_users: is_active NOT NULL (NULLs become true, the column default), the
  role CHECK and UNIQUE (email).
- usr_auth_identities: UNIQUE (provider, external_id), so one Telegram account
  cannot map to two identities.
- webui_system_config: UNIQUE (key).

New constraints are validated first: if existing rows violate one, the upgrade
raises with the offending values and, since Postgres DDL is transactional,
nothing is applied.
"""

import sqlalchemy as sa
from alembic import op

revision = "015_legacy_schema_fixes"
down_revision = "014_trade_ledger"
branch_labels = None
depends_on = None

_VIEW = "v_channel_health_summary"

# (description, query returning offending rows)
_PRECHECKS = [
    ("duplicate usr_users.email", "SELECT email, count(*) FROM usr_users WHERE email IS NOT NULL GROUP BY email HAVING count(*) > 1"),
    (
        "duplicate usr_auth_identities (provider, external_id)",
        "SELECT provider, external_id, count(*) FROM usr_auth_identities GROUP BY provider, external_id HAVING count(*) > 1",
    ),
    ("duplicate webui_system_config.key", 'SELECT "key", count(*) FROM webui_system_config GROUP BY "key" HAVING count(*) > 1'),
    ("usr_users.role outside admin/trader/viewer", "SELECT id, role FROM usr_users WHERE role NOT IN ('admin', 'trader', 'viewer')"),
    ("msg_system_health.component longer than 100", "SELECT id, length(component) FROM msg_system_health WHERE length(component) > 100"),
]


def _precheck(conn: sa.engine.Connection) -> None:
    problems = []
    for what, query in _PRECHECKS:
        rows = conn.execute(sa.text(query)).fetchall()
        if rows:
            problems.append(f"{what}: {[tuple(r) for r in rows[:10]]}")
    if problems:
        raise RuntimeError("015_legacy_schema_fixes: fix these rows first, nothing was changed:\n  " + "\n  ".join(problems))


def _save_view(conn: sa.engine.Connection) -> tuple[str, str | None] | None:
    """Return (definition, comment) of the view, or None when it does not exist."""
    row = conn.execute(
        sa.text(
            "SELECT pg_get_viewdef(c.oid, true), obj_description(c.oid, 'pg_class') FROM pg_class c "
            "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'public' AND c.relname = :v AND c.relkind = 'v'"
        ),
        {"v": _VIEW},
    ).first()
    return (row[0], row[1]) if row else None


def _restore_view(saved: tuple[str, str | None] | None) -> None:
    if saved is None:
        return
    definition, comment = saved
    op.execute(f"CREATE VIEW {_VIEW} AS {definition}")
    if comment:
        op.execute(sa.text(f"COMMENT ON VIEW {_VIEW} IS :c").bindparams(c=comment))


def _set_component_width(length: int) -> None:
    conn = op.get_bind()
    saved = _save_view(conn)
    if saved is not None:
        op.execute(f"DROP VIEW {_VIEW}")
    op.alter_column("msg_system_health", "component", type_=sa.String(length), existing_nullable=True)
    _restore_view(saved)


def upgrade() -> None:
    conn = op.get_bind()
    _precheck(conn)

    # msg_system_health
    op.execute("ALTER TABLE msg_system_health DROP CONSTRAINT IF EXISTS msg_channel_health_channel_key")
    op.execute("ALTER TABLE msg_system_health DROP CONSTRAINT IF EXISTS msg_channel_health_status_check")
    _set_component_width(100)

    # usr_users
    op.execute("UPDATE usr_users SET is_active = true WHERE is_active IS NULL")
    op.alter_column("usr_users", "is_active", existing_type=sa.Boolean(), nullable=False, existing_server_default=sa.text("true"))
    # op.f(): the name is final; otherwise the "ck" naming convention prefixes it a second time
    op.create_check_constraint(op.f("ck_usr_users_ck_users_role"), "usr_users", "role IN ('admin','trader','viewer')")
    op.create_unique_constraint("uq_usr_users_email", "usr_users", ["email"])

    op.create_unique_constraint("uq_auth_identities_provider_external", "usr_auth_identities", ["provider", "external_id"])
    op.create_unique_constraint("uq_webui_system_config_key", "webui_system_config", ["key"])


def downgrade() -> None:
    op.drop_constraint("uq_webui_system_config_key", "webui_system_config", type_="unique")
    op.drop_constraint("uq_auth_identities_provider_external", "usr_auth_identities", type_="unique")
    op.drop_constraint("uq_usr_users_email", "usr_users", type_="unique")
    op.drop_constraint(op.f("ck_usr_users_ck_users_role"), "usr_users", type_="check")
    op.alter_column("usr_users", "is_active", existing_type=sa.Boolean(), nullable=True, existing_server_default=sa.text("true"))

    # Restoring the old width fails if longer components were written since the upgrade.
    _set_component_width(50)
    op.execute(
        "ALTER TABLE msg_system_health ADD CONSTRAINT msg_channel_health_status_check "
        "CHECK (status IN ('HEALTHY', 'DEGRADED', 'DOWN'))"
    )
    op.execute("ALTER TABLE msg_system_health ADD CONSTRAINT msg_channel_health_channel_key UNIQUE (component)")
