#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump ONLY reconstructed DDL (from PRAGMAs) to Markdown.

- Tables: CREATE TABLE with columns, NOT NULL, DEFAULT, PRIMARY KEY (inline or composite),
          FOREIGN KEY constraints (multi-column).
- Indexes: CREATE [UNIQUE] INDEX with column order (ASC/DESC) when available.
- Views/Triggers are omitted (no reliable PRAGMA-based reconstruction).

Usage:
  python extract_schema_as_md.py                       # uses db/trading.db, stdout
  python extract_schema_as_md.py --out schema.md
  python extract_schema_as_md.py --db path/to/db.sqlite --out schema.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sqlite3
from collections import defaultdict
from typing import Dict, List


# ---------- SQLite helpers ----------

def mk_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # ensure foreign_keys pragma is on (not required for schema read, but sane)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
    except Exception:
        pass
    return conn


def ident(name: str) -> str:
    """SQLite identifier quoting with double quotes."""
    if name is None:
        return '""'
    return '"' + str(name).replace('"', '""') + '"'


def get_tables(conn: sqlite3.Connection) -> List[str]:
    sql = """
      SELECT name
      FROM sqlite_master
      WHERE type = 'table'
        AND name NOT LIKE 'sqlite_%'
      ORDER BY name
    """
    return [r["name"] for r in conn.execute(sql)]


def pragma_table_info(conn: sqlite3.Connection, table: str) -> List[sqlite3.Row]:
    return list(conn.execute(f"PRAGMA table_info({ident(table)})"))


def pragma_foreign_keys(conn: sqlite3.Connection, table: str) -> List[sqlite3.Row]:
    return list(conn.execute(f"PRAGMA foreign_key_list({ident(table)})"))


def pragma_index_list(conn: sqlite3.Connection, table: str) -> List[sqlite3.Row]:
    return list(conn.execute(f"PRAGMA index_list({ident(table)})"))


def pragma_index_xinfo(conn: sqlite3.Connection, index: str) -> List[sqlite3.Row]:
    # xinfo includes expressions and sort order (desc), key flag
    return list(conn.execute(f"PRAGMA index_xinfo({ident(index)})"))


# ---------- Reconstruction ----------

def reconstruct_create_table(conn: sqlite3.Connection, table: str) -> str:
    cols = pragma_table_info(conn, table)
    if not cols:
        # virtual table or unexpected
        return f"-- Could not reconstruct table {ident(table)} (no PRAGMA table_info rows)."

    # Determine PK columns (order by pk sequence)
    pk_cols = [(c["name"], c["pk"]) for c in cols if int(c["pk"] or 0) > 0]
    pk_cols.sort(key=lambda x: x[1])

    # Build column lines
    col_lines: List[str] = []
    inline_pk_allowed = (len(pk_cols) == 1)
    pk_name_inline = pk_cols[0][0] if inline_pk_allowed else None

    for c in sorted(cols, key=lambda r: r["cid"]):
        parts: List[str] = [ident(c["name"])]
        coltype = (c["type"] or "").strip()
        if coltype:
            parts.append(coltype)

        # Inline PK only if single-column PK
        if inline_pk_allowed and c["name"] == pk_name_inline:
            parts.append("PRIMARY KEY")

        if int(c["notnull"] or 0) == 1:
            parts.append("NOT NULL")

        dflt = c["dflt_value"]
        if dflt is not None:
            # PRAGMA returns textual default (already quoted if string)
            parts.append(f"DEFAULT {dflt}")

        col_lines.append("  " + " ".join(parts))

    # Table-level constraints
    constraint_lines: List[str] = []

    # Composite PK
    if not inline_pk_allowed and pk_cols:
        pk_names = [ident(name) for name, _ in pk_cols]
        constraint_lines.append(f"  PRIMARY KEY ({', '.join(pk_names)})")

    # Foreign keys (group by id; each id can be multi-column)
    fks = pragma_foreign_keys(conn, table)
    if fks:
        grouped: Dict[int, List[sqlite3.Row]] = defaultdict(list)
        for fk in fks:
            grouped[int(fk["id"])].append(fk)
        for _id, rows in sorted(grouped.items(), key=lambda kv: kv[0]):
            rows.sort(key=lambda r: r["seq"])
            ref_table = rows[0]["table"]
            from_cols = [ident(r["from"]) for r in rows]
            to_cols = [ident(r["to"]) for r in rows]
            on_upd = rows[0]["on_update"]
            on_del = rows[0]["on_delete"]
            match = rows[0]["match"]

            clause = f"  FOREIGN KEY ({', '.join(from_cols)}) REFERENCES {ident(ref_table)} ({', '.join(to_cols)})"
            if on_upd and on_upd.upper() != "NO ACTION":
                clause += f" ON UPDATE {on_upd}"
            if on_del and on_del.upper() != "NO ACTION":
                clause += f" ON DELETE {on_del}"
            if match and match.upper() != "NONE":
                clause += f" MATCH {match}"
            constraint_lines.append(clause)

    all_lines = col_lines + (["  ,"] if (col_lines and constraint_lines) else [])  # pretty comma separator
    # Actually commas must be between every definition; rebuild with commas properly:
    defs = col_lines + constraint_lines
    ddl = f"CREATE TABLE {ident(table)} (\n" + ",\n".join(defs) + "\n);"
    return ddl


def reconstruct_indexes(conn: sqlite3.Connection, table: str) -> List[str]:
    idxs = pragma_index_list(conn, table)
    ddls: List[str] = []
    for idx in idxs:
        name = idx["name"]
        # Skip implicit PK index (origin='pk') and 'autoindex' generated by constraints
        origin = idx["origin"]
        if origin in ("pk", "u"):  # keep UNIQUE? We'll reconstruct explicit unique indexes too
            # origin 'u' can still be a user-created unique index; keep it.
            pass
        # Some auto indexes come with 'sqlite_autoindex_%' names; skip them
        if name.startswith("sqlite_autoindex_"):
            continue

        # Gather columns/expressions from xinfo
        xinfo = pragma_index_xinfo(conn, name)
        # retain only key columns (xinfo.key == 1)
        key_rows = [r for r in xinfo if ("key" in r.keys() and int(r["key"]) == 1)]
        key_rows.sort(key=lambda r: (r["seqno"] if r["seqno"] is not None else 0))

        parts: List[str] = []
        incomplete_expr = False
        for r in key_rows:
            cid = r["cid"]
            nm = r["name"]
            desc = r["desc"]
            order = " DESC" if (desc == 1) else ""
            if cid == -2:  # expression-based (SQLite doesn't expose the expression text via PRAGMA)
                parts.append(f"/* <expr#{r['seqno']}> */{order}")
                incomplete_expr = True
            else:
                parts.append(ident(nm) + order)

        unique = "UNIQUE " if int(idx["unique"] or 0) == 1 else ""
        cols = ", ".join(parts) if parts else "/* <no-columns> */"
        stmt = f"CREATE {unique}INDEX {ident(name)} ON {ident(table)} ({cols});"
        if incomplete_expr:
            stmt += " -- expression index; exact expression not available via PRAGMA"
        ddls.append(stmt)
    return ddls


def get_row_count(conn: sqlite3.Connection, table: str) -> int | None:
    try:
        r = conn.execute(f"SELECT COUNT(*) AS c FROM {ident(table)}").fetchone()
        return int(r["c"])
    except Exception:
        return None


# ---------- Markdown emission ----------

def main() -> None:
    ap = argparse.ArgumentParser(description="Dump reconstructed SQLite DDL (from PRAGMAs) to Markdown.")
    ap.add_argument("--db", default="db/trading.db", help="Path to SQLite database (default: db/trading.db)")
    ap.add_argument("--out", default="", help="Output .md path (default: stdout)")
    ap.add_argument("--counts", action="store_true", help="Include row counts per table (optional)")
    args = ap.parse_args()

    abs_db = os.path.abspath(args.db)
    if not os.path.exists(abs_db):
        raise SystemExit(f"Database not found: {abs_db}")

    conn = mk_conn(abs_db)
    try:
        now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat() + "Z"
        sqlite_ver = conn.execute("SELECT sqlite_version() AS v").fetchone()["v"]

        tables = get_tables(conn)

        lines: List[str] = []
        lines.append(f"# Reconstructed DDL (from PRAGMAs) — `{abs_db}`")
        lines.append("")
        lines.append(f"- Generated: `{now}`")
        lines.append(f"- SQLite version: `{sqlite_ver}`")
        lines.append("")
        lines.append("> NOTE: Views and triggers are omitted (no PRAGMA-based reconstruction).")
        lines.append("")
        lines.append("---")
        lines.append("")

        for t in tables:
            lines.append(f"## {ident(t)}")
            lines.append("")
            # Optional row count
            if args.counts:
                cnt = get_row_count(conn, t)
                if cnt is not None:
                    lines.append(f"- **Row count:** {cnt}")
                    lines.append("")

            # CREATE TABLE
            table_sql = reconstruct_create_table(conn, t)
            lines.append("```sql")
            lines.append(table_sql)
            lines.append("```")
            lines.append("")

            # CREATE INDEX statements
            idx_sqls = reconstruct_indexes(conn, t)
            if idx_sqls:
                lines.append("```sql")
                for stmt in idx_sqls:
                    lines.append(stmt)
                lines.append("```")
                lines.append("")

            lines.append("---")
            lines.append("")

        content = "\n".join(lines)
        out = args.out.strip()
        if out:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Wrote Markdown to: {os.path.abspath(out)}")
        else:
            print(content)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
