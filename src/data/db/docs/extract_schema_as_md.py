#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract a SQLite schema into a Markdown document.

Usage:
  # simplest (uses default db/trading.db, prints to stdout)
  python extract_schema_as_md.py

  # write to a file
  python extract_schema_as_md.py --out schema.md

  # specify another DB and include row counts
  python extract_schema_as_md.py --db path/to/other.db --out schema.md --counts

Notes:
- Prints: database header, tables (DDL, columns, FKs, indexes, triggers, optional row counts),
  and views (DDL). System objects 'sqlite_%' are excluded.
- Uses PRAGMA table_info to list *all* columns reliably.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sqlite3
from typing import Dict, List, Tuple, Any


def mk_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def md_escape(text: str) -> str:
    """Escape pipes in markdown table cells."""
    if text is None:
        return ""
    return str(text).replace("|", r"\|")


def get_sqlite_version(conn: sqlite3.Connection) -> str:
    v = conn.execute("select sqlite_version() as v").fetchone()["v"]
    return v


def get_tables(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    # Exclude SQLite's internal tables
    sql = """
      select type, name, tbl_name, sql
      from sqlite_master
      where type in ('table','view')
        and name not like 'sqlite_%'
      order by case type when 'table' then 0 else 1 end, name
    """
    return list(conn.execute(sql))


def get_table_info(conn: sqlite3.Connection, name: str) -> List[sqlite3.Row]:
    return list(conn.execute(f"PRAGMA table_info('{name}')"))


def get_foreign_keys(conn: sqlite3.Connection, name: str) -> List[sqlite3.Row]:
    return list(conn.execute(f"PRAGMA foreign_key_list('{name}')"))


def get_index_list(conn: sqlite3.Connection, name: str) -> List[sqlite3.Row]:
    return list(conn.execute(f"PRAGMA index_list('{name}')"))


def get_index_info(conn: sqlite3.Connection, index_name: str) -> Tuple[List[sqlite3.Row], List[sqlite3.Row]]:
    # index_info: columns; index_xinfo: columns + expressions + sort order
    info = list(conn.execute(f"PRAGMA index_info('{index_name}')"))
    xinfo = list(conn.execute(f"PRAGMA index_xinfo('{index_name}')"))
    return info, xinfo


def get_triggers_for_table(conn: sqlite3.Connection, table_name: str) -> List[sqlite3.Row]:
    sql = """
      select name, tbl_name, sql
      from sqlite_master
      where type = 'trigger'
        and tbl_name = ?
        and name not like 'sqlite_%'
      order by name
    """
    return list(conn.execute(sql, (table_name,)))


def get_row_count(conn: sqlite3.Connection, table_name: str) -> int | None:
    try:
        # Skip views here; only attempt for real tables
        is_view = conn.execute(
            "select 1 from sqlite_master where type='view' and name=?", (table_name,)
        ).fetchone()
        if is_view:
            return None
        row = conn.execute(f"select count(*) as c from '{table_name}'").fetchone()
        return int(row["c"])
    except Exception:
        return None


def render_index_columns(xinfo: List[sqlite3.Row]) -> str:
    """
    Render columns/expressions for an index from PRAGMA index_xinfo.
    - xinfo columns: seqno, cid, name, desc, coll, key, etc.
    - For expression-based indexes, cid == -2 and 'name' is None; expression text is not available via pragma.
      SQLite doesn't expose the expression via PRAGMA, so we show <expr#N>.
    """
    if not xinfo:
        return ""
    parts: List[str] = []
    # Filter key columns only (key=1)
    for r in sorted(xinfo, key=lambda r: (r["seqno"] if r["seqno"] is not None else 0)):
        if "key" in r.keys() and r["key"] != 1:
            continue
        cid = r["cid"]
        nm = r["name"]
        desc = r["desc"]
        # order: NULL means ASC (SQLite), 1 => DESC
        order = "DESC" if (desc == 1) else "ASC"
        if cid == -2:  # expression
            parts.append(f"<expr#{r['seqno']}> {order}")
        else:
            parts.append(f"{nm} {order}")
    return ", ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump SQLite schema to Markdown.")
    ap.add_argument("--db", default="db/trading.db", help="Path to SQLite database (default: db/trading.db)")
    ap.add_argument("--out", default="", help="Output .md path (default: stdout)")
    ap.add_argument("--counts", action="store_true", help="Include row counts for tables")
    args = ap.parse_args()

    db_path = args.db
    out_path = args.out.strip()
    include_counts = args.counts

    abs_db = os.path.abspath(db_path)
    if not os.path.exists(abs_db):
        raise SystemExit(f"Database not found: {abs_db}")

    conn = mk_conn(abs_db)
    try:
        now = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        sqlite_ver = get_sqlite_version(conn)
        objects = get_tables(conn)

        lines: List[str] = []
        lines.append(f"# SQLite schema for `{abs_db}`")
        lines.append("")
        lines.append(f"- Generated: `{now}`")
        lines.append(f"- SQLite version: `{sqlite_ver}`")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Group objects
        tables = [o for o in objects if o["type"] == "table"]
        views = [o for o in objects if o["type"] == "view"]

        # Tables
        if tables:
            lines.append("# Tables")
            lines.append("")
        for t in tables:
            name = t["name"]
            ddl = t["sql"]

            lines.append(f"## `{name}`")
            lines.append("")
            if include_counts:
                cnt = get_row_count(conn, name)
                if cnt is not None:
                    lines.append(f"- **Row count:** {cnt}")
                    lines.append("")

            # DDL
            if ddl:
                lines.append("**DDL**")
                lines.append("")
                lines.append("```sql")
                lines.append(ddl)
                lines.append("```")
                lines.append("")
            else:
                lines.append("_No DDL available (virtual table or created without SQL)_")
                lines.append("")

            # Columns
            cols = get_table_info(conn, name)
            lines.append("**Columns**")
            lines.append("")
            if cols:
                lines.append("| cid | name | type | notnull | default | pk |")
                lines.append("|----:|------|------|--------:|---------|---:|")
                for c in cols:
                    dflt = "" if c["dflt_value"] is None else str(c["dflt_value"])
                    lines.append(
                        f"| {c['cid']} | {md_escape(c['name'])} | {md_escape(c['type'])} | "
                        f"{c['notnull']} | {md_escape(dflt)} | {c['pk']} |"
                    )
                lines.append("")
            else:
                lines.append("_No columns (unexpected)_")
                lines.append("")

            # Foreign keys
            fks = get_foreign_keys(conn, name)
            if fks:
                lines.append("**Foreign keys**")
                lines.append("")
                lines.append("| id | seq | from | → table | to | on_update | on_delete | match |")
                lines.append("|---:|----:|------|---------|----|-----------|-----------|-------|")
                for fk in fks:
                    lines.append(
                        f"| {fk['id']} | {fk['seq']} | {md_escape(fk['from'])} | "
                        f"{md_escape(fk['table'])} | {md_escape(fk['to'])} | "
                        f"{fk['on_update']} | {fk['on_delete']} | {fk['match']} |"
                    )
                lines.append("")

            # Indexes
            idxs = get_index_list(conn, name)
            if idxs:
                lines.append("**Indexes**")
                lines.append("")
                lines.append("| name | unique | origin | partial | columns/expressions |")
                lines.append("|------|-------:|--------|---------|---------------------|")
                for i in idxs:
                    idx_name = i["name"]
                    unique = i["unique"]
                    origin = i["origin"]
                    partial = i["partial"]
                    _, xinfo = get_index_info(conn, idx_name)
                    cols_expr = render_index_columns(xinfo)
                    lines.append(
                        f"| {md_escape(idx_name)} | {unique} | {origin} | {partial} | {md_escape(cols_expr)} |"
                    )
                lines.append("")

            # Triggers
            trgs = get_triggers_for_table(conn, name)
            if trgs:
                lines.append("**Triggers**")
                lines.append("")
                for tr in trgs:
                    lines.append(f"- `{tr['name']}`")
                lines.append("")

            lines.append("---")
            lines.append("")

        # Views
        if views:
            lines.append("# Views")
            lines.append("")
            for v in views:
                name = v["name"]
                ddl = v["sql"]
                lines.append(f"## `{name}`")
                lines.append("")
                if ddl:
                    lines.append("**DDL**")
                    lines.append("")
                    lines.append("```sql")
                    lines.append(ddl)
                    lines.append("```")
                    lines.append("")
                else:
                    lines.append("_No DDL available for this view_")
                    lines.append("")
                lines.append("---")
                lines.append("")

        # Output
        content = "\n".join(lines)
        if out_path:
            abs_out = os.path.abspath(out_path)
            os.makedirs(os.path.dirname(abs_out), exist_ok=True)
            with open(abs_out, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Wrote Markdown to: {abs_out}")
        else:
            print(content)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
