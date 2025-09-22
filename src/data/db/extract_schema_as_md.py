import sqlite3, json

db_path = "db/trading.db"
con = sqlite3.connect(db_path)
con.row_factory = sqlite3.Row
cur = con.cursor()

# list tables
tables = [r["name"] for r in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
)]

for t in tables:
    print(f"\n## {t}")
    # DDL
    ddl = cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone()["sql"]
    print("\n```sql\n" + ddl + "\n```")
    # columns
    cols = list(cur.execute(f"PRAGMA table_info('{t}')"))
    print("\n| # | column | type | not null | default | pk |")
    print("|-:|-------|------|----------:|---------|---:|")
    for c in cols:
        print(f"| {c['cid']} | {c['name']} | {c['type']} | {c['notnull']} | {c['dflt_value']} | {c['pk']} |")
    # fks
    fks = list(cur.execute(f"PRAGMA foreign_key_list('{t}')"))
    if fks:
        print("\n**Foreign keys**")
        print("\n| from | → table | to | on_update | on_delete |")
        print("|------|---------|----|-----------|-----------|")
        for fk in fks:
            print(f"| {fk['from']} | {fk['table']} | {fk['to']} | {fk['on_update']} | {fk['on_delete']} |")
    # indexes
    idxs = list(cur.execute(f"PRAGMA index_list('{t}')"))
    if idxs:
        print("\n**Indexes**")
        print("\n| name | unique | origin | partial |")
        print("|------|-------:|--------|---------|")
        for i in idxs:
            print(f"| {i['name']} | {i['unique']} | {i['origin']} | {i['partial']} |")
