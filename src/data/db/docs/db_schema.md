# SQLite schema for `C:\dev\cursor\e-trading\db\trading.db`

- Generated: `2025-09-27T20:46:00Z`
- SQLite version: `3.45.3`

---

# Tables

## `alembic_version`

**DDL**

```sql
CREATE TABLE alembic_version (
  version_num VARCHAR(32) NOT NULL,
  CONSTRAINT pk_alembic_version PRIMARY KEY (version_num)
)
```

**Columns**

| cid | name | type | notnull | default | pk |
|----:|------|------|--------:|---------|---:|
| 0 | version_num | VARCHAR(32) | 1 |  | 1 |

**Indexes**

| name | unique | origin | partial | columns/expressions |
|------|-------:|--------|---------|---------------------|
| sqlite_autoindex_alembic_version_1 | 1 | pk | 0 | version_num ASC |

---

## `auth_identities`

**DDL**

```sql
CREATE TABLE auth_identities (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  provider VARCHAR(32) NOT NULL,
  external_id VARCHAR(255) NOT NULL,
  metadata JSON,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT uq_auth_identities_provider_external_id
    UNIQUE (provider, external_id),
  CONSTRAINT fk_auth_identities_user_id_users
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
)
```

**Columns**

| cid | name | type | notnull | default | pk |
|----:|------|------|--------:|---------|---:|
| 0 | id | INTEGER | 0 |  | 1 |
| 1 | user_id | INTEGER | 1 |  | 0 |
| 2 | provider | VARCHAR(32) | 1 |  | 0 |
| 3 | external_id | VARCHAR(255) | 1 |  | 0 |
| 4 | metadata | JSON | 0 |  | 0 |
| 5 | created_at | DATETIME | 0 | CURRENT_TIMESTAMP | 0 |

**Foreign keys**

| id | seq | from | → table | to | on_update | on_delete | match |
|---:|----:|------|---------|----|-----------|-----------|-------|
| 0 | 0 | user_id | users | id | NO ACTION | CASCADE | NONE |

**Indexes**

| name | unique | origin | partial | columns/expressions |
|------|-------:|--------|---------|---------------------|
| ix_auth_identities_provider | 0 | c | 0 | provider ASC |
| ix_auth_identities_user_id | 0 | c | 0 | user_id ASC |
| sqlite_autoindex_auth_identities_1 | 1 | u | 0 | provider ASC, external_id ASC |

---

## `telegram_alerts`

**DDL**

```sql
CREATE TABLE telegram_alerts (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL, status TEXT, email BOOLEAN, created_at DATETIME, config_json TEXT, re_arm_config TEXT, trigger_count INTEGER, last_trigger_condition TEXT, last_triggered_at DATETIME,
  CONSTRAINT fk_telegram_alerts_user_id_users
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
)
```

**Columns**

| cid | name | type | notnull | default | pk |
|----:|------|------|--------:|---------|---:|
| 0 | id | INTEGER | 0 |  | 1 |
| 1 | user_id | INTEGER | 1 |  | 0 |
| 2 | status | TEXT | 0 |  | 0 |
| 3 | email | BOOLEAN | 0 |  | 0 |
| 4 | created_at | DATETIME | 0 |  | 0 |
| 5 | config_json | TEXT | 0 |  | 0 |
| 6 | re_arm_config | TEXT | 0 |  | 0 |
| 7 | trigger_count | INTEGER | 0 |  | 0 |
| 8 | last_trigger_condition | TEXT | 0 |  | 0 |
| 9 | last_triggered_at | DATETIME | 0 |  | 0 |

**Foreign keys**

| id | seq | from | → table | to | on_update | on_delete | match |
|---:|----:|------|---------|----|-----------|-----------|-------|
| 0 | 0 | user_id | users | id | NO ACTION | CASCADE | NONE |

**Indexes**

| name | unique | origin | partial | columns/expressions |
|------|-------:|--------|---------|---------------------|
| ix_telegram_alerts_user_id | 0 | c | 0 | user_id ASC |

---

## `telegram_broadcast_logs`

**DDL**

```sql
CREATE TABLE telegram_broadcast_logs (
  id INTEGER PRIMARY KEY,
  message TEXT NOT NULL,
  sent_by VARCHAR(255) NOT NULL,
  success_count INTEGER,
  total_count INTEGER)
```

**Columns**

| cid | name | type | notnull | default | pk |
|----:|------|------|--------:|---------|---:|
| 0 | id | INTEGER | 0 |  | 1 |
| 1 | message | TEXT | 1 |  | 0 |
| 2 | sent_by | VARCHAR(255) | 1 |  | 0 |
| 3 | success_count | INTEGER | 0 |  | 0 |
| 4 | total_count | INTEGER | 0 |  | 0 |

---

## `telegram_command_audits`

**DDL**

```sql
CREATE TABLE telegram_command_audits (
  id INTEGER PRIMARY KEY,
  telegram_user_id VARCHAR(255) NOT NULL,
  command VARCHAR(255) NOT NULL,
  full_message TEXT,
  is_registered_user BOOLEAN,
  user_email VARCHAR(255),
  success BOOLEAN,
  error_message TEXT,
  response_time_ms INTEGER,
  created_at DATETIME)
```

**Columns**

| cid | name | type | notnull | default | pk |
|----:|------|------|--------:|---------|---:|
| 0 | id | INTEGER | 0 |  | 1 |
| 1 | telegram_user_id | VARCHAR(255) | 1 |  | 0 |
| 2 | command | VARCHAR(255) | 1 |  | 0 |
| 3 | full_message | TEXT | 0 |  | 0 |
| 4 | is_registered_user | BOOLEAN | 0 |  | 0 |
| 5 | user_email | VARCHAR(255) | 0 |  | 0 |
| 6 | success | BOOLEAN | 0 |  | 0 |
| 7 | error_message | TEXT | 0 |  | 0 |
| 8 | response_time_ms | INTEGER | 0 |  | 0 |
| 9 | created_at | DATETIME | 0 |  | 0 |

**Indexes**

| name | unique | origin | partial | columns/expressions |
|------|-------:|--------|---------|---------------------|
| ix_telegram_command_audits_created | 0 | c | 0 | created_at ASC |
| ix_telegram_command_audits_command | 0 | c | 0 | command ASC |
| ix_telegram_command_audits_success | 0 | c | 0 | success ASC |
| ix_telegram_command_audits_telegram_user_id | 0 | c | 0 | telegram_user_id ASC |

---

## `telegram_feedbacks`

**DDL**

```sql
CREATE TABLE telegram_feedbacks (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  CONSTRAINT fk_telegram_feedbacks_user_id_users
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
)
```

**Columns**

| cid | name | type | notnull | default | pk |
|----:|------|------|--------:|---------|---:|
| 0 | id | INTEGER | 0 |  | 1 |
| 1 | user_id | INTEGER | 1 |  | 0 |

**Foreign keys**

| id | seq | from | → table | to | on_update | on_delete | match |
|---:|----:|------|---------|----|-----------|-----------|-------|
| 0 | 0 | user_id | users | id | NO ACTION | CASCADE | NONE |

---

## `telegram_schedules`

**DDL**

```sql
CREATE TABLE telegram_schedules (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  CONSTRAINT fk_telegram_schedules_user_id_users
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
)
```

**Columns**

| cid | name | type | notnull | default | pk |
|----:|------|------|--------:|---------|---:|
| 0 | id | INTEGER | 0 |  | 1 |
| 1 | user_id | INTEGER | 1 |  | 0 |

**Foreign keys**

| id | seq | from | → table | to | on_update | on_delete | match |
|---:|----:|------|---------|----|-----------|-----------|-------|
| 0 | 0 | user_id | users | id | NO ACTION | CASCADE | NONE |

---

## `telegram_settings`

**DDL**

```sql
CREATE TABLE telegram_settings (
  "key" VARCHAR(100) NOT NULL,
  value TEXT,
  CONSTRAINT pk_telegram_settings PRIMARY KEY ("key")
)
```

**Columns**

| cid | name | type | notnull | default | pk |
|----:|------|------|--------:|---------|---:|
| 0 | key | VARCHAR(100) | 1 |  | 1 |
| 1 | value | TEXT | 0 |  | 0 |

**Indexes**

| name | unique | origin | partial | columns/expressions |
|------|-------:|--------|---------|---------------------|
| sqlite_autoindex_telegram_settings_1 | 1 | pk | 0 | key ASC |

---

## `telegram_verification_codes`

**DDL**

```sql
CREATE TABLE telegram_verification_codes (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  code VARCHAR(32) NOT NULL,
  sent_time INTEGER NOT NULL,
  CONSTRAINT fk_telegram_verification_codes_user_id_users
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
)
```

**Columns**

| cid | name | type | notnull | default | pk |
|----:|------|------|--------:|---------|---:|
| 0 | id | INTEGER | 0 |  | 1 |
| 1 | user_id | INTEGER | 1 |  | 0 |
| 2 | code | VARCHAR(32) | 1 |  | 0 |
| 3 | sent_time | INTEGER | 1 |  | 0 |

**Foreign keys**

| id | seq | from | → table | to | on_update | on_delete | match |
|---:|----:|------|---------|----|-----------|-----------|-------|
| 0 | 0 | user_id | users | id | NO ACTION | CASCADE | NONE |

**Indexes**

| name | unique | origin | partial | columns/expressions |
|------|-------:|--------|---------|---------------------|
| ix_telegram_verification_codes_user_id | 0 | c | 0 | user_id ASC |

---

## `trading_bot_instances`

**DDL**

```sql
CREATE TABLE trading_bot_instances (
  id VARCHAR(255) NOT NULL,
  type VARCHAR(20) NOT NULL,
  config_file VARCHAR(255),
  status VARCHAR(20) NOT NULL,
  started_at DATETIME,
  last_heartbeat DATETIME,
  error_count INTEGER,
  current_balance NUMERIC(20,8),
  total_pnl NUMERIC(20,8),
  extra_metadata JSON,
  created_at DATETIME,
  updated_at DATETIME,
  CONSTRAINT pk_trading_bot_instances PRIMARY KEY (id),
  CONSTRAINT ck_trading_bot_instances_valid_bot_type
    CHECK (type IN ('live','paper','optimization')),
  CONSTRAINT ck_trading_bot_instances_valid_bot_status
    CHECK (status IN ('running','stopped','error','completed'))
)
```

**Columns**

| cid | name | type | notnull | default | pk |
|----:|------|------|--------:|---------|---:|
| 0 | id | VARCHAR(255) | 1 |  | 1 |
| 1 | type | VARCHAR(20) | 1 |  | 0 |
| 2 | config_file | VARCHAR(255) | 0 |  | 0 |
| 3 | status | VARCHAR(20) | 1 |  | 0 |
| 4 | started_at | DATETIME | 0 |  | 0 |
| 5 | last_heartbeat | DATETIME | 0 |  | 0 |
| 6 | error_count | INTEGER | 0 |  | 0 |
| 7 | current_balance | NUMERIC(20,8) | 0 |  | 0 |
| 8 | total_pnl | NUMERIC(20,8) | 0 |  | 0 |
| 9 | extra_metadata | JSON | 0 |  | 0 |
| 10 | created_at | DATETIME | 0 |  | 0 |
| 11 | updated_at | DATETIME | 0 |  | 0 |

**Indexes**

| name | unique | origin | partial | columns/expressions |
|------|-------:|--------|---------|---------------------|
| ix_trading_bot_instances_last_heartbeat | 0 | c | 0 | last_heartbeat ASC |
| ix_trading_bot_instances_type | 0 | c | 0 | type ASC |
| ix_trading_bot_instances_status | 0 | c | 0 | status ASC |
| sqlite_autoindex_trading_bot_instances_1 | 1 | pk | 0 | id ASC |

---

## `trading_performance_metrics`

**DDL**

```sql
CREATE TABLE trading_performance_metrics (
  id VARCHAR(36) PRIMARY KEY,
  bot_id VARCHAR(255) NOT NULL,
  trade_type VARCHAR(10) NOT NULL,
  symbol VARCHAR(20),
  interval VARCHAR(10),
  entry_logic_name VARCHAR(100),
  exit_logic_name VARCHAR(100),
  metrics JSON NOT NULL,
  calculated_at DATETIME,
  created_at DATETIME,
  CONSTRAINT fk_trading_performance_metrics_bot_id_trading_bot_instances
    FOREIGN KEY (bot_id) REFERENCES trading_bot_instances(id) ON DELETE CASCADE,
  CONSTRAINT ck_trading_performance_metrics_trade_type
    CHECK (trade_type IN ('paper','live','optimization'))
)
```

**Columns**

| cid | name | type | notnull | default | pk |
|----:|------|------|--------:|---------|---:|
| 0 | id | VARCHAR(36) | 0 |  | 1 |
| 1 | bot_id | VARCHAR(255) | 1 |  | 0 |
| 2 | trade_type | VARCHAR(10) | 1 |  | 0 |
| 3 | symbol | VARCHAR(20) | 0 |  | 0 |
| 4 | interval | VARCHAR(10) | 0 |  | 0 |
| 5 | entry_logic_name | VARCHAR(100) | 0 |  | 0 |
| 6 | exit_logic_name | VARCHAR(100) | 0 |  | 0 |
| 7 | metrics | JSON | 1 |  | 0 |
| 8 | calculated_at | DATETIME | 0 |  | 0 |
| 9 | created_at | DATETIME | 0 |  | 0 |

**Foreign keys**

| id | seq | from | → table | to | on_update | on_delete | match |
|---:|----:|------|---------|----|-----------|-----------|-------|
| 0 | 0 | bot_id | trading_bot_instances | id | NO ACTION | CASCADE | NONE |

**Indexes**

| name | unique | origin | partial | columns/expressions |
|------|-------:|--------|---------|---------------------|
| ix_trading_performance_metrics_bot_id_calculated_at | 0 | c | 0 | bot_id ASC, calculated_at ASC |
| ix_trading_performance_metrics_symbol | 0 | c | 0 | symbol ASC |
| ix_trading_performance_metrics_calculated_at | 0 | c | 0 | calculated_at ASC |
| ix_trading_performance_metrics_bot_id | 0 | c | 0 | bot_id ASC |
| sqlite_autoindex_trading_performance_metrics_1 | 1 | pk | 0 | id ASC |

---

## `trading_positi