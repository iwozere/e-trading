## alembic_version

```sql
CREATE TABLE alembic_version (
  version_num VARCHAR(32) NOT NULL,
  CONSTRAINT pk_alembic_version PRIMARY KEY (version_num)
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | version_num | VARCHAR(32) | 1 | None | 1 |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| sqlite_autoindex_alembic_version_1 | 1 | pk | 0 |

## auth_identities

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

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | INTEGER | 0 | None | 1 |
| 1 | user_id | INTEGER | 1 | None | 0 |
| 2 | provider | VARCHAR(32) | 1 | None | 0 |
| 3 | external_id | VARCHAR(255) | 1 | None | 0 |
| 4 | metadata | JSON | 0 | None | 0 |
| 5 | created_at | DATETIME | 0 | CURRENT_TIMESTAMP | 0 |

**Foreign keys**

| from | → table | to | on_update | on_delete |
|------|---------|----|-----------|-----------|
| user_id | users | id | NO ACTION | CASCADE |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| ix_auth_identities_provider | 0 | c | 0 |
| ix_auth_identities_user_id | 0 | c | 0 |
| sqlite_autoindex_auth_identities_1 | 1 | u | 0 |

## telegram_alerts

```sql
CREATE TABLE telegram_alerts (
  id INTEGER PRIMARY KEY,
  ticker VARCHAR(50) NOT NULL,
  user_id INTEGER NOT NULL,
  CONSTRAINT fk_telegram_alerts_user_id_users
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | INTEGER | 0 | None | 1 |
| 1 | ticker | VARCHAR(50) | 1 | None | 0 |
| 2 | user_id | INTEGER | 1 | None | 0 |

**Foreign keys**

| from | → table | to | on_update | on_delete |
|------|---------|----|-----------|-----------|
| user_id | users | id | NO ACTION | CASCADE |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| ix_telegram_alerts_user_id | 0 | c | 0 |

## telegram_broadcast_logs

```sql
CREATE TABLE telegram_broadcast_logs (
  id INTEGER PRIMARY KEY,
  message TEXT NOT NULL,
  sent_by VARCHAR(255) NOT NULL,
  success_count INTEGER,
  total_count INTEGER,
  created VARCHAR(40)
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | INTEGER | 0 | None | 1 |
| 1 | message | TEXT | 1 | None | 0 |
| 2 | sent_by | VARCHAR(255) | 1 | None | 0 |
| 3 | success_count | INTEGER | 0 | None | 0 |
| 4 | total_count | INTEGER | 0 | None | 0 |
| 5 | created | VARCHAR(40) | 0 | None | 0 |

## telegram_command_audits

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
  created VARCHAR(40)
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | INTEGER | 0 | None | 1 |
| 1 | telegram_user_id | VARCHAR(255) | 1 | None | 0 |
| 2 | command | VARCHAR(255) | 1 | None | 0 |
| 3 | full_message | TEXT | 0 | None | 0 |
| 4 | is_registered_user | BOOLEAN | 0 | None | 0 |
| 5 | user_email | VARCHAR(255) | 0 | None | 0 |
| 6 | success | BOOLEAN | 0 | None | 0 |
| 7 | error_message | TEXT | 0 | None | 0 |
| 8 | response_time_ms | INTEGER | 0 | None | 0 |
| 9 | created | VARCHAR(40) | 0 | None | 0 |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| ix_telegram_command_audits_command | 0 | c | 0 |
| ix_telegram_command_audits_success | 0 | c | 0 |
| ix_telegram_command_audits_telegram_user_id | 0 | c | 0 |
| ix_telegram_command_audits_created | 0 | c | 0 |

## telegram_feedbacks

```sql
CREATE TABLE telegram_feedbacks (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  CONSTRAINT fk_telegram_feedbacks_user_id_users
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | INTEGER | 0 | None | 1 |
| 1 | user_id | INTEGER | 1 | None | 0 |

**Foreign keys**

| from | → table | to | on_update | on_delete |
|------|---------|----|-----------|-----------|
| user_id | users | id | NO ACTION | CASCADE |

## telegram_schedules

```sql
CREATE TABLE telegram_schedules (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  CONSTRAINT fk_telegram_schedules_user_id_users
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | INTEGER | 0 | None | 1 |
| 1 | user_id | INTEGER | 1 | None | 0 |

**Foreign keys**

| from | → table | to | on_update | on_delete |
|------|---------|----|-----------|-----------|
| user_id | users | id | NO ACTION | CASCADE |

## telegram_settings

```sql
CREATE TABLE telegram_settings (
  "key" VARCHAR(100) NOT NULL,
  value TEXT,
  CONSTRAINT pk_telegram_settings PRIMARY KEY ("key")
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | key | VARCHAR(100) | 1 | None | 1 |
| 1 | value | TEXT | 0 | None | 0 |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| sqlite_autoindex_telegram_settings_1 | 1 | pk | 0 |

## telegram_verification_codes

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

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | INTEGER | 0 | None | 1 |
| 1 | user_id | INTEGER | 1 | None | 0 |
| 2 | code | VARCHAR(32) | 1 | None | 0 |
| 3 | sent_time | INTEGER | 1 | None | 0 |

**Foreign keys**

| from | → table | to | on_update | on_delete |
|------|---------|----|-----------|-----------|
| user_id | users | id | NO ACTION | CASCADE |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| ix_telegram_verification_codes_user_id | 0 | c | 0 |

## trading_bot_instances

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

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | VARCHAR(255) | 1 | None | 1 |
| 1 | type | VARCHAR(20) | 1 | None | 0 |
| 2 | config_file | VARCHAR(255) | 0 | None | 0 |
| 3 | status | VARCHAR(20) | 1 | None | 0 |
| 4 | started_at | DATETIME | 0 | None | 0 |
| 5 | last_heartbeat | DATETIME | 0 | None | 0 |
| 6 | error_count | INTEGER | 0 | None | 0 |
| 7 | current_balance | NUMERIC(20,8) | 0 | None | 0 |
| 8 | total_pnl | NUMERIC(20,8) | 0 | None | 0 |
| 9 | extra_metadata | JSON | 0 | None | 0 |
| 10 | created_at | DATETIME | 0 | None | 0 |
| 11 | updated_at | DATETIME | 0 | None | 0 |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| ix_trading_bot_instances_last_heartbeat | 0 | c | 0 |
| ix_trading_bot_instances_type | 0 | c | 0 |
| ix_trading_bot_instances_status | 0 | c | 0 |
| sqlite_autoindex_trading_bot_instances_1 | 1 | pk | 0 |

## trading_performance_metrics

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

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | VARCHAR(36) | 0 | None | 1 |
| 1 | bot_id | VARCHAR(255) | 1 | None | 0 |
| 2 | trade_type | VARCHAR(10) | 1 | None | 0 |
| 3 | symbol | VARCHAR(20) | 0 | None | 0 |
| 4 | interval | VARCHAR(10) | 0 | None | 0 |
| 5 | entry_logic_name | VARCHAR(100) | 0 | None | 0 |
| 6 | exit_logic_name | VARCHAR(100) | 0 | None | 0 |
| 7 | metrics | JSON | 1 | None | 0 |
| 8 | calculated_at | DATETIME | 0 | None | 0 |
| 9 | created_at | DATETIME | 0 | None | 0 |

**Foreign keys**

| from | → table | to | on_update | on_delete |
|------|---------|----|-----------|-----------|
| bot_id | trading_bot_instances | id | NO ACTION | CASCADE |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| ix_trading_performance_metrics_bot_id_calculated_at | 0 | c | 0 |
| ix_trading_performance_metrics_symbol | 0 | c | 0 |
| ix_trading_performance_metrics_calculated_at | 0 | c | 0 |
| ix_trading_performance_metrics_bot_id | 0 | c | 0 |
| sqlite_autoindex_trading_performance_metrics_1 | 1 | pk | 0 |

## trading_positions

```sql
CREATE TABLE trading_positions (
  id VARCHAR(36) PRIMARY KEY,
  bot_id VARCHAR(255) NOT NULL,
  trade_type VARCHAR(10) NOT NULL,
  symbol VARCHAR(20) NOT NULL,
  direction VARCHAR(10) NOT NULL,
  opened_at DATETIME,
  closed_at DATETIME,
  qty_open NUMERIC(20,8) NOT NULL DEFAULT 0,
  avg_price NUMERIC(20,8),
  realized_pnl NUMERIC(20,8) DEFAULT 0,
  status VARCHAR(12) NOT NULL,
  extra_metadata JSON,
  CONSTRAINT fk_trading_positions_bot_id_trading_bot_instances
    FOREIGN KEY (bot_id) REFERENCES trading_bot_instances(id) ON DELETE CASCADE,
  CONSTRAINT ck_trading_positions_direction
    CHECK (direction IN ('long','short')),
  CONSTRAINT ck_trading_positions_status
    CHECK (status IN ('open','closed'))
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | VARCHAR(36) | 0 | None | 1 |
| 1 | bot_id | VARCHAR(255) | 1 | None | 0 |
| 2 | trade_type | VARCHAR(10) | 1 | None | 0 |
| 3 | symbol | VARCHAR(20) | 1 | None | 0 |
| 4 | direction | VARCHAR(10) | 1 | None | 0 |
| 5 | opened_at | DATETIME | 0 | None | 0 |
| 6 | closed_at | DATETIME | 0 | None | 0 |
| 7 | qty_open | NUMERIC(20,8) | 1 | 0 | 0 |
| 8 | avg_price | NUMERIC(20,8) | 0 | None | 0 |
| 9 | realized_pnl | NUMERIC(20,8) | 0 | 0 | 0 |
| 10 | status | VARCHAR(12) | 1 | None | 0 |
| 11 | extra_metadata | JSON | 0 | None | 0 |

**Foreign keys**

| from | → table | to | on_update | on_delete |
|------|---------|----|-----------|-----------|
| bot_id | trading_bot_instances | id | NO ACTION | CASCADE |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| ix_trading_positions_bot_id_status | 0 | c | 0 |
| ix_trading_positions_symbol | 0 | c | 0 |
| ix_trading_positions_bot_id | 0 | c | 0 |
| sqlite_autoindex_trading_positions_1 | 1 | pk | 0 |

## trading_trades

```sql
CREATE TABLE trading_trades (
  id VARCHAR(36) PRIMARY KEY,
  bot_id VARCHAR(255) NOT NULL,
  trade_type VARCHAR(10) NOT NULL,
  strategy_name VARCHAR(100),
  entry_logic_name VARCHAR(100) NOT NULL,
  exit_logic_name VARCHAR(100) NOT NULL,
  symbol VARCHAR(20) NOT NULL,
  interval VARCHAR(10) NOT NULL,
  entry_time DATETIME,
  exit_time DATETIME,
  buy_order_created DATETIME,
  buy_order_closed DATETIME,
  sell_order_created DATETIME,
  sell_order_closed DATETIME,
  entry_price NUMERIC(20,8),
  exit_price NUMERIC(20,8),
  entry_value NUMERIC(20,8),
  exit_value NUMERIC(20,8),
  size NUMERIC(20,8),
  direction VARCHAR(10) NOT NULL,
  commission NUMERIC(20,8),
  gross_pnl NUMERIC(20,8),
  net_pnl NUMERIC(20,8),
  pnl_percentage NUMERIC(10,4),
  exit_reason VARCHAR(100),
  status VARCHAR(20) NOT NULL,
  extra_metadata JSON,
  created_at DATETIME,
  updated_at DATETIME,
  position_id VARCHAR(36),
  CONSTRAINT fk_trading_trades_bot_id_trading_bot_instances
    FOREIGN KEY (bot_id) REFERENCES trading_bot_instances(id) ON DELETE CASCADE,
  CONSTRAINT fk_trading_trades_position_id_trading_positions
    FOREIGN KEY (position_id) REFERENCES trading_positions(id) ON DELETE SET NULL,
  CONSTRAINT ck_trading_trades_trade_type
    CHECK (trade_type IN ('paper','live','optimization')),
  CONSTRAINT ck_trading_trades_direction
    CHECK (direction IN ('long','short')),
  CONSTRAINT ck_trading_trades_status
    CHECK (status IN ('open','closed','cancelled'))
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | VARCHAR(36) | 0 | None | 1 |
| 1 | bot_id | VARCHAR(255) | 1 | None | 0 |
| 2 | trade_type | VARCHAR(10) | 1 | None | 0 |
| 3 | strategy_name | VARCHAR(100) | 0 | None | 0 |
| 4 | entry_logic_name | VARCHAR(100) | 1 | None | 0 |
| 5 | exit_logic_name | VARCHAR(100) | 1 | None | 0 |
| 6 | symbol | VARCHAR(20) | 1 | None | 0 |
| 7 | interval | VARCHAR(10) | 1 | None | 0 |
| 8 | entry_time | DATETIME | 0 | None | 0 |
| 9 | exit_time | DATETIME | 0 | None | 0 |
| 10 | buy_order_created | DATETIME | 0 | None | 0 |
| 11 | buy_order_closed | DATETIME | 0 | None | 0 |
| 12 | sell_order_created | DATETIME | 0 | None | 0 |
| 13 | sell_order_closed | DATETIME | 0 | None | 0 |
| 14 | entry_price | NUMERIC(20,8) | 0 | None | 0 |
| 15 | exit_price | NUMERIC(20,8) | 0 | None | 0 |
| 16 | entry_value | NUMERIC(20,8) | 0 | None | 0 |
| 17 | exit_value | NUMERIC(20,8) | 0 | None | 0 |
| 18 | size | NUMERIC(20,8) | 0 | None | 0 |
| 19 | direction | VARCHAR(10) | 1 | None | 0 |
| 20 | commission | NUMERIC(20,8) | 0 | None | 0 |
| 21 | gross_pnl | NUMERIC(20,8) | 0 | None | 0 |
| 22 | net_pnl | NUMERIC(20,8) | 0 | None | 0 |
| 23 | pnl_percentage | NUMERIC(10,4) | 0 | None | 0 |
| 24 | exit_reason | VARCHAR(100) | 0 | None | 0 |
| 25 | status | VARCHAR(20) | 1 | None | 0 |
| 26 | extra_metadata | JSON | 0 | None | 0 |
| 27 | created_at | DATETIME | 0 | None | 0 |
| 28 | updated_at | DATETIME | 0 | None | 0 |
| 29 | position_id | VARCHAR(36) | 0 | None | 0 |

**Foreign keys**

| from | → table | to | on_update | on_delete |
|------|---------|----|-----------|-----------|
| position_id | trading_positions | id | NO ACTION | SET NULL |
| bot_id | trading_bot_instances | id | NO ACTION | CASCADE |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| ix_trading_trades_bot_id_symbol | 0 | c | 0 |
| ix_trading_trades_bot_id_entry_time | 0 | c | 0 |
| ix_trading_trades_bot_id_status | 0 | c | 0 |
| ix_trading_trades_strategy_name | 0 | c | 0 |
| ix_trading_trades_trade_type | 0 | c | 0 |
| ix_trading_trades_symbol | 0 | c | 0 |
| ix_trading_trades_bot_id | 0 | c | 0 |
| ix_trading_trades_entry_time | 0 | c | 0 |
| ix_trading_trades_status | 0 | c | 0 |
| sqlite_autoindex_trading_trades_1 | 1 | pk | 0 |

## users

```sql
CREATE TABLE "users" (
    id INTEGER PRIMARY KEY,
    email VARCHAR(100) UNIQUE,
    role VARCHAR(20) NOT NULL DEFAULT 'trader',
    is_active BOOLEAN DEFAULT TRUE,

    -- Telegram integration fields
    telegram_user_id VARCHAR(255) UNIQUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME,
    last_login DATETIME,

    -- Constraints
    CHECK (role IN ('admin', 'trader', 'viewer'))
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | INTEGER | 0 | None | 1 |
| 1 | email | VARCHAR(100) | 0 | None | 0 |
| 2 | role | VARCHAR(20) | 1 | 'trader' | 0 |
| 3 | is_active | BOOLEAN | 0 | TRUE | 0 |
| 4 | telegram_user_id | VARCHAR(255) | 0 | None | 0 |
| 5 | created_at | DATETIME | 0 | CURRENT_TIMESTAMP | 0 |
| 6 | updated_at | DATETIME | 0 | None | 0 |
| 7 | last_login | DATETIME | 0 | None | 0 |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| ix_users_email | 0 | c | 0 |
| sqlite_autoindex_users_2 | 1 | u | 0 |
| sqlite_autoindex_users_1 | 1 | u | 0 |

## webui_audit_logs

```sql
CREATE TABLE webui_audit_logs (
  id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL,
  action VARCHAR(100) NOT NULL,
  resource_type VARCHAR(50),
  resource_id VARCHAR(100),
  details JSON,
  ip_address VARCHAR(45),
  user_agent VARCHAR(500),
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_webui_audit_logs_user_id_users
    FOREIGN KEY (user_id) REFERENCES users(id)
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | INTEGER | 0 | None | 1 |
| 1 | user_id | INTEGER | 1 | None | 0 |
| 2 | action | VARCHAR(100) | 1 | None | 0 |
| 3 | resource_type | VARCHAR(50) | 0 | None | 0 |
| 4 | resource_id | VARCHAR(100) | 0 | None | 0 |
| 5 | details | JSON | 0 | None | 0 |
| 6 | ip_address | VARCHAR(45) | 0 | None | 0 |
| 7 | user_agent | VARCHAR(500) | 0 | None | 0 |
| 8 | created_at | DATETIME | 0 | CURRENT_TIMESTAMP | 0 |

**Foreign keys**

| from | → table | to | on_update | on_delete |
|------|---------|----|-----------|-----------|
| user_id | users | id | NO ACTION | NO ACTION |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| ix_webui_audit_logs_action | 0 | c | 0 |
| ix_webui_audit_logs_user_id | 0 | c | 0 |

## webui_performance_snapshots

```sql
CREATE TABLE webui_performance_snapshots (
  id INTEGER PRIMARY KEY,
  strategy_id VARCHAR(100) NOT NULL,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  pnl JSON NOT NULL,
  positions JSON,
  trades_count INTEGER DEFAULT 0,
  win_rate JSON,
  drawdown JSON,
  metrics JSON
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | INTEGER | 0 | None | 1 |
| 1 | strategy_id | VARCHAR(100) | 1 | None | 0 |
| 2 | timestamp | DATETIME | 0 | CURRENT_TIMESTAMP | 0 |
| 3 | pnl | JSON | 1 | None | 0 |
| 4 | positions | JSON | 0 | None | 0 |
| 5 | trades_count | INTEGER | 0 | 0 | 0 |
| 6 | win_rate | JSON | 0 | None | 0 |
| 7 | drawdown | JSON | 0 | None | 0 |
| 8 | metrics | JSON | 0 | None | 0 |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| ix_webui_performance_snapshots_strategy_id | 0 | c | 0 |

## webui_strategy_templates

```sql
CREATE TABLE webui_strategy_templates (
  id INTEGER PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  description TEXT,
  template_data JSON NOT NULL,
  is_public BOOLEAN DEFAULT FALSE,
  created_by INTEGER NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME,
  CONSTRAINT fk_webui_strategy_templates_created_by_users
    FOREIGN KEY (created_by) REFERENCES users(id)
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | INTEGER | 0 | None | 1 |
| 1 | name | VARCHAR(100) | 1 | None | 0 |
| 2 | description | TEXT | 0 | None | 0 |
| 3 | template_data | JSON | 1 | None | 0 |
| 4 | is_public | BOOLEAN | 0 | FALSE | 0 |
| 5 | created_by | INTEGER | 1 | None | 0 |
| 6 | created_at | DATETIME | 0 | CURRENT_TIMESTAMP | 0 |
| 7 | updated_at | DATETIME | 0 | None | 0 |

**Foreign keys**

| from | → table | to | on_update | on_delete |
|------|---------|----|-----------|-----------|
| created_by | users | id | NO ACTION | NO ACTION |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| ix_webui_strategy_templates_created_by | 0 | c | 0 |

## webui_system_config

```sql
CREATE TABLE webui_system_config (
  id INTEGER PRIMARY KEY,
  "key" VARCHAR(100) NOT NULL UNIQUE,
  value JSON NOT NULL,
  description TEXT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | INTEGER | 0 | None | 1 |
| 1 | key | VARCHAR(100) | 1 | None | 0 |
| 2 | value | JSON | 1 | None | 0 |
| 3 | description | TEXT | 0 | None | 0 |
| 4 | created_at | DATETIME | 0 | CURRENT_TIMESTAMP | 0 |
| 5 | updated_at | DATETIME | 0 | None | 0 |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| sqlite_autoindex_webui_system_config_1 | 1 | u | 0 |