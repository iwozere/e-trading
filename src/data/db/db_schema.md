## bot_instances

```sql
CREATE TABLE bot_instances (
        id VARCHAR(255) NOT NULL, 
        type VARCHAR(20) NOT NULL,
        config_file VARCHAR(255),
        status VARCHAR(20) NOT NULL,
        started_at DATETIME,
        last_heartbeat DATETIME,
        error_count INTEGER,
        current_balance NUMERIC(20, 8),
        total_pnl NUMERIC(20, 8),
        extra_metadata JSON,
        created_at DATETIME,
        updated_at DATETIME,
        PRIMARY KEY (id),
        CONSTRAINT valid_bot_type CHECK (type IN ('live', 'paper', 'optimization')),
        CONSTRAINT valid_bot_status CHECK (status IN ('running', 'stopped', 'error', 'completed'))
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
| 7 | current_balance | NUMERIC(20, 8) | 0 | None | 0 |
| 8 | total_pnl | NUMERIC(20, 8) | 0 | None | 0 |
| 9 | extra_metadata | JSON | 0 | None | 0 |
| 10 | created_at | DATETIME | 0 | None | 0 |
| 11 | updated_at | DATETIME | 0 | None | 0 |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| idx_bot_status | 0 | c | 0 |
| idx_bot_type | 0 | c | 0 |
| idx_last_heartbeat | 0 | c | 0 |
| sqlite_autoindex_bot_instances_1 | 1 | pk | 0 |

## performance_metrics

```sql
CREATE TABLE performance_metrics (
        id VARCHAR(36) NOT NULL,
        bot_id VARCHAR(255) NOT NULL,
        trade_type VARCHAR(10) NOT NULL,
        symbol VARCHAR(20),
        interval VARCHAR(10),
        entry_logic_name VARCHAR(100),
        exit_logic_name VARCHAR(100),
        metrics JSON NOT NULL,
        calculated_at DATETIME,
        created_at DATETIME,
        PRIMARY KEY (id),
        CONSTRAINT valid_metrics_trade_type CHECK (trade_type IN ('paper', 'live', 'optimization'))
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | VARCHAR(36) | 1 | None | 1 |
| 1 | bot_id | VARCHAR(255) | 1 | None | 0 |
| 2 | trade_type | VARCHAR(10) | 1 | None | 0 |
| 3 | symbol | VARCHAR(20) | 0 | None | 0 |
| 4 | interval | VARCHAR(10) | 0 | None | 0 |
| 5 | entry_logic_name | VARCHAR(100) | 0 | None | 0 |
| 6 | exit_logic_name | VARCHAR(100) | 0 | None | 0 |
| 7 | metrics | JSON | 1 | None | 0 |
| 8 | calculated_at | DATETIME | 0 | None | 0 |
| 9 | created_at | DATETIME | 0 | None | 0 |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| ix_performance_metrics_bot_id | 0 | c | 0 |
| idx_metrics_calculated_at | 0 | c | 0 |
| idx_metrics_strategy | 0 | c | 0 |
| idx_metrics_bot_id | 0 | c | 0 |
| sqlite_autoindex_performance_metrics_1 | 1 | pk | 0 |

## telegram_alerts

```sql
CREATE TABLE telegram_alerts (
        id INTEGER NOT NULL,
        ticker VARCHAR(50) NOT NULL,
        user_id VARCHAR(255) NOT NULL,
        price NUMERIC(20, 8),
        condition VARCHAR(50),
        active BOOLEAN,
        email BOOLEAN,
        created VARCHAR(40),
        alert_type VARCHAR(20),
        timeframe VARCHAR(10),
        config_json TEXT,
        alert_action VARCHAR(50), re_arm_config TEXT, is_armed BOOLEAN DEFAULT 1, last_price REAL, last_triggered_at TEXT,
        PRIMARY KEY (id),
        FOREIGN KEY(user_id) REFERENCES telegram_users (telegram_user_id)
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | INTEGER | 1 | None | 1 |
| 1 | ticker | VARCHAR(50) | 1 | None | 0 |
| 2 | user_id | VARCHAR(255) | 1 | None | 0 |
| 3 | price | NUMERIC(20, 8) | 0 | None | 0 |
| 4 | condition | VARCHAR(50) | 0 | None | 0 |
| 5 | active | BOOLEAN | 0 | None | 0 |
| 6 | email | BOOLEAN | 0 | None | 0 |
| 7 | created | VARCHAR(40) | 0 | None | 0 |
| 8 | alert_type | VARCHAR(20) | 0 | None | 0 |
| 9 | timeframe | VARCHAR(10) | 0 | None | 0 |
| 10 | config_json | TEXT | 0 | None | 0 |
| 11 | alert_action | VARCHAR(50) | 0 | None | 0 |
| 12 | re_arm_config | TEXT | 0 | None | 0 |
| 13 | is_armed | BOOLEAN | 0 | 1 | 0 |
| 14 | last_price | REAL | 0 | None | 0 |
| 15 | last_triggered_at | TEXT | 0 | None | 0 |

**Foreign keys**

| from | → table | to | on_update | on_delete |
|------|---------|----|-----------|-----------|
| user_id | telegram_users | telegram_user_id | NO ACTION | NO ACTION |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| ix_telegram_alerts_ticker | 0 | c | 0 |
| ix_telegram_alerts_user_id | 0 | c | 0 |
| ix_telegram_alerts_alert_type | 0 | c | 0 |
| ix_telegram_alerts_active | 0 | c | 0 |
| idx_alerts_user_active | 0 | c | 0 |

## telegram_broadcast_log

```sql
CREATE TABLE telegram_broadcast_log (
        id INTEGER NOT NULL,
        message TEXT NOT NULL,
        sent_by VARCHAR(255) NOT NULL,
        success_count INTEGER,
        total_count INTEGER,
        created VARCHAR(40),
        PRIMARY KEY (id)
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | INTEGER | 1 | None | 1 |
| 1 | message | TEXT | 1 | None | 0 |
| 2 | sent_by | VARCHAR(255) | 1 | None | 0 |
| 3 | success_count | INTEGER | 0 | None | 0 |
| 4 | total_count | INTEGER | 0 | None | 0 |
| 5 | created | VARCHAR(40) | 0 | None | 0 |

## telegram_command_audit

```sql
CREATE TABLE telegram_command_audit (
        id INTEGER NOT NULL,
        telegram_user_id VARCHAR(255) NOT NULL,
        command VARCHAR(255) NOT NULL,
        full_message TEXT,
        is_registered_user BOOLEAN,
        user_email VARCHAR(255),
        success BOOLEAN,
        error_message TEXT,
        response_time_ms INTEGER,
        created VARCHAR(40),
        PRIMARY KEY (id)
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | INTEGER | 1 | None | 1 |
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
| idx_command_audit_created | 0 | c | 0 |
| ix_telegram_command_audit_telegram_user_id | 0 | c | 0 |
| ix_telegram_command_audit_success | 0 | c | 0 |
| idx_command_audit_command | 0 | c | 0 |

## telegram_feedback

```sql
CREATE TABLE telegram_feedback (
        id INTEGER NOT NULL,
        user_id VARCHAR(255) NOT NULL,
        type VARCHAR(50) NOT NULL,
        message TEXT NOT NULL,
        created VARCHAR(40),
        status VARCHAR(20),
        PRIMARY KEY (id),
        FOREIGN KEY(user_id) REFERENCES telegram_users (telegram_user_id)
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | INTEGER | 1 | None | 1 |
| 1 | user_id | VARCHAR(255) | 1 | None | 0 |
| 2 | type | VARCHAR(50) | 1 | None | 0 |
| 3 | message | TEXT | 1 | None | 0 |
| 4 | created | VARCHAR(40) | 0 | None | 0 |
| 5 | status | VARCHAR(20) | 0 | None | 0 |

**Foreign keys**

| from | → table | to | on_update | on_delete |
|------|---------|----|-----------|-----------|
| user_id | telegram_users | telegram_user_id | NO ACTION | NO ACTION |

## telegram_schedules

```sql
CREATE TABLE telegram_schedules (
        id INTEGER NOT NULL,
        ticker VARCHAR(50),
        user_id VARCHAR(255) NOT NULL,
        scheduled_time VARCHAR(20) NOT NULL,
        period VARCHAR(20),
        active BOOLEAN,
        email BOOLEAN,
        indicators TEXT,
        interval VARCHAR(10),
        provider VARCHAR(20),
        created VARCHAR(40),
        schedule_type VARCHAR(20),
        list_type VARCHAR(50),
        config_json TEXT,
        schedule_config VARCHAR(20),
        PRIMARY KEY (id),
        FOREIGN KEY(user_id) REFERENCES telegram_users (telegram_user_id)
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | INTEGER | 1 | None | 1 |
| 1 | ticker | VARCHAR(50) | 0 | None | 0 |
| 2 | user_id | VARCHAR(255) | 1 | None | 0 |
| 3 | scheduled_time | VARCHAR(20) | 1 | None | 0 |
| 4 | period | VARCHAR(20) | 0 | None | 0 |
| 5 | active | BOOLEAN | 0 | None | 0 |
| 6 | email | BOOLEAN | 0 | None | 0 |
| 7 | indicators | TEXT | 0 | None | 0 |
| 8 | interval | VARCHAR(10) | 0 | None | 0 |
| 9 | provider | VARCHAR(20) | 0 | None | 0 |
| 10 | created | VARCHAR(40) | 0 | None | 0 |
| 11 | schedule_type | VARCHAR(20) | 0 | None | 0 |
| 12 | list_type | VARCHAR(50) | 0 | None | 0 |
| 13 | config_json | TEXT | 0 | None | 0 |
| 14 | schedule_config | VARCHAR(20) | 0 | None | 0 |

**Foreign keys**

| from | → table | to | on_update | on_delete |
|------|---------|----|-----------|-----------|
| user_id | telegram_users | telegram_user_id | NO ACTION | NO ACTION |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| ix_telegram_schedules_user_id | 0 | c | 0 |
| ix_telegram_schedules_active | 0 | c | 0 |
| ix_telegram_schedules_schedule_type | 0 | c | 0 |
| ix_telegram_schedules_schedule_config | 0 | c | 0 |

## telegram_settings

```sql
CREATE TABLE telegram_settings (
        "key" VARCHAR(100) NOT NULL,
        value TEXT,
        PRIMARY KEY ("key")
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

## telegram_users

```sql
CREATE TABLE telegram_users (
        telegram_user_id VARCHAR(255) NOT NULL,
        email VARCHAR(255),
        verification_code VARCHAR(32),
        code_sent_time INTEGER,
        verified BOOLEAN,
        approved BOOLEAN,
        language VARCHAR(10),
        is_admin BOOLEAN,
        max_alerts INTEGER,
        max_schedules INTEGER,
        created_at DATETIME,
        updated_at DATETIME, role VARCHAR(20) NOT NULL DEFAULT 'trader', is_active BOOLEAN NOT NULL DEFAULT TRUE, last_login DATETIME,
        PRIMARY KEY (telegram_user_id)
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | telegram_user_id | VARCHAR(255) | 1 | None | 1 |
| 1 | email | VARCHAR(255) | 0 | None | 0 |
| 2 | verification_code | VARCHAR(32) | 0 | None | 0 |
| 3 | code_sent_time | INTEGER | 0 | None | 0 |
| 4 | verified | BOOLEAN | 0 | None | 0 |
| 5 | approved | BOOLEAN | 0 | None | 0 |
| 6 | language | VARCHAR(10) | 0 | None | 0 |
| 7 | is_admin | BOOLEAN | 0 | None | 0 |
| 8 | max_alerts | INTEGER | 0 | None | 0 |
| 9 | max_schedules | INTEGER | 0 | None | 0 |
| 10 | created_at | DATETIME | 0 | None | 0 |
| 11 | updated_at | DATETIME | 0 | None | 0 |
| 12 | role | VARCHAR(20) | 1 | 'trader' | 0 |
| 13 | is_active | BOOLEAN | 1 | TRUE | 0 |
| 14 | last_login | DATETIME | 0 | None | 0 |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| ix_telegram_users_is_admin | 0 | c | 0 |
| idx_telegram_users_email | 0 | c | 0 |
| ix_telegram_users_verified | 0 | c | 0 |
| ix_telegram_users_approved | 0 | c | 0 |
| sqlite_autoindex_telegram_users_1 | 1 | pk | 0 |

## telegram_verification_codes

```sql
CREATE TABLE telegram_verification_codes (
        id INTEGER NOT NULL,
        telegram_user_id VARCHAR(255) NOT NULL,
        code VARCHAR(32) NOT NULL,
        sent_time INTEGER NOT NULL,
        PRIMARY KEY (id),
        FOREIGN KEY(telegram_user_id) REFERENCES telegram_users (telegram_user_id)
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | INTEGER | 1 | None | 1 |
| 1 | telegram_user_id | VARCHAR(255) | 1 | None | 0 |
| 2 | code | VARCHAR(32) | 1 | None | 0 |
| 3 | sent_time | INTEGER | 1 | None | 0 |

**Foreign keys**

| from | → table | to | on_update | on_delete |
|------|---------|----|-----------|-----------|
| telegram_user_id | telegram_users | telegram_user_id | NO ACTION | NO ACTION |

## trades

```sql
CREATE TABLE trades (
        id VARCHAR(36) NOT NULL,
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
        entry_price NUMERIC(20, 8),
        exit_price NUMERIC(20, 8),
        entry_value NUMERIC(20, 8),
        exit_value NUMERIC(20, 8),
        size NUMERIC(20, 8),
        direction VARCHAR(10) NOT NULL,
        commission NUMERIC(20, 8),
        gross_pnl NUMERIC(20, 8),
        net_pnl NUMERIC(20, 8),
        pnl_percentage NUMERIC(10, 4),
        exit_reason VARCHAR(100),
        status VARCHAR(20) NOT NULL,
        extra_metadata JSON,
        created_at DATETIME,
        updated_at DATETIME,
        PRIMARY KEY (id),
        CONSTRAINT valid_trade_type CHECK (trade_type IN ('paper', 'live', 'optimization')),
        CONSTRAINT valid_direction CHECK (direction IN ('long', 'short')),
        CONSTRAINT valid_status CHECK (status IN ('open', 'closed', 'cancelled'))
)
```

| # | column | type | not null | default | pk |
|-:|-------|------|----------:|---------|---:|
| 0 | id | VARCHAR(36) | 1 | None | 1 |
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
| 14 | entry_price | NUMERIC(20, 8) | 0 | None | 0 |
| 15 | exit_price | NUMERIC(20, 8) | 0 | None | 0 |
| 16 | entry_value | NUMERIC(20, 8) | 0 | None | 0 |
| 17 | exit_value | NUMERIC(20, 8) | 0 | None | 0 |
| 18 | size | NUMERIC(20, 8) | 0 | None | 0 |
| 19 | direction | VARCHAR(10) | 1 | None | 0 |
| 20 | commission | NUMERIC(20, 8) | 0 | None | 0 |
| 21 | gross_pnl | NUMERIC(20, 8) | 0 | None | 0 |
| 22 | net_pnl | NUMERIC(20, 8) | 0 | None | 0 |
| 23 | pnl_percentage | NUMERIC(10, 4) | 0 | None | 0 |
| 24 | exit_reason | VARCHAR(100) | 0 | None | 0 |
| 25 | status | VARCHAR(20) | 1 | None | 0 |
| 26 | extra_metadata | JSON | 0 | None | 0 |
| 27 | created_at | DATETIME | 0 | None | 0 |
| 28 | updated_at | DATETIME | 0 | None | 0 |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| ix_trades_status | 0 | c | 0 |
| idx_entry_time | 0 | c | 0 |
| idx_bot_trade_type | 0 | c | 0 |
| idx_symbol_status | 0 | c | 0 |
| idx_strategy | 0 | c | 0 |
| ix_trades_trade_type | 0 | c | 0 |
| ix_trades_bot_id | 0 | c | 0 |
| ix_trades_symbol | 0 | c | 0 |
| sqlite_autoindex_trades_1 | 1 | pk | 0 |

## users

```sql
CREATE TABLE "users" (
    id INTEGER PRIMARY KEY,
    email VARCHAR(100) UNIQUE,
    role VARCHAR(20) NOT NULL DEFAULT 'trader',
    is_active BOOLEAN DEFAULT TRUE,

    -- Telegram integration fields
    telegram_user_id VARCHAR(255) UNIQUE,
    telegram_verified BOOLEAN DEFAULT FALSE,
    telegram_approved BOOLEAN DEFAULT FALSE,
    telegram_language VARCHAR(10),
    telegram_is_admin BOOLEAN DEFAULT FALSE,
    telegram_max_alerts INTEGER,
    telegram_max_schedules INTEGER,
    telegram_verification_code VARCHAR(32),
    telegram_code_sent_time INTEGER,

    -- Audit fields
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
| 5 | telegram_verified | BOOLEAN | 0 | FALSE | 0 |
| 6 | telegram_approved | BOOLEAN | 0 | FALSE | 0 |
| 7 | telegram_language | VARCHAR(10) | 0 | None | 0 |
| 8 | telegram_is_admin | BOOLEAN | 0 | FALSE | 0 |
| 9 | telegram_max_alerts | INTEGER | 0 | None | 0 |
| 10 | telegram_max_schedules | INTEGER | 0 | None | 0 |
| 11 | telegram_verification_code | VARCHAR(32) | 0 | None | 0 |
| 12 | telegram_code_sent_time | INTEGER | 0 | None | 0 |
| 13 | created_at | DATETIME | 0 | CURRENT_TIMESTAMP | 0 |
| 14 | updated_at | DATETIME | 0 | None | 0 |
| 15 | last_login | DATETIME | 0 | None | 0 |

**Indexes**

| name | unique | origin | partial |
|------|-------:|--------|---------|
| ix_users_telegram_user_id | 0 | c | 0 |
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
                    FOREIGN KEY (user_id) REFERENCES users (id)
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
                    FOREIGN KEY (created_by) REFERENCES users (id)
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
                    key VARCHAR(100) UNIQUE NOT NULL,
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
| ix_webui_system_config_key | 0 | c | 0 |
| sqlite_autoindex_webui_system_config_1 | 1 | u | 0 |