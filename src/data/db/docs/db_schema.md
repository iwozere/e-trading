# Reconstructed DDL (from PRAGMAs) — `C:\dev\cursor\e-trading\db\trading.db`

- Generated: `2025-09-27T20:52:26Z`
- SQLite version: `3.45.3`

> NOTE: Views and triggers are omitted (no PRAGMA-based reconstruction).

---

## "alembic_version"

```sql
CREATE TABLE "alembic_version" (
  "version_num" VARCHAR(32) PRIMARY KEY NOT NULL
);
```

---

## "auth_identities"

```sql
CREATE TABLE "auth_identities" (
  "id" INTEGER PRIMARY KEY,
  "user_id" INTEGER NOT NULL,
  "provider" VARCHAR(32) NOT NULL,
  "external_id" VARCHAR(255) NOT NULL,
  "metadata" JSON,
  "created_at" DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE CASCADE
);
```

```sql
CREATE INDEX "ix_auth_identities_provider" ON "auth_identities" ("provider");
CREATE INDEX "ix_auth_identities_user_id" ON "auth_identities" ("user_id");
```

---

## "telegram_alerts"

```sql
CREATE TABLE "telegram_alerts" (
  "id" INTEGER PRIMARY KEY,
  "user_id" INTEGER NOT NULL,
  "status" TEXT,
  "email" BOOLEAN,
  "created_at" DATETIME,
  "config_json" TEXT,
  "re_arm_config" TEXT,
  "trigger_count" INTEGER,
  "last_trigger_condition" TEXT,
  "last_triggered_at" DATETIME,
  FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE CASCADE
);
```

```sql
CREATE INDEX "ix_telegram_alerts_user_id" ON "telegram_alerts" ("user_id");
```

---

## "telegram_broadcast_logs"

```sql
CREATE TABLE "telegram_broadcast_logs" (
  "id" INTEGER PRIMARY KEY,
  "message" TEXT NOT NULL,
  "sent_by" VARCHAR(255) NOT NULL,
  "success_count" INTEGER,
  "total_count" INTEGER
);
```

---

## "telegram_command_audits"

```sql
CREATE TABLE "telegram_command_audits" (
  "id" INTEGER PRIMARY KEY,
  "telegram_user_id" VARCHAR(255) NOT NULL,
  "command" VARCHAR(255) NOT NULL,
  "full_message" TEXT,
  "is_registered_user" BOOLEAN,
  "user_email" VARCHAR(255),
  "success" BOOLEAN,
  "error_message" TEXT,
  "response_time_ms" INTEGER,
  "created_at" DATETIME
);
```

```sql
CREATE INDEX "ix_telegram_command_audits_created" ON "telegram_command_audits" ("created_at");
CREATE INDEX "ix_telegram_command_audits_command" ON "telegram_command_audits" ("command");
CREATE INDEX "ix_telegram_command_audits_success" ON "telegram_command_audits" ("success");
CREATE INDEX "ix_telegram_command_audits_telegram_user_id" ON "telegram_command_audits" ("telegram_user_id");
```

---

## "telegram_feedbacks"

```sql
CREATE TABLE "telegram_feedbacks" (
  "id" INTEGER PRIMARY KEY,
  "user_id" INTEGER NOT NULL,
  FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE CASCADE
);
```

---

## "telegram_schedules"

```sql
CREATE TABLE "telegram_schedules" (
  "id" INTEGER PRIMARY KEY,
  "user_id" INTEGER NOT NULL,
  FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE CASCADE
);
```

---

## "telegram_settings"

```sql
CREATE TABLE "telegram_settings" (
  "key" VARCHAR(100) PRIMARY KEY NOT NULL,
  "value" TEXT
);
```

---

## "telegram_verification_codes"

```sql
CREATE TABLE "telegram_verification_codes" (
  "id" INTEGER PRIMARY KEY,
  "user_id" INTEGER NOT NULL,
  "code" VARCHAR(32) NOT NULL,
  "sent_time" INTEGER NOT NULL,
  FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE CASCADE
);
```

```sql
CREATE INDEX "ix_telegram_verification_codes_user_id" ON "telegram_verification_codes" ("user_id");
```

---

## "trading_bot_instances"

```sql
CREATE TABLE "trading_bot_instances" (
  "id" VARCHAR(255) PRIMARY KEY NOT NULL,
  "type" VARCHAR(20) NOT NULL,
  "config_file" VARCHAR(255),
  "status" VARCHAR(20) NOT NULL,
  "started_at" DATETIME,
  "last_heartbeat" DATETIME,
  "error_count" INTEGER,
  "current_balance" NUMERIC(20,8),
  "total_pnl" NUMERIC(20,8),
  "extra_metadata" JSON,
  "created_at" DATETIME,
  "updated_at" DATETIME
);
```

```sql
CREATE INDEX "ix_trading_bot_instances_last_heartbeat" ON "trading_bot_instances" ("last_heartbeat");
CREATE INDEX "ix_trading_bot_instances_type" ON "trading_bot_instances" ("type");
CREATE INDEX "ix_trading_bot_instances_status" ON "trading_bot_instances" ("status");
```

---

## "trading_performance_metrics"

```sql
CREATE TABLE "trading_performance_metrics" (
  "id" VARCHAR(36) PRIMARY KEY,
  "bot_id" VARCHAR(255) NOT NULL,
  "trade_type" VARCHAR(10) NOT NULL,
  "symbol" VARCHAR(20),
  "interval" VARCHAR(10),
  "entry_logic_name" VARCHAR(100),
  "exit_logic_name" VARCHAR(100),
  "metrics" JSON NOT NULL,
  "calculated_at" DATETIME,
  "created_at" DATETIME,
  FOREIGN KEY ("bot_id") REFERENCES "trading_bot_instances" ("id") ON DELETE CASCADE
);
```

```sql
CREATE INDEX "ix_trading_performance_metrics_bot_id_calculated_at" ON "trading_performance_metrics" ("bot_id", "calculated_at");
CREATE INDEX "ix_trading_performance_metrics_symbol" ON "trading_performance_metrics" ("symbol");
CREATE INDEX "ix_trading_performance_metrics_calculated_at" ON "trading_performance_metrics" ("calculated_at");
CREATE INDEX "ix_trading_performance_metrics_bot_id" ON "trading_performance_metrics" ("bot_id");
```

---

## "trading_positions"

```sql
CREATE TABLE "trading_positions" (
  "id" VARCHAR(36) PRIMARY KEY,
  "bot_id" VARCHAR(255) NOT NULL,
  "trade_type" VARCHAR(10) NOT NULL,
  "symbol" VARCHAR(20) NOT NULL,
  "direction" VARCHAR(10) NOT NULL,
  "opened_at" DATETIME,
  "closed_at" DATETIME,
  "qty_open" NUMERIC(20,8) NOT NULL DEFAULT 0,
  "avg_price" NUMERIC(20,8),
  "realized_pnl" NUMERIC(20,8) DEFAULT 0,
  "status" VARCHAR(12) NOT NULL,
  "extra_metadata" JSON,
  FOREIGN KEY ("bot_id") REFERENCES "trading_bot_instances" ("id") ON DELETE CASCADE
);
```

```sql
CREATE INDEX "ix_trading_positions_bot_id_status" ON "trading_positions" ("bot_id", "status");
CREATE INDEX "ix_trading_positions_symbol" ON "trading_positions" ("symbol");
CREATE INDEX "ix_trading_positions_bot_id" ON "trading_positions" ("bot_id");
```

---

## "trading_trades"

```sql
CREATE TABLE "trading_trades" (
  "id" VARCHAR(36) PRIMARY KEY,
  "bot_id" VARCHAR(255) NOT NULL,
  "trade_type" VARCHAR(10) NOT NULL,
  "strategy_name" VARCHAR(100),
  "entry_logic_name" VARCHAR(100) NOT NULL,
  "exit_logic_name" VARCHAR(100) NOT NULL,
  "symbol" VARCHAR(20) NOT NULL,
  "interval" VARCHAR(10) NOT NULL,
  "entry_time" DATETIME,
  "exit_time" DATETIME,
  "buy_order_created" DATETIME,
  "buy_order_closed" DATETIME,
  "sell_order_created" DATETIME,
  "sell_order_closed" DATETIME,
  "entry_price" NUMERIC(20,8),
  "exit_price" NUMERIC(20,8),
  "entry_value" NUMERIC(20,8),
  "exit_value" NUMERIC(20,8),
  "size" NUMERIC(20,8),
  "direction" VARCHAR(10) NOT NULL,
  "commission" NUMERIC(20,8),
  "gross_pnl" NUMERIC(20,8),
  "net_pnl" NUMERIC(20,8),
  "pnl_percentage" NUMERIC(10,4),
  "exit_reason" VARCHAR(100),
  "status" VARCHAR(20) NOT NULL,
  "extra_metadata" JSON,
  "created_at" DATETIME,
  "updated_at" DATETIME,
  "position_id" VARCHAR(36),
  FOREIGN KEY ("position_id") REFERENCES "trading_positions" ("id") ON DELETE SET NULL,
  FOREIGN KEY ("bot_id") REFERENCES "trading_bot_instances" ("id") ON DELETE CASCADE
);
```

```sql
CREATE INDEX "ix_trading_trades_bot_id_symbol" ON "trading_trades" ("bot_id", "symbol");
CREATE INDEX "ix_trading_trades_bot_id_entry_time" ON "trading_trades" ("bot_id", "entry_time");
CREATE INDEX "ix_trading_trades_bot_id_status" ON "trading_trades" ("bot_id", "status");
CREATE INDEX "ix_trading_trades_strategy_name" ON "trading_trades" ("strategy_name");
CREATE INDEX "ix_trading_trades_trade_type" ON "trading_trades" ("trade_type");
CREATE INDEX "ix_trading_trades_symbol" ON "trading_trades" ("symbol");
CREATE INDEX "ix_trading_trades_bot_id" ON "trading_trades" ("bot_id");
CREATE INDEX "ix_trading_trades_entry_time" ON "trading_trades" ("entry_time");
CREATE INDEX "ix_trading_trades_status" ON "trading_trades" ("status");
```

---

## "users"

```sql
CREATE TABLE "users" (
  "id" INTEGER PRIMARY KEY,
  "email" VARCHAR(100),
  "role" VARCHAR(20) NOT NULL DEFAULT 'trader',
  "is_active" BOOLEAN DEFAULT TRUE,
  "telegram_user_id" VARCHAR(255),
  "created_at" DATETIME DEFAULT CURRENT_TIMESTAMP,
  "updated_at" DATETIME,
  "last_login" DATETIME
);
```

```sql
CREATE INDEX "ix_users_email" ON "users" ("email");
```

---

## "webui_audit_logs"

```sql
CREATE TABLE "webui_audit_logs" (
  "id" INTEGER PRIMARY KEY,
  "user_id" INTEGER NOT NULL,
  "action" VARCHAR(100) NOT NULL,
  "resource_type" VARCHAR(50),
  "resource_id" VARCHAR(100),
  "details" JSON,
  "ip_address" VARCHAR(45),
  "user_agent" VARCHAR(500),
  "created_at" DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY ("user_id") REFERENCES "users" ("id")
);
```

```sql
CREATE INDEX "ix_webui_audit_logs_action" ON "webui_audit_logs" ("action");
CREATE INDEX "ix_webui_audit_logs_user_id" ON "webui_audit_logs" ("user_id");
```

---

## "webui_performance_snapshots"

```sql
CREATE TABLE "webui_performance_snapshots" (
  "id" INTEGER PRIMARY KEY,
  "strategy_id" VARCHAR(100) NOT NULL,
  "timestamp" DATETIME DEFAULT CURRENT_TIMESTAMP,
  "pnl" JSON NOT NULL,
  "positions" JSON,
  "trades_count" INTEGER DEFAULT 0,
  "win_rate" JSON,
  "drawdown" JSON,
  "metrics" JSON
);
```

```sql
CREATE INDEX "ix_webui_performance_snapshots_strategy_id" ON "webui_performance_snapshots" ("strategy_id");
```

---

## "webui_strategy_templates"

```sql
CREATE TABLE "webui_strategy_templates" (
  "id" INTEGER PRIMARY KEY,
  "name" VARCHAR(100) NOT NULL,
  "description" TEXT,
  "template_data" JSON NOT NULL,
  "is_public" BOOLEAN DEFAULT FALSE,
  "created_by" INTEGER NOT NULL,
  "created_at" DATETIME DEFAULT CURRENT_TIMESTAMP,
  "updated_at" DATETIME,
  FOREIGN KEY ("created_by") REFERENCES "users" ("id")
);
```

```sql
CREATE INDEX "ix_webui_strategy_templates_created_by" ON "webui_strategy_templates" ("created_by");
```

---

## "webui_system_config"

```sql
CREATE TABLE "webui_system_config" (
  "id" INTEGER PRIMARY KEY,
  "key" VARCHAR(100) NOT NULL,
  "value" JSON NOT NULL,
  "description" TEXT,
  "created_at" DATETIME DEFAULT CURRENT_TIMESTAMP,
  "updated_at" DATETIME
);
```

---