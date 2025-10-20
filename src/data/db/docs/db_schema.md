# Database Schema Documentation

## Schema Information

**Schema:** `public`  
**Owner:** `pg_database_owner`  
**Description:** standard public schema

---

## Tables

### User Management

#### `usr_users`
User accounts and authentication

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | int4 | PRIMARY KEY, GENERATED ALWAYS AS IDENTITY | User ID |
| email | varchar(100) | | User email address |
| role | varchar(20) | NOT NULL, DEFAULT 'trader' | User role |
| is_active | bool | DEFAULT true | Account active status |
| created_at | timestamptz | DEFAULT now() | Account creation timestamp |
| updated_at | timestamp | | Last update timestamp |
| last_login | timestamp | | Last login timestamp |

**Indexes:**
- `ix_users_email` on (email)

#### `usr_auth_identities`
OAuth and external authentication identities

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | int4 | PRIMARY KEY, GENERATED ALWAYS AS IDENTITY | Identity ID |
| user_id | int4 | NOT NULL, FK → usr_users(id) ON DELETE CASCADE | Associated user |
| provider | varchar(32) | NOT NULL | Auth provider name |
| external_id | varchar(255) | NOT NULL | External provider user ID |
| metadata | jsonb | | Additional provider data |
| created_at | timestamptz | DEFAULT now() | Creation timestamp |

**Indexes:**
- `ix_auth_identities_provider` on (provider)
- `ix_auth_identities_user_id` on (user_id)

#### `usr_verification_codes`
User verification codes for authentication

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | int4 | PRIMARY KEY, GENERATED ALWAYS AS IDENTITY | Code ID |
| user_id | int4 | NOT NULL, FK → usr_users(id) ON DELETE CASCADE | Associated user |
| code | varchar(32) | NOT NULL | Verification code |
| sent_time | int4 | NOT NULL | Unix timestamp when sent |
| provider | varchar(20) | DEFAULT 'telegram' | Delivery provider |
| created_at | timestamptz | DEFAULT now() | Creation timestamp |

**Indexes:**
- `ix_verification_codes_user_id` on (user_id)

---

### Trading System

#### `trading_bots`
Trading bot configurations and status

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | int4 | PRIMARY KEY, GENERATED ALWAYS AS IDENTITY | Bot ID |
| user_id | int4 | NOT NULL, FK → usr_users(id) ON DELETE CASCADE | Bot owner |
| type | varchar(20) | NOT NULL | Bot type |
| status | varchar(20) | NOT NULL | Current status |
| config | jsonb | NOT NULL | Bot configuration |
| description | text | | Bot description (defined by trader) |
| started_at | timestamp | | Start time |
| last_heartbeat | timestamp | | Last heartbeat timestamp |
| error_count | int4 | | Number of errors encountered |
| current_balance | numeric(20,8) | | Current balance |
| total_pnl | numeric(20,8) | | Total profit/loss |
| extra_metadata | jsonb | | Additional metadata |
| created_at | timestamptz | DEFAULT now() | Creation timestamp |
| updated_at | timestamp | | Last update timestamp |

**Triggers:**
- `update_bots_updated_at` - Updates updated_at on row modification

#### `trading_positions`
Open and closed trading positions

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | int4 | PRIMARY KEY, GENERATED ALWAYS AS IDENTITY | Position ID |
| bot_id | int4 | NOT NULL, FK → trading_bots(id) ON DELETE CASCADE | Associated bot |
| trade_type | varchar(10) | NOT NULL | Trade type |
| symbol | varchar(20) | NOT NULL | Trading symbol |
| direction | varchar(10) | NOT NULL | Position direction (LONG/SHORT) |
| opened_at | timestamp | | Position open time |
| closed_at | timestamp | | Position close time |
| qty_open | numeric(20,8) | NOT NULL, DEFAULT 0 | Open quantity |
| avg_price | numeric(20,8) | | Average entry price |
| realized_pnl | numeric(20,8) | DEFAULT 0 | Realized profit/loss |
| status | varchar(12) | NOT NULL | Position status |
| extra_metadata | jsonb | | Additional metadata |

**Indexes:**
- `ix_trading_positions_bot_id` on (bot_id)
- `ix_trading_positions_symbol` on (symbol)

#### `trading_trades`
Individual trade records

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | int4 | PRIMARY KEY, GENERATED ALWAYS AS IDENTITY | Trade ID |
| bot_id | int4 | NOT NULL, FK → trading_bots(id) ON DELETE CASCADE | Associated bot |
| position_id | int4 | FK → trading_positions(id) ON DELETE SET NULL | Associated position |
| trade_type | varchar(10) | NOT NULL | Trade type |
| strategy_name | varchar(100) | | Strategy name |
| entry_logic_name | varchar(100) | NOT NULL | Entry logic identifier |
| exit_logic_name | varchar(100) | NOT NULL | Exit logic identifier |
| symbol | varchar(20) | NOT NULL | Trading symbol |
| interval | varchar(10) | NOT NULL | Time interval |
| direction | varchar(10) | NOT NULL | Trade direction |
| entry_time | timestamp | | Entry timestamp |
| exit_time | timestamp | | Exit timestamp |
| buy_order_created | timestamp | | Buy order creation time |
| buy_order_closed | timestamp | | Buy order close time |
| sell_order_created | timestamp | | Sell order creation time |
| sell_order_closed | timestamp | | Sell order close time |
| entry_price | numeric(20,8) | | Entry price |
| exit_price | numeric(20,8) | | Exit price |
| entry_value | numeric(20,8) | | Entry value |
| exit_value | numeric(20,8) | | Exit value |
| size | numeric(20,8) | | Trade size |
| commission | numeric(20,8) | | Trading commission |
| gross_pnl | numeric(20,8) | | Gross profit/loss |
| net_pnl | numeric(20,8) | | Net profit/loss |
| pnl_percentage | numeric(10,4) | | PnL percentage |
| exit_reason | varchar(100) | | Reason for exit |
| status | varchar(20) | NOT NULL | Trade status |
| extra_metadata | jsonb | | Additional metadata |
| created_at | timestamptz | DEFAULT now() | Creation timestamp |
| updated_at | timestamp | | Last update timestamp |

**Indexes:**
- `ix_trading_trades_bot_id` on (bot_id)
- `ix_trading_trades_entry_time` on (entry_time)
- `ix_trading_trades_status` on (status)
- `ix_trading_trades_strategy_name` on (strategy_name)
- `ix_trading_trades_symbol` on (symbol)
- `ix_trading_trades_trade_type` on (trade_type)

#### `trading_performance_metrics`
Performance metrics for trading strategies

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | int4 | PRIMARY KEY, GENERATED ALWAYS AS IDENTITY | Metric ID |
| bot_id | int4 | NOT NULL, FK → trading_bots(id) ON DELETE CASCADE | Associated bot |
| trade_type | varchar(10) | NOT NULL | Trade type |
| symbol | varchar(20) | | Trading symbol |
| interval | varchar(10) | | Time interval |
| entry_logic_name | varchar(100) | | Entry logic identifier |
| exit_logic_name | varchar(100) | | Exit logic identifier |
| metrics | jsonb | NOT NULL | Performance metrics data |
| calculated_at | timestamp | | Calculation timestamp |
| created_at | timestamptz | DEFAULT now() | Creation timestamp |

**Indexes:**
- `ix_trading_performance_metrics_bot_id` on (bot_id)
- `ix_trading_performance_metrics_calculated_at` on (calculated_at)
- `ix_trading_performance_metrics_symbol` on (symbol)

---

### Messaging System

#### `msg_messages`
Notification messages queue with priority and retry support

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | bigserial | PRIMARY KEY | Message ID |
| message_type | varchar(50) | NOT NULL | Type of message |
| priority | varchar(20) | NOT NULL, DEFAULT 'NORMAL', CHECK (LOW/NORMAL/HIGH/CRITICAL) | Message priority |
| channels | _text | NOT NULL | Delivery channels array |
| recipient_id | varchar(100) | | Recipient identifier |
| template_name | varchar(100) | | Template name |
| content | jsonb | NOT NULL | Message content |
| metadata | jsonb | | Additional metadata |
| status | varchar(20) | NOT NULL, DEFAULT 'PENDING', CHECK (PENDING/PROCESSING/DELIVERED/FAILED/CANCELLED) | Message status |
| scheduled_for | timestamptz | DEFAULT now() | Scheduled delivery time |
| retry_count | int4 | NOT NULL, DEFAULT 0, CHECK (≥ 0) | Current retry count |
| max_retries | int4 | NOT NULL, DEFAULT 3, CHECK (≥ 0) | Maximum retry attempts |
| last_error | text | | Last error message |
| processed_at | timestamptz | | Processing timestamp |
| created_at | timestamptz | DEFAULT now() | Creation timestamp |

**Indexes:**
- `idx_msg_messages_created` on (created_at)
- `idx_msg_messages_pending_priority` on (status, scheduled_for, priority) WHERE status = 'PENDING'
- `idx_msg_messages_priority` on (priority)
- `idx_msg_messages_recipient` on (recipient_id)
- `idx_msg_messages_retry_eligible` on (status, retry_count, max_retries, processed_at) WHERE status = 'FAILED'
- `idx_msg_messages_scheduled` on (scheduled_for)
- `idx_msg_messages_status` on (status)
- `idx_msg_messages_type` on (message_type)

#### `msg_delivery_status`
Per-channel delivery status tracking for each message

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | bigserial | PRIMARY KEY | Status ID |
| message_id | int8 | NOT NULL, FK → msg_messages(id) ON DELETE CASCADE | Associated message |
| channel | varchar(50) | NOT NULL | Delivery channel |
| status | varchar(20) | NOT NULL, CHECK (PENDING/SENT/DELIVERED/FAILED/BOUNCED) | Delivery status |
| delivered_at | timestamptz | | Delivery timestamp |
| response_time_ms | int4 | CHECK (≥ 0) | Response time in milliseconds |
| error_message | text | | Error message if failed |
| external_id | varchar(255) | | External provider message ID |
| created_at | timestamptz | DEFAULT now() | Creation timestamp |

**Indexes:**
- `idx_delivery_status_channel` on (channel)
- `idx_delivery_status_composite` on (message_id, channel, status)
- `idx_delivery_status_delivered` on (delivered_at)
- `idx_delivery_status_message` on (message_id)
- `idx_delivery_status_status` on (status)

#### `msg_channel_configs`
Channel plugin configuration and settings

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | bigserial | PRIMARY KEY | Config ID |
| channel | varchar(50) | NOT NULL, UNIQUE | Channel name |
| enabled | bool | NOT NULL, DEFAULT true | Channel enabled status |
| config | jsonb | NOT NULL | Channel configuration |
| rate_limit_per_minute | int4 | NOT NULL, DEFAULT 60, CHECK (> 0) | Rate limit per minute |
| max_retries | int4 | NOT NULL, DEFAULT 3, CHECK (≥ 0) | Maximum retry attempts |
| timeout_seconds | int4 | NOT NULL, DEFAULT 30, CHECK (> 0) | Request timeout in seconds |
| created_at | timestamptz | DEFAULT now() | Creation timestamp |
| updated_at | timestamptz | DEFAULT now() | Last update timestamp |

**Indexes:**
- `idx_channel_configs_enabled` on (enabled)
- `idx_channel_configs_updated` on (updated_at)

**Triggers:**
- `update_msg_channel_configs_updated_at` - Updates updated_at on row modification

#### `msg_channel_health`
Health monitoring for notification channels

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | bigserial | PRIMARY KEY | Health record ID |
| channel | varchar(50) | NOT NULL, UNIQUE | Channel name |
| status | varchar(20) | NOT NULL, CHECK (HEALTHY/DEGRADED/DOWN) | Health status |
| last_success | timestamptz | | Last successful delivery |
| last_failure | timestamptz | | Last failed delivery |
| failure_count | int4 | NOT NULL, DEFAULT 0, CHECK (≥ 0) | Consecutive failure count |
| avg_response_time_ms | int4 | CHECK (≥ 0) | Average response time |
| error_message | text | | Latest error message |
| checked_at | timestamptz | DEFAULT now() | Last health check time |

**Indexes:**
- `idx_channel_health_checked` on (checked_at)
- `idx_channel_health_status` on (status)

#### `msg_rate_limits`
Per-user rate limiting configuration and state

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | bigserial | PRIMARY KEY | Rate limit ID |
| user_id | varchar(100) | NOT NULL | User identifier |
| channel | varchar(50) | NOT NULL | Channel name |
| tokens | int4 | NOT NULL, CHECK (≥ 0) | Current token count |
| last_refill | timestamptz | DEFAULT now() | Last token refill time |
| max_tokens | int4 | NOT NULL, CHECK (> 0) | Maximum token capacity |
| refill_rate | int4 | NOT NULL, CHECK (> 0) | Token refill rate |
| created_at | timestamptz | DEFAULT now() | Creation timestamp |

**Constraints:**
- `unique_user_channel_rate_limit` UNIQUE (user_id, channel)

**Indexes:**
- `idx_rate_limits_channel` on (channel)
- `idx_rate_limits_last_refill` on (last_refill)
- `idx_rate_limits_user` on (user_id)

---

### Job Scheduling

#### `job_schedules`
Scheduled job configurations

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | int4 | PRIMARY KEY, GENERATED ALWAYS AS IDENTITY | Schedule ID |
| user_id | int4 | NOT NULL | Job owner |
| name | varchar(255) | NOT NULL | Schedule name |
| job_type | varchar(50) | NOT NULL | Type of job |
| target | varchar(255) | NOT NULL | Job target |
| task_params | jsonb | NOT NULL, DEFAULT '{}' | Job parameters |
| cron | varchar(100) | NOT NULL | Cron expression |
| enabled | bool | NOT NULL, DEFAULT true | Schedule enabled status |
| next_run_at | timestamptz | | Next scheduled run time |
| created_at | timestamptz | DEFAULT now() | Creation timestamp |
| updated_at | timestamptz | DEFAULT now() | Last update timestamp |

#### `job_schedule_runs`
Job execution history and results

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | int4 | PRIMARY KEY, GENERATED ALWAYS AS IDENTITY | Run ID |
| job_type | text | NOT NULL | Type of job |
| job_id | int4 | FK → job_schedules(id) ON DELETE CASCADE | Associated schedule |
| user_id | int8 | | Job executor |
| status | text | | Execution status |
| scheduled_for | timestamptz | | Scheduled execution time |
| enqueued_at | timestamptz | DEFAULT now() | Queue entry time |
| started_at | timestamptz | | Execution start time |
| finished_at | timestamptz | | Execution finish time |
| job_snapshot | jsonb | | Job state snapshot |
| result | jsonb | | Execution result |
| error | text | | Error message if failed |

**Constraints:**
- `ux_runs_job_scheduled_for` UNIQUE (job_type, job_id, scheduled_for)

---

### Telegram Integration

#### `telegram_broadcast_logs`
Telegram broadcast message logs

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | int4 | PRIMARY KEY, GENERATED ALWAYS AS IDENTITY | Log ID |
| message | text | NOT NULL | Broadcast message content |
| sent_by | varchar(255) | NOT NULL | Sender identifier |
| success_count | int4 | | Successful deliveries |
| total_count | int4 | | Total recipients |
| created_at | timestamptz | DEFAULT now() | Broadcast timestamp |

#### `telegram_command_audits`
Telegram command execution audit log

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | int4 | PRIMARY KEY, GENERATED ALWAYS AS IDENTITY | Audit ID |
| telegram_user_id | varchar(255) | NOT NULL | Telegram user ID |
| command | varchar(255) | NOT NULL | Executed command |
| full_message | text | | Complete message text |
| is_registered_user | bool | | User registration status |
| user_email | varchar(255) | | Associated email |
| success | bool | | Command success status |
| error_message | text | | Error message if failed |
| response_time_ms | int4 | | Response time in milliseconds |
| created_at | timestamptz | DEFAULT now() | Execution timestamp |

**Indexes:**
- `ix_telegram_command_audits_command` on (command)
- `ix_telegram_command_audits_created` on (created_at)
- `ix_telegram_command_audits_success` on (success)
- `ix_telegram_command_audits_telegram_user_id` on (telegram_user_id)

#### `telegram_feedbacks`
User feedback submitted via Telegram

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | int4 | PRIMARY KEY, GENERATED ALWAYS AS IDENTITY | Feedback ID |
| user_id | int4 | NOT NULL, FK → usr_users(id) ON DELETE CASCADE | User who submitted feedback |
| type | varchar(50) | | Feedback type/category |
| message | text | | Feedback message |
| status | varchar(20) | | Processing status |
| created_at | timestamptz | DEFAULT now() | Submission timestamp |

#### `telegram_settings`
Telegram bot configuration settings

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| key | varchar(100) | PRIMARY KEY | Setting key |
| value | text | | Setting value |

---

### Web UI

#### `webui_audit_logs`
User action audit logs

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | int4 | PRIMARY KEY, GENERATED ALWAYS AS IDENTITY | Log ID |
| user_id | int4 | NOT NULL, FK → usr_users(id) | User who performed action |
| action | varchar(100) | NOT NULL | Action performed |
| resource_type | varchar(50) | | Type of resource affected |
| resource_id | varchar(100) | | ID of resource affected |
| details | jsonb | | Additional details |
| ip_address | varchar(45) | | User IP address |
| user_agent | varchar(500) | | User agent string |
| created_at | timestamptz | DEFAULT now() | Action timestamp |

**Indexes:**
- `ix_webui_audit_logs_action` on (action)
- `ix_webui_audit_logs_user_id` on (user_id)

#### `webui_performance_snapshots`
Strategy performance snapshots

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | int4 | PRIMARY KEY, GENERATED ALWAYS AS IDENTITY | Snapshot ID |
| strategy_id | varchar(100) | NOT NULL | Strategy identifier |
| timestamp | timestamptz | DEFAULT now() | Snapshot timestamp |
| pnl | jsonb | NOT NULL | Profit/loss data |
| positions | jsonb | | Position data |
| trades_count | int4 | DEFAULT 0 | Number of trades |
| win_rate | jsonb | | Win rate statistics |
| drawdown | jsonb | | Drawdown data |
| metrics | jsonb | | Additional metrics |

**Indexes:**
- `ix_webui_performance_snapshots_strategy_id` on (strategy_id)

#### `webui_strategy_templates`
Reusable strategy templates

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | int4 | PRIMARY KEY, GENERATED ALWAYS AS IDENTITY | Template ID |
| name | varchar(100) | NOT NULL | Template name |
| description | text | | Template description |
| template_data | jsonb | NOT NULL | Template configuration |
| is_public | bool | DEFAULT false | Public visibility |
| created_by | int4 | NOT NULL, FK → usr_users(id) | Template creator |
| created_at | timestamptz | DEFAULT now() | Creation timestamp |
| updated_at | timestamp | | Last update timestamp |

**Indexes:**
- `ix_webui_strategy_templates_created_by` on (created_by)

#### `webui_system_config`
System-wide configuration settings

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | int4 | PRIMARY KEY, GENERATED ALWAYS AS IDENTITY | Config ID |
| key | varchar(100) | NOT NULL | Configuration key |
| value | jsonb | NOT NULL | Configuration value |
| description | text | | Configuration description |
| created_at | timestamptz | DEFAULT now() | Creation timestamp |
| updated_at | timestamp | | Last update timestamp |

---

## Views

### `v_pending_messages`
Messages ready for processing, ordered by priority

Returns messages with status 'PENDING' and scheduled_for <= now(), ordered by priority (CRITICAL → HIGH → NORMAL → LOW) and scheduled time.

**Columns:** id, message_type, priority, channels, recipient_id, content, metadata, scheduled_for, retry_count, max_retries, created_at, priority_order

### `v_retry_eligible_messages`
Failed messages eligible for retry

Returns messages with status 'FAILED', retry_count < max_retries, and processed_at more than 5 minutes ago (or NULL).

**Columns:** id, message_type, priority, channels, recipient_id, content, retry_count, max_retries, last_error, processed_at, created_at

### `v_channel_health_summary`
Combined channel health and configuration view

Joins msg_channel_health with msg_channel_configs to provide comprehensive channel status.

**Columns:** channel, status, failure_count, avg_response_time_ms, last_success, last_failure, checked_at, config_enabled, rate_limit_per_minute, config_max_retries, timeout_seconds

### `v_delivery_stats`
Daily delivery statistics by channel

Aggregates delivery statistics by channel and day for the last 30 days.

**Columns:** channel, total_deliveries, successful_deliveries, failed_deliveries, bounced_deliveries, avg_response_time_ms, delivery_date

---

## Functions

### Message Queue Management

#### `enqueue_message()`
Enqueues a new message for delivery

**Parameters:**
- `p_message_type` varchar - Type of message
- `p_channels` text[] - Delivery channels
- `p_content` jsonb - Message content
- `p_priority` varchar - Priority level (default: 'NORMAL')
- `p_recipient_id` varchar - Recipient identifier (optional)
- `p_template_name` varchar - Template name (optional)
- `p_metadata` jsonb - Additional metadata (optional)
- `p_scheduled_for` timestamptz - Scheduled delivery time (default: now)
- `p_max_retries` integer - Maximum retry attempts (default: 3)

**Returns:** bigint (message_id)

#### `update_message_status()`
Updates the status of a message

**Parameters:**
- `p_message_id` bigint - Message ID
- `p_status` varchar - New status
- `p_error_message` text - Error message (optional)

**Returns:** boolean (success)

#### `record_delivery_status()`
Records delivery status for a specific channel

**Parameters:**
- `p_message_id` bigint - Message ID
- `p_channel` varchar - Delivery channel
- `p_status` varchar - Delivery status
- `p_response_time_ms` integer - Response time (optional)
- `p_error_message` text - Error message (optional)
- `p_external_id` varchar - External provider ID (optional)

**Returns:** bigint (delivery_id)

#### `cleanup_old_messages()`
Removes delivered messages older than specified days

**Parameters:**
- `p_days_to_keep` integer - Number of days to retain (default: 30)

**Returns:** integer (deleted_count)

#### `get_queue_statistics()`
Returns message queue statistics grouped by status and priority

**Returns:** TABLE(status varchar, priority varchar, count bigint)

### Channel Health Management

#### `update_channel_health()`
Updates or creates channel health record

**Parameters:**
- `p_channel` varchar - Channel name
- `p_status` varchar - Health status (HEALTHY/DEGRADED/DOWN)
- `p_response_time_ms` integer - Response time (optional)
- `p_error_message` text - Error message (optional)

**Returns:** void

### Utility Functions

#### `update_updated_at_column()`
Trigger function to automatically update updated_at timestamp

**Returns:** trigger

### UUID Functions

The schema includes standard UUID generation functions from the uuid-ossp extension:
- `uuid_generate_v1()` - Time-based UUID
- `uuid_generate_v1mc()` - Time-based UUID with MAC address
- `uuid_generate_v3(namespace, name)` - MD5-based UUID
- `uuid_generate_v4()` - Random UUID
- `uuid_generate_v5(namespace, name)` - SHA1-based UUID
- `uuid_nil()` - Nil UUID
- `uuid_ns_dns()` - DNS namespace UUID
- `uuid_ns_oid()` - OID namespace UUID
- `uuid_ns_url()` - URL namespace UUID
- `uuid_ns_x500()` - X.500 namespace UUID

---

## Sequences

The schema uses the following sequences for auto-incrementing IDs:

- `job_schedule_runs_id_seq` (1 to 2,147,483,647)
- `job_schedules_id_seq` (1 to 2,147,483,647)
- `msg_channel_configs_id_seq` (1 to 9,223,372,036,854,775,807)
- `msg_channel_health_id_seq` (1 to 9,223,372,036,854,775,807)
- `msg_delivery_status_id_seq` (1 to 9,223,372,036,854,775,807)
- `msg_messages_id_seq` (1 to 9,223,372,036,854,775,807)
- `msg_rate_limits_id_seq` (1 to 9,223,372,036,854,775,807)
- `telegram_broadcast_logs_id_seq1` (1 to 2,147,483,647)
- `telegram_command_audits_id_seq1` (1 to 2,147,483,647)
- `telegram_feedbacks_id_seq1` (1 to 2,147,483,647)
- `trading_bots_id_seq` (1 to 2,147,483,647)
- `trading_performance_metrics_id_seq` (1 to 2,147,483,647)
- `trading_positions_id_seq` (1 to 2,147,483,647)
- `trading_trades_id_seq` (1 to 2,147,483,647)
- `usr_auth_identities_id_seq` (1 to 2,147,483,647)
- `usr_users_id_seq` (1 to 2,147,483,647)
- `usr_verification_codes_id_seq` (1 to 2,147,483,647)
- `webui_audit_logs_id_seq1` (1 to 2,147,483,647)
- `webui_performance_snapshots_id_seq1` (1 to 2,147,483,647)
- `webui_strategy_templates_id_seq1` (1 to 2,147,483,647)
- `webui_system_config_id_seq1` (1 to 2,147,483,647)

---

## Key Relationships

### User System
- `usr_users` ← `usr_auth_identities` (one-to-many)
- `usr_users` ← `usr_verification_codes` (one-to-many)
- `usr_users` ← `telegram_feedbacks` (one-to-many)
- `usr_users` ← `webui_audit_logs` (one-to-many)
- `usr_users` ← `webui_strategy_templates` (one-to-many)
- `usr_users` ← `trading_bots` (one-to-many)

### Trading System
- `trading_bots` ← `trading_positions` (one-to-many)
- `trading_bots` ← `trading_trades` (one-to-many)
- `trading_bots` ← `trading_performance_metrics` (one-to-many)
- `trading_positions` ← `trading_trades` (one-to-many, optional)

### Messaging System
- `msg_messages` ← `msg_delivery_status` (one-to-many)

### Job Scheduling
- `job_schedules` ← `job_schedule_runs` (one-to-many)