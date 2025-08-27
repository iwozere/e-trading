# requirements.md

## 1. Scope

Design and implement a multi-user–ready trading database and data-access layer (DAL) that works on SQLite today and can be migrated to PostgreSQL. The system powers a Telegram bot and custom trading strategies (live and paper).

## 2. Users & Access

* Up to 20 registered users.
* All users can run screeners, alerts, reports, and trade.
* Admin role exists to approve users and manage system-wide settings.
* Authentication/authorization handled by the app (Telegram user id); DB enforces ownership via foreign keys.

## 3. Portfolios & Accounts

* Each user can have **multiple portfolios**.
* Each portfolio is bound to **one broker account** (e.g., Binance, IBKR). One portfolio → one broker.
* A user can have multiple broker accounts.

## 4. Strategies

* Each order/trade/position may be linked to the **strategy** that generated it (nullable for manual actions).
* Strategies have versioned configs (JSON), owned by a user but can be shared in the future.

## 5. Instruments

* Minimal reference for instruments/assets used for trading (symbol, exchange, asset type, currency, tick size).
* No OHLCV storage in DB at this stage (external data source).

## 6. Orders, Trades, Positions

* Store **open orders** and **executed trades** (fills).
* Positions are **materialized** in DB for fast reporting; they must reconcile from trades.
* Positions are unique per `(portfolio, instrument, strategy?)` and track qty, avg price, realized PnL.
* Support both long and short (qty sign indicates direction).

## 7. Risk Management

* Risk policies may be defined at **user**, **portfolio**, or **strategy** scope.
* Policies stored as JSON (SQLite TEXT, PostgreSQL JSONB), e.g., max risk per trade, max exposure per asset, daily loss limit, leverage caps.
* Optional policy enforcement takes place in the app; DB stores policies and **breach logs**.

## 8. Alerts, Schedules, Screeners

* Alerts, schedules (cron-like), and screeners stored per user with JSON config.
* Telegram delivery addresses stored in users table.

## 9. Auditing & Observability

* Audit trail for sensitive actions (e.g., order placement/cancel, policy changes).
* Minimal metrics tables for strategy runs if needed later.

## 10. Non-Functional Requirements

* **SQLite first**, **PostgreSQL-ready**: types and constraints chosen to be portable; avoid PG-only features for v1.
* Migrations with Alembic (or yoyo) from day one.
* Indices for hot paths: orders by status/time, trades by portfolio/time, positions by portfolio/instrument.
* Time stored as UTC ISO timestamps; app handles timezone.
* Idempotency for order/trade ingestion via unique broker IDs.
* PII/secrets (API keys) are **not** stored in plain DB; use external secret store.

---

# design.md

## 1. Entity Model (high-level)

**Users** (admin flag, approval) → **Accounts** (broker) → **Portfolios** → { **Orders**, **Trades**, **Positions** } linked to **Instruments** and optionally **Strategies**. Risk is orthogonal: **RiskPolicies** and **RiskBreaches** reference user/portfolio/strategy.

## 2. Tables (portable SQL)

> Note: Use INTEGER primary keys (auto-increment) for SQLite portability; add a `uid` TEXT (UUID) column for cross-system references.

### users

* id INTEGER PK
* uid TEXT UNIQUE
* telegram\_user\_id TEXT UNIQUE NOT NULL
* username TEXT
* is\_admin INTEGER NOT NULL DEFAULT 0
* is\_active INTEGER NOT NULL DEFAULT 1
* approved\_by\_user\_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL
* approved\_at TEXT NULL
* created\_at TEXT NOT NULL

**Indexes**: (telegram\_user\_id), (is\_active)

### brokers (reference)

* id INTEGER PK
* code TEXT UNIQUE NOT NULL  -- e.g., BINANCE, IBKR
* name TEXT NOT NULL

### accounts

* id INTEGER PK
* uid TEXT UNIQUE
* user\_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE
* broker\_id INTEGER NOT NULL REFERENCES brokers(id)
* name TEXT NOT NULL  -- user-facing label
* account\_uid TEXT NULL  -- broker-side ID
* base\_currency TEXT NOT NULL DEFAULT 'USD'
* created\_at TEXT NOT NULL

**Uniq**: (user\_id, name)

### portfolios

* id INTEGER PK
* uid TEXT UNIQUE
* user\_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE
* account\_id INTEGER NOT NULL REFERENCES accounts(id) ON DELETE CASCADE
* name TEXT NOT NULL
* description TEXT NULL
* is\_default INTEGER NOT NULL DEFAULT 0
* created\_at TEXT NOT NULL

**Uniq**: (user\_id, name)
**Idx**: (account\_id)

### strategies

* id INTEGER PK
* uid TEXT UNIQUE
* user\_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE
* name TEXT NOT NULL
* version TEXT NOT NULL DEFAULT '1.0'
* params\_json TEXT NOT NULL DEFAULT '{}'  -- JSONB in Postgres
* is\_active INTEGER NOT NULL DEFAULT 1
* created\_at TEXT NOT NULL

**Uniq**: (user\_id, name, version)

### instruments

* id INTEGER PK
* uid TEXT UNIQUE
* symbol TEXT NOT NULL  -- e.g., BTCUSDT, AAPL
* exchange TEXT NULL     -- e.g., BINANCE, NASDAQ
* asset\_type TEXT NOT NULL  -- ENUM-ish: 'CRYPTO','EQUITY','FUTURE','FOREX','ETF'
* currency TEXT NOT NULL
* tick\_size REAL NULL
* lot\_size REAL NULL

**Uniq**: (symbol, exchange)
**Idx**: (asset\_type)

### orders

* id INTEGER PK
* uid TEXT UNIQUE
* portfolio\_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE
* instrument\_id INTEGER NOT NULL REFERENCES instruments(id)
* strategy\_id INTEGER NULL REFERENCES strategies(id) ON DELETE SET NULL
* side TEXT NOT NULL  -- 'BUY'|'SELL'
* type TEXT NOT NULL  -- 'MARKET','LIMIT','STOP','STOP\_LIMIT','TRAILING'
* status TEXT NOT NULL  -- 'NEW','PARTIALLY\_FILLED','FILLED','CANCELED','REJECTED','EXPIRED'
* time\_in\_force TEXT NULL  -- 'GTC','IOC','FOK','DAY'
* quantity REAL NOT NULL
* price REAL NULL
* stop\_price REAL NULL
* client\_order\_id TEXT NULL
* broker\_order\_id TEXT NULL  -- unique per broker
* placed\_at TEXT NOT NULL
* updated\_at TEXT NOT NULL
* meta\_json TEXT NOT NULL DEFAULT '{}'  -- raw broker payloads, error codes

**Uniq (nullable aware)**: (broker\_order\_id) with partial uniqueness per broker via app layer; alternatively (portfolio\_id, broker\_order\_id)
**Idx**: (portfolio\_id, status), (instrument\_id), (placed\_at)

### trades (executions/fills)

* id INTEGER PK
* uid TEXT UNIQUE
* portfolio\_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE
* instrument\_id INTEGER NOT NULL REFERENCES instruments(id)
* strategy\_id INTEGER NULL REFERENCES strategies(id) ON DELETE SET NULL
* order\_id INTEGER NULL REFERENCES orders(id) ON DELETE SET NULL
* side TEXT NOT NULL  -- direction of execution
* quantity REAL NOT NULL
* price REAL NOT NULL
* fee REAL NOT NULL DEFAULT 0
* fee\_currency TEXT NULL
* exec\_time TEXT NOT NULL
* broker\_trade\_id TEXT NULL
* meta\_json TEXT NOT NULL DEFAULT '{}'

**Idx**: (portfolio\_id, exec\_time), (instrument\_id, exec\_time)
**Idempotency**: app should upsert by (portfolio\_id, broker\_trade\_id) when available.

### positions

* id INTEGER PK
* uid TEXT UNIQUE
* portfolio\_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE
* instrument\_id INTEGER NOT NULL REFERENCES instruments(id)
* strategy\_id INTEGER NULL REFERENCES strategies(id) ON DELETE SET NULL
* quantity REAL NOT NULL  -- signed; 0 means closed
* avg\_price REAL NOT NULL DEFAULT 0
* realized\_pnl REAL NOT NULL DEFAULT 0
* opened\_at TEXT NULL
* updated\_at TEXT NOT NULL

**Uniq**: (portfolio\_id, instrument\_id, strategy\_id)
**Idx**: (portfolio\_id), (instrument\_id)

> **Reconciliation**: positions are updated by the app when inserting trades (see DAL logic) and can be fully recomputed from trades for audit.

### risk\_policies

* id INTEGER PK
* uid TEXT UNIQUE
* scope\_type TEXT NOT NULL  -- 'USER'|'PORTFOLIO'|'STRATEGY'
* scope\_id INTEGER NOT NULL  -- FK to users/portfolios/strategies (validated in app)
* name TEXT NOT NULL
* config\_json TEXT NOT NULL  -- e.g., {"max\_risk\_per\_trade":0.01, "max\_daily\_loss":0.03}
* is\_active INTEGER NOT NULL DEFAULT 1
* created\_at TEXT NOT NULL

**Idx**: (scope\_type, scope\_id), (is\_active)

### risk\_breaches

* id INTEGER PK
* policy\_id INTEGER NOT NULL REFERENCES risk\_policies(id) ON DELETE CASCADE
* portfolio\_id INTEGER NULL REFERENCES portfolios(id) ON DELETE SET NULL
* instrument\_id INTEGER NULL REFERENCES instruments(id) ON DELETE SET NULL
* observed\_json TEXT NOT NULL  -- snapshot of metrics that caused breach
* action TEXT NOT NULL  -- 'BLOCKED','WARNED','REDUCED\_SIZE','CLOSED\_POSITION'
* created\_at TEXT NOT NULL

**Idx**: (policy\_id, created\_at)

### alerts

* id INTEGER PK
* uid TEXT UNIQUE
* user\_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE
* portfolio\_id INTEGER NULL REFERENCES portfolios(id) ON DELETE SET NULL
* name TEXT NOT NULL
* type TEXT NOT NULL  -- 'PRICE','INDICATOR','POSITION','RISK','SCHEDULE'
* config\_json TEXT NOT NULL
* is\_active INTEGER NOT NULL DEFAULT 1
* created\_at TEXT NOT NULL

**Idx**: (user\_id, is\_active)

### schedules

* id INTEGER PK
* uid TEXT UNIQUE
* user\_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE
* name TEXT NOT NULL
* cron TEXT NOT NULL  -- cron expression (UTC)
* config\_json TEXT NOT NULL DEFAULT '{}'
* is\_active INTEGER NOT NULL DEFAULT 1
* created\_at TEXT NOT NULL

### screeners

* id INTEGER PK
* uid TEXT UNIQUE
* user\_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE
* name TEXT NOT NULL
* config\_json TEXT NOT NULL
* is\_active INTEGER NOT NULL DEFAULT 1
* created\_at TEXT NOT NULL

### audit\_log

* id INTEGER PK
* user\_id INTEGER NULL REFERENCES users(id) ON DELETE SET NULL
* entity\_type TEXT NOT NULL  -- 'ORDER','TRADE','POSITION','POLICY','PORTFOLIO','STRATEGY'
* entity\_id INTEGER NULL
* action TEXT NOT NULL  -- 'CREATE','UPDATE','DELETE','SYNC','CANCEL','REJECT','BREACH'
* details\_json TEXT NOT NULL
* created\_at TEXT NOT NULL

## 3. DAL Design

* Use SQLAlchemy (Core or ORM) with an **abstract repository** pattern and **Unit of Work** to keep DB logic centralized.
* Idempotent upserts for orders/trades using broker IDs (client or broker-provided).
* Transactional position updates: insert trade → adjust position (qty, avg\_price, realized\_pnl) → audit log, all in one transaction.
* Provide DAL services per aggregate:

  * `OrderService`: place/cancel/update status; fetch open orders.
  * `TradeService`: ingest executions; ensure idempotency; position reconciliation hook.
  * `PositionService`: get snapshot by portfolio; recompute from trades (for audit).
  * `RiskService`: load policies, evaluate limits (in app), record breaches.
  * `PortfolioService`: CRUD portfolios/accounts; permission checks.

## 4. Position math (FIFO average cost)

* On buy: new\_avg = (old\_qty*avg + fill\_qty*price) / (old\_qty + fill\_qty)
* On sell that reduces/offsets qty: realized\_pnl += (sell\_price - avg\_price) \* sell\_qty \* sign
* When qty crosses zero → position closed: quantity=0, avg\_price=0; realized PnL persists.

## 5. Multi-user isolation

* Every query must filter by the caller's `user_id` (via portfolio/account ownership) unless admin.
* Unique constraints include `user_id` where relevant (e.g., portfolio names).

## 6. Migration to PostgreSQL

* Replace TEXT JSON with JSONB; add GIN indexes for frequently filtered JSON fields (later).
* Consider switching integer PKs to BIGINT or UUID PKs (keep `uid` for external references now).
* Add time zone type (TIMESTAMPTZ) and check constraints for enums.

## 7. Example DDL (portable)

```sql
-- enums via CHECK for portability
CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY,
  uid TEXT UNIQUE,
  portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
  instrument_id INTEGER NOT NULL REFERENCES instruments(id),
  strategy_id INTEGER REFERENCES strategies(id) ON DELETE SET NULL,
  side TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
  type TEXT NOT NULL CHECK (type IN ('MARKET','LIMIT','STOP','STOP_LIMIT','TRAILING')),
  status TEXT NOT NULL CHECK (status IN ('NEW','PARTIALLY_FILLED','FILLED','CANCELED','REJECTED','EXPIRED')),
  time_in_force TEXT CHECK (time_in_force IN ('GTC','IOC','FOK','DAY')),
  quantity REAL NOT NULL,
  price REAL,
  stop_price REAL,
  client_order_id TEXT,
  broker_order_id TEXT,
  placed_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  meta_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_orders_portfolio_status ON orders(portfolio_id, status);
```

---

# tasks.md

## Phase 0 — Prep

1. Create a new DB migration project (Alembic). Baseline the current `users` table.
2. Add UTC time helpers; standardize timestamps to ISO 8601.
3. Add a UUID helper in app (string UUIDv4) for `uid` columns.

## Phase 1 — Schema

1. Create reference tables: `brokers`, `instruments`.
2. Create ownership tables: `accounts`, `portfolios`, `strategies`.
3. Create trading tables: `orders`, `trades`, `positions`.
4. Create risk & aux: `risk_policies`, `risk_breaches`, `alerts`, `schedules`, `screeners`, `audit_log`.
5. Indices & unique constraints from design.md.
6. Seed `brokers` (BINANCE, IBKR, etc.).

## Phase 2 — DAL

1. Implement SQLAlchemy models (or table metadata) matching schema.
2. Implement repositories and services:

   * `PortfolioService`: CRUD with user ownership checks.
   * `OrderService`: place/cancel/update; map app enums ↔ DB strings.
   * `TradeService`: `ingest_fill(fill)` with idempotency on `(portfolio_id, broker_trade_id)`; within a transaction call `PositionService.apply_fill(...)`.
   * `PositionService`:

     * `apply_fill(position, side, qty, price, fee)` updates qty/avg/realized.
     * `recompute(portfolio_id)` derives positions from trades for audit.
   * `RiskService`:

     * `load_policies(scope)`; `evaluate(order/portfolio snapshot)` in app; `record_breach(...)`.
3. Add `AuditLogger.log(user_id, entity_type, action, details_json)`.

## Phase 3 — Bot integration

1. Extend Telegram handlers to include `portfolio_id` context for every command.
2. Commands:

   * `/orders [status]` — list open orders.
   * `/positions [portfolio]` — show positions.
   * `/pnl [period]` — compute from trades.
   * `/risk [portfolio]` — show active policies and latest breaches.
3. Add admin-only commands to approve users and manage brokers/accounts.

## Phase 4 — Strategy integration

1. Strategy runtime must provide `(user_id, portfolio_id, strategy_id)` for every action.
2. Implement an **order gateway** abstraction per broker; map broker events to `orders` & `trades`.
3. Ensure idempotent ingestion using `broker_order_id` and `broker_trade_id`.
4. Add a **paper-trading adapter** that writes to the same schema.

## Phase 5 — Testing & Migration

1. Factory fixtures for users, portfolios, instruments.
2. Property-based tests for `PositionService.apply_fill` (buys/sells/shorts, partial fills, crossing zero).
3. Load test: 20 users × 3 portfolios × 200 trades/day; verify indices.
4. Dry-run migration to PostgreSQL; convert TEXT JSON → JSONB; validate constraints and timestamp types.

## Phase 6 — Ops

1. Backups: periodic SQLite file backup (per hour/day) with retention.
2. Vacuum/Analyze schedule for SQLite; tune WAL mode for writers.
3. Logging: structured logs for order/trade ingestion with correlation IDs (uid/client\_order\_id).

---

## Appendix — Minimal create order/trade → position flow

1. Place order: create `orders` row (status NEW).
2. Broker fill event arrives → upsert into `trades` (idempotent).
3. Start transaction:

   * Update `orders.status` if filled/partial.
   * `positions` upsert on (portfolio\_id, instrument\_id, strategy\_id):

     * Update qty/avg/realized per formula.
     * Set `updated_at`.
   * Insert `audit_log` record.
4. Commit.
