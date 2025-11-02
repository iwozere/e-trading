# Repository Tests Setup Guide

## Prerequisites

The repository-layer tests require a **local PostgreSQL server** (version 12+) to create and drop temporary test databases.

## Setup Options

### Option 1: Local PostgreSQL Installation (Recommended if no Docker)

1. **Install PostgreSQL** (if not already installed):
   - Windows: Download from https://www.postgresql.org/download/windows/
   - During installation, remember the password you set for the `postgres` superuser

2. **Update your `.env` file** at `config/donotshare/.env`:
   ```bash
   # Use the postgres superuser password you set during installation
   POSTGRES_PASSWORD='your_postgres_password_here'
   POSTGRES_USER="postgres"
   POSTGRES_DATABASE="trading"  # Not used for tests
   POSTGRES_HOST="localhost"
   POSTGRES_PORT=5432
   ```

3. **Verify PostgreSQL is running**:
   ```powershell
   # Test connection
   psql -U postgres -h localhost -c "SELECT version();"
   # Enter your password when prompted
   ```

4. **Run the repo tests**:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pytest -q src/data/db/tests/repos
   ```

### Option 2: Docker Compose (If you have Docker)

1. Update `.env` file and start Postgres:
   ```powershell
   docker compose up -d postgres
   ```

2. Run tests as above

## How the Tests Work

- Tests auto-load credentials from `config/donotshare/.env`
- A temporary test database (`e_trading_test_XXXXXXXX`) is created
- Alembic runs migrations to set up the schema
- Tests run in isolated transactions
- Test database is dropped after all tests complete

## Troubleshooting

### "password authentication failed for user postgres"
- Your POSTGRES_PASSWORD in `.env` doesn't match your local PostgreSQL installation
- Update the password in `.env` to match what you set during PostgreSQL installation

### "connection refused"
- PostgreSQL service is not running
- Windows: Check Services (services.msc) for "postgresql-x64-XX" service
- Or verify with: `pg_isready -h localhost -p 5432`

### Tests are skipped
- No valid database connection URL found
- Ensure POSTGRES_PASSWORD is set in `.env`

### Port 5432 already in use
- Another PostgreSQL instance is running
- Either use that instance (update `.env` credentials) or stop it and use your own

## Current Configuration

The test harness tries to connect using (in order):
1. `PG_ADMIN_URL` environment variable (if set)
2. `postgres` user with `POSTGRES_PASSWORD` from `.env`
3. `POSTGRES_USER` with `POSTGRES_PASSWORD` from `.env`

**For most users:** Just set `POSTGRES_PASSWORD` in `.env` to your local PostgreSQL password.
