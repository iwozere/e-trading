# PostgreSQL Migration Guide

This guide helps you complete the switch from SQLite to PostgreSQL for your trading system.

## Prerequisites

1. **PostgreSQL Server**: Make sure PostgreSQL is installed and running
2. **Database Created**: Create a database named `trading_db`
3. **User Setup**: Ensure user `trading_admin` has access to the database
4. **Environment Variables**: Set `POSTGRES_PASSWORD` in your `.env` file

## Migration Steps

### 1. Install PostgreSQL Driver

The PostgreSQL driver has been added to `requirements.txt`. Install it:

```bash
pip install -r requirements.txt
```

### 2. Verify Configuration

Check that your `config/donotshare/donotshare.py` has the correct PostgreSQL settings:

```python
POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432
POSTGRES_USER = "trading_admin"
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DATABASE = "trading_db"
```

### 3. Set Environment Variable

Add to your `config/donotshare/.env` file:

```bash
POSTGRES_PASSWORD=your_postgres_password_here
```

### 4. Initialize Database Schema

Run the initialization script to create all tables:

```bash
python scripts/init_postgres.py
```

### 5. Verify Migration

Run the verification script to ensure everything is working:

```bash
python scripts/verify_migration.py
```

## Configuration Details

### Database Connection

The system now uses this connection string format:
```
postgresql+psycopg2://trading_admin:password@localhost:5432/trading_db
```

### Environment Override

You can still override the database URL using the `DB_URL` environment variable:

```bash
export DB_URL="postgresql+psycopg2://user:pass@host:port/dbname"
```

## Troubleshooting

### Connection Issues

1. **Check PostgreSQL is running**:
   ```bash
   # On Windows
   net start postgresql-x64-14
   
   # On Linux/Mac
   sudo systemctl status postgresql
   ```

2. **Verify database exists**:
   ```sql
   psql -U postgres -c "\l"
   ```

3. **Check user permissions**:
   ```sql
   psql -U postgres -c "\du"
   ```

### Common Errors

- **"database does not exist"**: Create the `trading_db` database
- **"authentication failed"**: Check username/password in `.env`
- **"connection refused"**: Ensure PostgreSQL is running on the correct port

### Creating Database and User

If you need to set up the database and user:

```sql
-- Connect as postgres superuser
psql -U postgres

-- Create database
CREATE DATABASE trading_db;

-- Create user
CREATE USER trading_admin WITH PASSWORD 'your_password_here';

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE trading_db TO trading_admin;
GRANT ALL ON SCHEMA public TO trading_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_admin;

-- For future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO trading_admin;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO trading_admin;
```

## Performance Considerations

PostgreSQL offers several advantages over SQLite:

1. **Concurrent Access**: Multiple processes can read/write simultaneously
2. **Better Performance**: Optimized for larger datasets
3. **Advanced Features**: Full-text search, JSON operations, etc.
4. **Scalability**: Can handle much larger databases

## Rollback Plan

If you need to rollback to SQLite temporarily:

1. Set environment variable:
   ```bash
   export DB_URL="sqlite:///db/trading.db"
   ```

2. Or modify `donotshare.py` to use the old SQLite path

## Next Steps

After successful migration:

1. **Test all functionality** to ensure everything works
2. **Monitor performance** - PostgreSQL may be faster for complex queries
3. **Set up backups** for your PostgreSQL database
4. **Consider connection pooling** for high-traffic scenarios

## Support

If you encounter issues:

1. Check the logs for detailed error messages
2. Verify all configuration settings
3. Test database connectivity manually using `psql`
4. Ensure all required tables were created properly