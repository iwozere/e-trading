-- Fix all table permissions and ownership for trading_admin
-- Run this after restoring a database backup

-- Grant privileges on all tables in public schema
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_admin;

-- Grant privileges on all sequences in public schema
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_admin;

-- Change owner of all tables to trading_admin
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'public'
    LOOP
        EXECUTE 'ALTER TABLE ' || quote_ident(r.tablename) || ' OWNER TO trading_admin';
    END LOOP;
END $$;

-- Change owner of all sequences to trading_admin
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN SELECT sequence_name FROM information_schema.sequences WHERE sequence_schema = 'public'
    LOOP
        EXECUTE 'ALTER SEQUENCE ' || quote_ident(r.sequence_name) || ' OWNER TO trading_admin';
    END LOOP;
END $$;

-- Grant schema usage
GRANT USAGE ON SCHEMA public TO trading_admin;
GRANT CREATE ON SCHEMA public TO trading_admin;

-- Verify the changes
SELECT
    tablename,
    tableowner
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY tablename;
