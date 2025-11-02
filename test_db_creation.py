import psycopg2
from dotenv import dotenv_values
import pathlib

env_path = pathlib.Path('config/donotshare/.env')
values = dotenv_values(env_path)

password = values.get('POSTGRES_PASSWORD')

# Try connecting to a test database (create it first if it doesn't exist)
test_db = 'test_conn_check'

# Create test database
print("Creating test database...")
try:
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        user='postgres',
        password=password,
        database='postgres'
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"DROP DATABASE IF EXISTS {test_db}")
    cur.execute(f"CREATE DATABASE {test_db}")
    cur.close()
    conn.close()
    print(f"✓ Created database {test_db}")
except Exception as e:
    print(f"✗ Failed to create database: {e}")
    exit(1)

# Try connecting to the test database
print(f"\nTesting connection to {test_db}...")
try:
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        user='postgres',
        password=password,
        database=test_db
    )
    print("✓ Successfully connected to test database")
    conn.close()
except Exception as e:
    print(f"✗ Failed to connect to test database: {e}")

# Clean up
print(f"\nCleaning up...")
try:
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        user='postgres',
        password=password,
        database='postgres'
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"DROP DATABASE IF EXISTS {test_db}")
    cur.close()
    conn.close()
    print(f"✓ Dropped database {test_db}")
except Exception as e:
    print(f"✗ Failed to cleanup: {e}")
