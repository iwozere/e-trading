import psycopg2
from dotenv import dotenv_values
import pathlib

env_path = pathlib.Path('config/donotshare/.env')
values = dotenv_values(env_path)

password = values.get('POSTGRES_PASSWORD')
host = values.get('POSTGRES_HOST', 'localhost')
port = values.get('POSTGRES_PORT', 5432)

# Test connection with trading_admin user
print(f"Testing trading_admin user with password from .env...")
try:
    conn = psycopg2.connect(
        host=host,
        port=port,
        user='trading_admin',
        password=password,
        database='postgres'
    )
    print("✓ Successfully connected with trading_admin")
    conn.close()
except Exception as e:
    print(f"✗ Failed with trading_admin: {e}")

# Test connection with postgres user
print(f"\nTesting postgres user with password from .env...")
try:
    conn = psycopg2.connect(
        host=host,
        port=port,
        user='postgres',
        password=password,
        database='postgres'
    )
    print("✓ Successfully connected with postgres")
    conn.close()
except Exception as e:
    print(f"✗ Failed with postgres: {e}")
