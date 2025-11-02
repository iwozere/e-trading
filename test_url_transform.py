from dotenv import dotenv_values
import pathlib
from urllib.parse import quote_plus
from sqlalchemy.engine.url import make_url

env_path = pathlib.Path('config/donotshare/.env')
values = dotenv_values(env_path)

pwd = values.get('POSTGRES_PASSWORD')
encoded_pwd = quote_plus(pwd)

# Simulate what _get_admin_db_url_candidates does
admin_url = f"postgresql+psycopg2://postgres:{encoded_pwd}@localhost:5432/postgres"
print(f"Admin URL: {admin_url}")

# Simulate what _create_test_database does
u = make_url(admin_url)
print(f"\nAfter make_url():")
print(f"  user: {u.username}")
print(f"  password: {u.password}")
print(f"  database: {u.database}")

# Change the database name
u = u.set(database="e_trading_test_abc123")
print(f"\nAfter .set(database=...):")
print(f"  user: {u.username}")
print(f"  password: {u.password}")
print(f"  database: {u.database}")

# Convert back to string
test_url = str(u)
print(f"\nFinal test URL: {test_url}")

# Try connecting with it
from sqlalchemy import create_engine, text
try:
    engine = create_engine(test_url, pool_pre_ping=True, future=True)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("\n✓ Successfully connected with final URL!")
except Exception as e:
    print(f"\n✗ Failed to connect with final URL: {e}")
