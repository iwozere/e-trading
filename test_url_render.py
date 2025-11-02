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
u = u.set(database="e_trading_test_abc123")

# Different ways to get the URL string:
print(f"\nstr(u): {str(u)}")
print(f"\nu.render_as_string(hide_password=False): {u.render_as_string(hide_password=False)}")

# Try connecting with the correct method
from sqlalchemy import create_engine, text
try:
    test_url = u.render_as_string(hide_password=False)
    engine = create_engine(test_url, pool_pre_ping=True, future=True)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("\n✓ Successfully connected!")
except Exception as e:
    print(f"\n✗ Failed: {e}")
