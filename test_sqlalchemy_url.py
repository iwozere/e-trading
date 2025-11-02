from dotenv import dotenv_values
import pathlib
from urllib.parse import quote_plus

env_path = pathlib.Path('config/donotshare/.env')
values = dotenv_values(env_path)

pwd = values.get('POSTGRES_PASSWORD')
host = values.get('POSTGRES_HOST', 'localhost')
port = values.get('POSTGRES_PORT', '5432')

# Test what URL we're creating
encoded_pwd = quote_plus(pwd)
url = f"postgresql+psycopg2://postgres:{encoded_pwd}@{host}:{port}/postgres"

print(f"Raw password: {pwd}")
print(f"Encoded password: {encoded_pwd}")
print(f"Constructed URL: {url}")

# Now try connecting with SQLAlchemy using that URL
from sqlalchemy import create_engine, text

try:
    engine = create_engine(url, pool_pre_ping=True, future=True)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("✓ Successfully connected with SQLAlchemy!")
except Exception as e:
    print(f"✗ Failed to connect with SQLAlchemy: {e}")
