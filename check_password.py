from dotenv import dotenv_values
import pathlib
from urllib.parse import quote_plus

env_path = pathlib.Path('config/donotshare/.env')
values = dotenv_values(env_path)

password = values.get('POSTGRES_PASSWORD')
print(f"Raw password: {password}")
print(f"Password length: {len(password) if password else 0}")
print(f"URL-encoded password: {quote_plus(password) if password else 'None'}")
print(f"Has special chars: {any(c in password for c in '.!@#$%^&*()[]{}:;' if password)}")
