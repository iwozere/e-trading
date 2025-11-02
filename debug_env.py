from dotenv import dotenv_values
import pathlib

env_path = pathlib.Path('config/donotshare/.env')
values = dotenv_values(env_path)
print('POSTGRES_USER:', values.get('POSTGRES_USER'))
print('POSTGRES_PASSWORD (first 5 chars):', values.get('POSTGRES_PASSWORD')[:5] if values.get('POSTGRES_PASSWORD') else 'None')
print('POSTGRES_HOST:', values.get('POSTGRES_HOST'))
print('POSTGRES_PORT:', values.get('POSTGRES_PORT'))
