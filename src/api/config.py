"""
API-scoped settings loaded via pydantic-settings.

Reads from config/donotshare/.env using an absolute path derived from this
file's location, so the app works regardless of the working directory.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_FILE = Path(__file__).resolve().parents[2] / "config" / "donotshare" / ".env"


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Security — required, app refuses to start without them
    jwt_secret_key: str

    # Optional security
    cors_origins: str = ""
    internal_api_token: str = ""

    # Runtime
    api_reload: bool = False
    trading_api_port: int = 5003
    trading_webgui_port: int = 5002

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = APISettings()
