from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, read from AT_* environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AT_", extra="ignore")

    database_url: str = "sqlite:///data/autotrader.db"
    data_dir: Path = Path("data")
    yahoo_requests_per_sec: float = 2.0
    finnhub_api_key: str | None = None
    http_timeout_sec: float = 15.0

    @property
    def bar_dir(self) -> Path:
        return self.data_dir / "bars"


@lru_cache
def get_settings() -> Settings:
    return Settings()
