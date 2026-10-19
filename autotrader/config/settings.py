from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, read from AT_* environment variables or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AT_", extra="ignore")

    database_url: str = "sqlite:///data/autotrader.db"
    data_dir: Path = Path("data")
    yahoo_requests_per_sec: float = 2.0
    finnhub_api_key: str | None = None
    http_timeout_sec: float = 15.0

    alpaca_paper_key_id: str | None = Field(
        None, validation_alias=AliasChoices("ALPACA_PAPER_KEY_ID", "AT_ALPACA_PAPER_KEY_ID"))
    alpaca_paper_secret_key: str | None = Field(
        None, validation_alias=AliasChoices("ALPACA_PAPER_SECRET_KEY", "AT_ALPACA_PAPER_SECRET_KEY"))

    risk_file: Path = Path(__file__).parent / "risk.yaml"
    bots_file: Path = Path(__file__).parent / "bots.yaml"
    collar_bps: float = 15.0  # marketable-limit collar around the last price
    engine_data_delay_sec: int = 20  # wait after a bar closes before fetching it
    daily_tick_et: str = "19:05"
    evaluate_et: str = "16:30"  # daily evaluation (and Friday allocation) after the close

    ntfy_url: str | None = None  # e.g. https://ntfy.sh/<your-private-topic>
    discord_webhook: str | None = None  # Alpaca accepts OPG orders from 19:00 ET

    @property
    def bar_dir(self) -> Path:
        return self.data_dir / "bars"


@lru_cache
def get_settings() -> Settings:
    return Settings()
