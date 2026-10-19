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
    discord_webhook: str | None = None  # https://discord.com/api/webhooks/<id>/<token>
    # Or DMs from your own bot (takes precedence over the webhook). The bot must share a server with you.
    discord_bot_token: str | None = None
    discord_user_id: str | None = None

    # Dashboard (phase 6). It refuses to serve pages without a token.
    dashboard_token: str | None = None
    dashboard_url: str = "http://127.0.0.1:8005"  # for links in Discord posts
    dashboard_secure_cookie: bool = False  # set true when served over HTTPS

    # Agents (phase 5)
    agent_database_url: str | None = None  # least-privilege role for agent tools; else database_url
    workspace_dir: Path = Path("workspace")  # agents may write only workspace/job_<id>/
    opencode_bin: str = "opencode"
    agent_user: str | None = None  # run opencode as this unprivileged user (setpriv), in Docker
    agent_home: Path | None = None  # HOME for opencode (holds its auth.json and state)
    opencode_auth_file: Path | None = None  # copied into agent_home before each job (Docker)

    @property
    def agent_log_dir(self) -> Path:
        return self.data_dir / "agent_logs"

    @property
    def bar_dir(self) -> Path:
        return self.data_dir / "bars"


@lru_cache
def get_settings() -> Settings:
    return Settings()
