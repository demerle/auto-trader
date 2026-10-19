"""Agent settings (config/agents.yaml): models, budget, caps, data windows."""

from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, field_validator

DEFAULT_PATH = Path(__file__).parent.parent / "config" / "agents.yaml"
ROLES = ("researcher", "auditor", "reviewer", "supervisor", "options_researcher", "options_auditor")


class Windows(BaseModel):
    five_hours: float
    week: float
    month: float


class Budget(BaseModel):
    five_hours: float = 6.0
    week: float = 15.0
    month: float = 30.0
    go_caps: Windows = Windows(five_hours=12.0, week=30.0, month=60.0)
    go_guard: float = 0.9
    alert_at: float = 0.8
    seed_estimates: dict[str, float] = {"researcher": 1.5, "auditor": 0.4, "reviewer": 0.1,
                                        "supervisor": 0.02, "options_researcher": 1.5,
                                        "options_auditor": 0.4}

    @property
    def limits(self) -> dict[str, float]:
        return {"five_hours": self.five_hours, "week": self.week, "month": self.month}


class PerJob(BaseModel):
    backtests: int = 40
    submissions: int = 3


class Review(BaseModel):
    weekday: int = 5
    min_window_frac: float = 0.5


class AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a stray `models:` key must not be ignored

    model: str  # the only model any agent call may use
    budget: Budget = Budget()
    timeouts_min: dict[str, float] = {"researcher": 45, "auditor": 15, "reviewer": 10, "supervisor": 5,
                                      "options_researcher": 45, "options_auditor": 15}
    max_attempts: int = 2
    concurrency: int = 1
    per_job: PerJob = PerJob()
    research_intervals: list[str] = ["1d", "1h", "15m", "5m"]
    min_history_sessions: int = 60
    history_years: dict[str, float] = {"1d": 12}
    max_universe: dict[str, int] = {"1d": 60}
    holdout_days: dict[str, int] = {"1d": 365}
    review: Review = Review()
    auto_commit: bool = True
    daily_commentary: bool = True  # supervisor agent comments on each daily report
    sandbox_memory_mb: int = 3072

    @field_validator("model")
    @classmethod
    def _provider_model(cls, v: str) -> str:
        if "/" not in v or v.strip() != v:
            raise ValueError("model must look like provider/model, e.g. opencode-go/glm-5.3-flash")
        return v

    def model_for(self, role: str, attempt: int = 1) -> str:
        """Every role and every attempt uses the same configured model."""
        return self.model

    def holdout_start(self, interval: str, now: pd.Timestamp | None = None) -> pd.Timestamp:
        """Bars at or after this are sealed from agents (start of a UTC day, so stable intraday)."""
        now = now or pd.Timestamp.now(tz="UTC")
        return now.normalize() - pd.Timedelta(days=self.holdout_days.get(interval, 365))

    def history_start(self, interval: str, now: pd.Timestamp | None = None) -> pd.Timestamp:
        now = now or pd.Timestamp.now(tz="UTC")
        return now.normalize() - pd.Timedelta(days=int(365 * self.history_years.get(interval, 2)))


def load_agents_config(path: Path = DEFAULT_PATH) -> AgentsConfig:
    """Read fresh each time, so editing agents.yaml (e.g. the model) applies to the next job."""
    return AgentsConfig.model_validate(yaml.safe_load(path.read_text()))
