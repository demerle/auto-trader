"""Agent settings (config/agents.yaml): models, budget, caps, data windows."""

from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel

DEFAULT_PATH = Path(__file__).parent.parent / "config" / "agents.yaml"
ROLES = ("researcher", "auditor", "reviewer")


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
    seed_estimates: dict[str, float] = {"researcher": 1.5, "auditor": 0.4, "reviewer": 0.1}

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
    models: dict[str, str]
    fallback_models: dict[str, str] = {}
    budget: Budget = Budget()
    timeouts_min: dict[str, float] = {"researcher": 45, "auditor": 15, "reviewer": 10}
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
    sandbox_memory_mb: int = 3072

    def model_for(self, role: str, attempt: int = 1) -> str:
        if attempt > 1 and role in self.fallback_models:
            return self.fallback_models[role]
        return self.models[role]

    def holdout_start(self, interval: str, now: pd.Timestamp | None = None) -> pd.Timestamp:
        """Bars at or after this are sealed from agents (start of a UTC day, so stable intraday)."""
        now = now or pd.Timestamp.now(tz="UTC")
        return now.normalize() - pd.Timedelta(days=self.holdout_days.get(interval, 365))

    def history_start(self, interval: str, now: pd.Timestamp | None = None) -> pd.Timestamp:
        now = now or pd.Timestamp.now(tz="UTC")
        return now.normalize() - pd.Timedelta(days=int(365 * self.history_years.get(interval, 2)))


@lru_cache
def load_agents_config(path: Path = DEFAULT_PATH) -> AgentsConfig:
    return AgentsConfig.model_validate(yaml.safe_load(path.read_text()))
