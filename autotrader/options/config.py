from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from autotrader.config import get_settings


class ChainCfg(BaseModel):
    max_dte: int = 45
    moneyness: float = 0.25
    refresh_sec: int = 300
    record: bool = True


class ExecCfg(BaseModel):
    chase_sec: int = 30
    chase_steps: int = 4
    chase_frac: float = 0.25
    fee_per_contract: float = 0.05


class SmokeCfg(BaseModel):
    sessions: int = 20
    min_trades: int = 5


class GauntletConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account: str = "paper-options"
    slots: int = 6
    capital: float = 5000.0
    cull: int = 2
    min_sessions_to_cull: int = 3
    tick_minutes: int = 5
    first_tick_et: str = "09:35"
    reset_et: str = "09:00"
    score_et: str = "16:30"
    underlyings: dict[str, list[str]]
    cutoff_et: dict[str, str] = {"etfs": "15:30", "stocks": "15:15"}
    chain: ChainCfg = ChainCfg()
    execution: ExecCfg = ExecCfg()
    smoke: SmokeCfg = SmokeCfg()
    practice_until: str | None = None  # weeks starting (Monday) on or before this date: no cull

    @property
    def universe(self) -> tuple[str, ...]:
        return tuple(s for group in self.underlyings.values() for s in group)

    def cutoff(self, underlying: str, day) -> pd.Timestamp:
        """Alpaca's last order time (UTC) for contracts on `underlying` expiring on `day`."""
        group = next((g for g, syms in self.underlyings.items() if underlying in syms), "stocks")
        hh, mm = map(int, self.cutoff_et.get(group, "15:15").split(":"))
        return (pd.Timestamp(day) + pd.Timedelta(hours=hh, minutes=mm)) \
            .tz_localize("America/New_York").tz_convert("UTC")


@lru_cache
def load_gauntlet_config(path: Path | None = None) -> GauntletConfig:
    path = path or get_settings().options_file
    return GauntletConfig.model_validate(yaml.safe_load(Path(path).read_text()))
