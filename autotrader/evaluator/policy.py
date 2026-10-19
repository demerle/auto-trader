from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

DEFAULT_PATH = Path(__file__).parent.parent / "config" / "evaluation.yaml"


class Window(BaseModel):
    days: int
    trades: int


class EradicateRules(BaseModel):
    max_drawdown: float = 0.15
    min_psr: float = 0.5
    backtest_z: float = -2.0
    league_bottom_n: int = 1
    league_min_size: int = 4
    league_consecutive: int = 2


class PromoteRules(BaseModel):
    min_psr: float = 0.95
    min_alpha: float = 0.0
    min_excess_return: float = 0.0
    backtest_z: float = -1.0


class GateRules(BaseModel):
    min_oos_sharpe: float = 0.8
    min_dsr: float = 0.9
    max_drawdown: float = 0.25
    min_trades: int | dict[str, int] = 50  # flat, or per interval
    min_excess_sharpe: float = 0.0
    # On the sealed holdout (agent strategies only): the last stretch of history the
    # agent never saw. Looser than the walk-forward bar, since it is short.
    holdout_min_sharpe: float = 0.0
    holdout_min_excess_sharpe: float = -0.25

    def trades_for(self, interval: str) -> int:
        if isinstance(self.min_trades, int):
            return self.min_trades
        return self.min_trades.get(interval, max(self.min_trades.values()))


class AllocatorRules(BaseModel):
    min_mult: float = 0.5
    max_mult: float = 3.0
    prior_days: int = 20
    min_days: int = 5
    min_change_pct: float = 0.02


class Policy(BaseModel):
    slots: int = 12
    base_capital: float = 10_000.0
    windows: dict[str, Window]
    max_window_mult: float = 3.0
    eradicate: EradicateRules = EradicateRules()
    promote: PromoteRules = PromoteRules()
    gate: GateRules = GateRules()
    allocator: AllocatorRules = AllocatorRules()


@lru_cache
def load_policy(path: Path = DEFAULT_PATH) -> Policy:
    return Policy.model_validate(yaml.safe_load(path.read_text()))
