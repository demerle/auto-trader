"""Pure verdict logic: given a bot's paper record, keep it, kill it, or flag it for live money."""

import math
from dataclasses import dataclass, field
from typing import Literal

from autotrader.db.models import BotStage
from autotrader.evaluator.policy import Policy

Action = Literal["keep", "eradicate", "flag_live", "unflag"]


@dataclass
class Verdict:
    action: Action
    reasons: list[str] = field(default_factory=list)

    @property
    def rationale(self) -> str:
        return "; ".join(self.reasons)


def _bad(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def decide(stage: BotStage, interval: str, days: int, trades: int, m: dict, policy: Policy,
           league_bottom: bool = False, prior_bottom_streak: int = 0) -> Verdict:
    w = policy.windows[interval]
    er, pr = policy.eradicate, policy.promote
    progress = f"{days}/{w.days} days, {trades}/{w.trades} trades"

    if days >= 5 and m.get("max_drawdown", 0.0) < -er.max_drawdown:
        return Verdict("eradicate", [f"drawdown {m['max_drawdown']:.1%} beyond -{er.max_drawdown:.0%}"])

    matured = days >= w.days and trades >= w.trades
    expired = days >= w.days * policy.max_window_mult
    if not matured:
        if expired:
            return Verdict("eradicate", [f"too inactive to judge after {days} days ({progress})"])
        return Verdict("keep", [f"probation: {progress}"])

    psr, z = m.get("psr", 0.0), m.get("backtest_z")
    excess = m.get("excess_return")
    if psr < er.min_psr:
        return Verdict("eradicate", [f"PSR {psr:.2f} < {er.min_psr} ({progress})"])
    if not _bad(z) and z < er.backtest_z:
        return Verdict("eradicate", [f"paper returns {z:.1f} std-errors below backtest"])
    if league_bottom and prior_bottom_streak + 1 >= er.league_consecutive \
            and not _bad(excess) and excess < 0:
        return Verdict("eradicate", [f"bottom of league {prior_bottom_streak + 1}x running "
                                     f"and trailing SPY by {-excess:.1%}"])

    proven = (psr >= pr.min_psr and m.get("alpha", 0.0) >= pr.min_alpha
              and not _bad(excess) and excess >= pr.min_excess_return
              and (_bad(z) or z >= pr.backtest_z))
    if stage == BotStage.PAPER_PROBATION and proven:
        return Verdict("flag_live", [f"PSR {psr:.2f}, alpha {m.get('alpha', 0):.1%}, "
                                     f"beat SPY by {excess:.1%} over {days} days"])
    if stage == BotStage.PAPER_PROVEN and not proven and psr < pr.min_psr - 0.1:
        return Verdict("unflag", [f"PSR fell to {psr:.2f}"])
    if stage == BotStage.PAPER_PROBATION and expired:
        return Verdict("eradicate", [f"not proven within {days} days (PSR {psr:.2f})"])
    return Verdict("keep", [f"PSR {psr:.2f}, Sharpe {m.get('sharpe', 0):.2f} ({progress})"])
