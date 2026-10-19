"""Weekly greedy capital tilt: more paper capital to bots with better (shrunk) Sharpe.

Each active bot's capital = base_capital * multiplier, where the multiplier is
1 + shrunk annualized Sharpe, clipped to [min_mult, max_mult], then rescaled so
the arena's total paper capital stays base_capital * n_bots. Bots with too
little history keep a multiplier of 1. Moves are recorded as CashFlows.
"""

from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from autotrader.db.models import ARENA, Bot, CashFlow
from autotrader.evaluator.policy import Policy
from autotrader.evaluator.record import build_record, shrunk_sharpe
from autotrader.evaluator.run import EVALUATED_STAGES


@dataclass
class Move:
    bot: Bot
    current: float
    target: float
    score: float


def plan_allocation(session: Session, policy: Policy) -> list[Move]:
    a = policy.allocator
    bots = session.scalars(select(Bot).where(Bot.stage.in_(EVALUATED_STAGES), Bot.group == ARENA).order_by(Bot.id)).all()
    if not bots:
        return []
    scores, mults = {}, {}
    for bot in bots:
        rec = build_record(session, bot)
        s = shrunk_sharpe(rec.returns, a.prior_days) if rec.days >= a.min_days else 0.0
        scores[bot.id] = s
        mults[bot.id] = float(np.clip(1 + s, a.min_mult, a.max_mult))
    caps = bounded_shares(policy.base_capital * len(bots), mults,
                          policy.base_capital * a.min_mult, policy.base_capital * a.max_mult)
    return [Move(b, b.capital, caps[b.id], scores[b.id]) for b in bots]


def bounded_shares(total: float, weights: dict[int, float], lo: float, hi: float) -> dict[int, float]:
    """Split `total` in proportion to weights, with every share kept inside [lo, hi]."""
    fixed: dict[int, float] = {}
    for _ in range(len(weights) + 1):
        free = {k: w for k, w in weights.items() if k not in fixed}
        if not free:
            break
        remaining = total - sum(fixed.values())
        wsum = sum(free.values())
        shares = {k: remaining * w / wsum for k, w in free.items()}
        violators = {k: (lo if v < lo else hi) for k, v in shares.items() if v < lo or v > hi}
        if not violators:
            return fixed | shares
        fixed |= violators
    return fixed


def apply_allocation(session: Session, policy: Policy, moves: list[Move]) -> list[Move]:
    """All-or-nothing, so the arena's total capital is conserved exactly."""
    pct = policy.allocator.min_change_pct
    if all(abs(mv.target - mv.current) < pct * mv.current for mv in moves):
        return []
    done = []
    for mv in moves:
        delta = mv.target - mv.current
        if not delta:
            continue
        session.add(CashFlow(bot_id=mv.bot.id, amount=delta,
                             reason=f"weekly tilt: shrunk Sharpe {mv.score:+.2f}"))
        mv.bot.capital = mv.target
        done.append(mv)
    return done
