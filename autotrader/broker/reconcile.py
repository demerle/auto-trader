from dataclasses import dataclass

from sqlalchemy.orm import Session

from autotrader.broker.base import Broker
from autotrader.ledger.store import ledger_positions


@dataclass(frozen=True)
class Drift:
    symbol: str
    ledger: float
    broker: float


def reconcile(session: Session, broker: Broker) -> list[Drift]:
    """Compare the sum of bot ledgers with the broker's actual positions."""
    ours = ledger_positions(session)
    theirs = broker.positions()
    return [
        Drift(s, ours.get(s, 0.0), theirs.get(s, 0.0))
        for s in sorted(set(ours) | set(theirs))
        if abs(ours.get(s, 0.0) - theirs.get(s, 0.0)) > 1e-6
    ]
