from sqlalchemy.orm import Session

from autotrader.db.models import SystemState

KILL_SWITCH = "kill_switch"
ACCOUNT_HWM = "account_hwm"
ARENA_ACCOUNT = "paper"


def account_key(key: str, account: str = ARENA_ACCOUNT) -> str:
    """Per-account state key; the arena's paper account keeps the original names."""
    return key if account == ARENA_ACCOUNT else f"{key}:{account}"


def get_state(session: Session, key: str, default: dict | None = None) -> dict:
    row = session.get(SystemState, key)
    return dict(row.value) if row else dict(default or {})


def set_state(session: Session, key: str, value: dict) -> None:
    row = session.get(SystemState, key)
    if row is None:
        session.add(SystemState(key=key, value=value))
    else:
        row.value = value


def is_killed(session: Session, account: str = ARENA_ACCOUNT) -> bool:
    return bool(get_state(session, account_key(KILL_SWITCH, account)).get("on"))
