from sqlalchemy.orm import Session

from autotrader.db.models import SystemState

KILL_SWITCH = "kill_switch"
ACCOUNT_HWM = "account_hwm"


def get_state(session: Session, key: str, default: dict | None = None) -> dict:
    row = session.get(SystemState, key)
    return dict(row.value) if row else dict(default or {})


def set_state(session: Session, key: str, value: dict) -> None:
    row = session.get(SystemState, key)
    if row is None:
        session.add(SystemState(key=key, value=value))
    else:
        row.value = value


def is_killed(session: Session) -> bool:
    return bool(get_state(session, KILL_SWITCH).get("on"))
