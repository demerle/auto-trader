"""Dashboard login: one shared token (AT_DASHBOARD_TOKEN) exchanged for a signed cookie.

The cookie is HttpOnly and SameSite=Strict; every form also carries a CSRF token
derived from the session, so a forged cross-site POST is rejected twice over.
"""

import hashlib
import hmac
import secrets
import time

from fastapi import Form, HTTPException, Request

from autotrader.config import get_settings

COOKIE = "at_session"
SESSION_DAYS = 7


class NeedLogin(Exception):
    """Raised by page dependencies; the app turns it into a redirect to /login."""


def _key() -> bytes:
    token = get_settings().dashboard_token
    if not token:
        raise RuntimeError("AT_DASHBOARD_TOKEN is not set; the dashboard will not run without it")
    return hashlib.sha256(b"autotrader-dashboard:" + token.encode()).digest()


def _sign(payload: str) -> str:
    return hmac.new(_key(), payload.encode(), hashlib.sha256).hexdigest()


def check_token(candidate: str) -> bool:
    token = get_settings().dashboard_token or ""
    return bool(token) and hmac.compare_digest(candidate.encode(), token.encode())


def new_session(now: float | None = None) -> str:
    exp = int((now or time.time()) + SESSION_DAYS * 86400)
    payload = f"{secrets.token_urlsafe(16)}.{exp}"
    return f"{payload}.{_sign(payload)}"


def session_id(value: str | None, now: float | None = None) -> str | None:
    """The session nonce if the cookie is authentic and unexpired."""
    if not value or value.count(".") != 2:
        return None
    nonce, exp, sig = value.split(".")
    if not hmac.compare_digest(sig, _sign(f"{nonce}.{exp}")):
        return None
    if not exp.isdigit() or int(exp) < (now or time.time()):
        return None
    return nonce


def csrf_for(nonce: str) -> str:
    return _sign(f"csrf:{nonce}")[:32]


def require_login(request: Request) -> str:
    nonce = session_id(request.cookies.get(COOKIE))
    if nonce is None:
        raise NeedLogin()
    request.state.csrf = csrf_for(nonce)
    return nonce


def require_csrf(request: Request, csrf: str = Form("")) -> str:
    nonce = require_login(request)
    if not hmac.compare_digest(csrf.encode(), csrf_for(nonce).encode()):
        raise HTTPException(403, "bad or missing CSRF token; reload the page")
    return nonce
