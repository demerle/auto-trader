from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from autotrader.api.auth import NeedLogin
from autotrader.config import get_settings
from autotrader.data.intervals import Interval
from autotrader.signals import get_buy_sell_signals

HERE = Path(__file__).parent
SECURITY_HEADERS = {
    "Content-Security-Policy": ("default-src 'none'; style-src 'self' 'unsafe-inline'; script-src 'self'; "
                                "img-src 'self' data:; form-action 'self'; frame-ancestors 'none'; "
                                "base-uri 'none'"),
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


class TickerRequest(BaseModel):
    tickers: list[str] = Field(..., json_schema_extra={"example": ["SPY", "QQQ", "AAPL", "NVDA"]})
    data_range: str = Field(default="1y", json_schema_extra={"example": "1y"})
    interval: Interval = Field(default="1d")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not get_settings().dashboard_token:
        raise RuntimeError("AT_DASHBOARD_TOKEN is not set; refusing to serve the dashboard. "
                           "Add a long random value to .env.")
    yield


def create_app() -> FastAPI:
    from autotrader.api import actions, pages, pages_options

    app = FastAPI(title="Autotrader", version="0.2.0", lifespan=lifespan,
                  description="Strategy arena dashboard. /signals is the original stochastic scan.",
                  docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        for k, v in SECURITY_HEADERS.items():
            response.headers.setdefault(k, v)
        if request.url.path not in ("/health",) and not request.url.path.startswith("/static"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(NeedLogin)
    async def need_login(request, exc):
        return RedirectResponse(f"/login?next={quote(request.url.path)}", status_code=303)

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/signals", include_in_schema=False)
    def calculate_signals(request: TickerRequest, _=pages.LOGIN) -> dict:
        if not request.tickers:
            raise HTTPException(status_code=400, detail="Tickers list cannot be empty")
        return get_buy_sell_signals(request.tickers, request.data_range, request.interval)

    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    app.include_router(pages.router)
    app.include_router(pages_options.router)
    app.include_router(actions.router)
    return app


app = create_app()
