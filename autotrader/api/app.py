from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from autotrader.data.intervals import Interval
from autotrader.signals import get_buy_sell_signals

app = FastAPI(
    title="Autotrader",
    description="Strategy arena API. /signals is the original stochastic+momentum scan.",
    version="0.1.0",
)


class TickerRequest(BaseModel):
    tickers: list[str] = Field(..., json_schema_extra={"example": ["SPY", "QQQ", "AAPL", "NVDA"]})
    data_range: str = Field(default="1y", json_schema_extra={"example": "1y"})
    interval: Interval = Field(default="1d")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/signals")
def calculate_signals(request: TickerRequest) -> dict:
    if not request.tickers:
        raise HTTPException(status_code=400, detail="Tickers list cannot be empty")
    return get_buy_sell_signals(request.tickers, request.data_range, request.interval)
