# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import List
from services import get_buy_sell_signals

app = FastAPI(
    title="Stock Signal API",
    description="API for calculating Buy/Sell signals based on Stochastics and Momentum.",
    version="1.0.0"
)

class TickerRequest(BaseModel):
    tickers: List[str] = Field(..., description="List of stock tickers to check", json_schema_extra={"example": ["SPY", "QQQ", "AAPL", "NVDA"]})
    data_range: str = Field(default="1y", description="Time range for the Yahoo Finance data lookback", json_schema_extra={"example": "1y"})
    interval: str = Field(default="1d", description="Stock price bar interval", json_schema_extra={"example": "1d"})

@app.post("/signals")
def calculate_signals(request: TickerRequest):
    if not request.tickers:
        raise HTTPException(status_code=400, detail="Tickers list cannot be empty")
    try:
        signals = get_buy_sell_signals(
            list_of_tickers=request.tickers,
            data_range=request.data_range,
            interval=request.interval
        )
        return signals
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
