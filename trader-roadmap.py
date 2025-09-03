#!/usr/bin/env python3
"""
Alpaca × TradingView Autotrader (Skeleton)
------------------------------------------

Purpose
  - Receive TradingView webhooks (e.g., Chandelier Exit signals)
  - Apply basic risk controls & position sizing
  - Route orders to Alpaca (paper or live)
  - Maintain portfolio-level circuit breakers

Notes
  - Keep this file as your single-entry script for production.
  - Start with paper trading. Test thoroughly before going live.

Quick start
  1) pip install alpaca-trade-api flask python-dotenv tenacity pydantic
  2) Set env vars (see ENV CONFIG below) or create a .env file.
  3) python alpaca_tradingview_autobot_skeleton.py
  4) In TradingView Alert → set Webhook URL to http://YOUR_SERVER:5000/webhook
     Payload example (JSON):
       {
         "symbol": "SPY",
         "action": "buy",         // or "sell", "close"
         "strategy": "chandelier", // free-form tag
         "timeframe": "5m",
         "confidence": 0.78
       }

Security
  - Optionally require a shared secret in the header X-Webhook-Secret.
  - Run behind a reverse proxy (e.g., Nginx) + TLS.

"""

import json
import logging
import math
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Alpaca SDK
try:
    import alpaca_trade_api as tradeapi
except Exception as e:
    print("[WARN] alpaca_trade_api not installed yet. Install with: pip install alpaca-trade-api", file=sys.stderr)
    tradeapi = None

# =====================
# ENV CONFIG
# =====================
load_dotenv()

ALPACA_KEY = os.getenv("ALPACA_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "")
# Paper: https://paper-api.alpaca.markets ; Live: https://api.alpaca.markets
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Risk & Ops
ACCOUNT_CURRENCY = os.getenv("ACCOUNT_CURRENCY", "USD")
MAX_DAILY_DRAWDOWN_PCT = float(os.getenv("MAX_DAILY_DRAWDOWN_PCT", 2.0))   # e.g., 2% of equity
PER_TRADE_RISK_PCT = float(os.getenv("PER_TRADE_RISK_PCT", 1.0))           # risk per trade as % equity
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", 4))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", 20))
ALLOWED_SYMBOLS = set(os.getenv("ALLOWED_SYMBOLS", "SPY,QQQ").replace(" ", "").split(","))

# Execution
DEFAULT_TIF = os.getenv("DEFAULT_TIF", "day")  # or gtc
SLIPPAGE_BPS = int(os.getenv("SLIPPAGE_BPS", 5)) # 5 bps = 0.05%
USE_MARKET_ORDERS = os.getenv("USE_MARKET_ORDERS", "false").lower() == "true"

# Webhook security
REQUIRE_SECRET = os.getenv("REQUIRE_SECRET", "true").lower() == "true"
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "change-me")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "autobot.log")

# =====================
# LOGGING SETUP
# =====================
logger = logging.getLogger("autobot")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
handler = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=5)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
handler.setFormatter(fmt)
logger.addHandler(handler)
console = logging.StreamHandler(sys.stdout)
console.setFormatter(fmt)
logger.addHandler(console)

# =====================
# DATA MODELS
# =====================
class TVPayload(BaseModel):
    symbol: str
    action: str = Field(..., regex=r"^(buy|sell|close)$")
    strategy: Optional[str] = None
    timeframe: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    price: Optional[float] = None  # optional if you include it in your alert

# =====================
# STATE & UTILITIES
# =====================
class State:
    def __init__(self):
        self.start_of_day_equity: Optional[float] = None
        self.trades_today: int = 0
        self.last_reset_date: Optional[str] = None
        self.order_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.kill_switch: bool = False

STATE = State()


def is_new_trading_day() -> bool:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return STATE.last_reset_date != today


def reset_day(api: tradeapi.REST):
    account = api.get_account()
    STATE.start_of_day_equity = float(account.equity)
    STATE.trades_today = 0
    STATE.last_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logger.info(f"Day reset. Starting equity: {STATE.start_of_day_equity:.2f} {ACCOUNT_CURRENCY}")


class RiskError(Exception):
    pass


# =====================
# ALPACA CLIENT
# =====================

def get_api() -> tradeapi.REST:
    if tradeapi is None:
        raise RuntimeError("alpaca_trade_api not available. Install it first.")
    return tradeapi.REST(ALPACA_KEY, ALPACA_SECRET, ALPACA_BASE_URL)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4), reraise=True,
       retry=retry_if_exception_type(Exception))
def safe_submit_order(api: tradeapi.REST, **kwargs):
    return api.submit_order(**kwargs)


def fetch_last_trade_price(api: tradeapi.REST, symbol: str) -> float:
    # Using last quote/last trade depending on SDK; fallback to last close via bars
    try:
        last_trade = api.get_latest_trade(symbol)
        return float(last_trade.price)
    except Exception:
        bars = api.get_bars(symbol, "1Min", limit=1)
        if not bars:
            raise RuntimeError(f"No price data for {symbol}")
        return float(bars[0].c)


# =====================
# RISK & SIZING
# =====================

def check_global_risk(api: tradeapi.REST):
    # Reset each UTC day
    if is_new_trading_day():
        reset_day(api)

    if STATE.kill_switch:
        raise RiskError("Kill switch engaged")

    account = api.get_account()
    equity = float(account.equity)
    if STATE.start_of_day_equity is None:
        STATE.start_of_day_equity = equity

    dd = (STATE.start_of_day_equity - equity) / STATE.start_of_day_equity * 100.0
    if dd >= MAX_DAILY_DRAWDOWN_PCT:
        STATE.kill_switch = True
        raise RiskError(f"Max daily drawdown exceeded: {dd:.2f}% >= {MAX_DAILY_DRAWDOWN_PCT}%")

    if STATE.trades_today >= MAX_TRADES_PER_DAY:
        raise RiskError(f"Max trades per day reached: {MAX_TRADES_PER_DAY}")


def position_sizing_qty(api: tradeapi.REST, symbol: str, stop_pct: float = 0.5) -> int:
    """
    Compute shares based on risk per trade and an assumed stop distance.
    stop_pct: e.g., 0.5 means 0.5% distance between entry and stop.
    """
    account = api.get_account()
    equity = float(account.equity)
    max_risk = equity * (PER_TRADE_RISK_PCT / 100.0)

    price = fetch_last_trade_price(api, symbol)
    stop_distance = price * (stop_pct / 100.0)
    if stop_distance <= 0:
        return 0

    qty = math.floor(max_risk / stop_distance)
    return max(qty, 0)


# =====================
# ORDER ROUTING
# =====================

def build_order(api: tradeapi.REST, symbol: str, side: str, qty: int) -> Dict[str, Any]:
    if qty <= 0:
        raise RiskError("Calculated quantity is zero; skipping order")

    price = fetch_last_trade_price(api, symbol)

    if USE_MARKET_ORDERS:
        return dict(symbol=symbol, qty=qty, side=side, type="market", time_in_force=DEFAULT_TIF)

    # Marketable limit: offset by slippage bps
    offset = price * (SLIPPAGE_BPS / 10_000.0)
    if side == "buy":
        limit_price = round(price + offset, 2)
    else:
        limit_price = round(price - offset, 2)

    return dict(symbol=symbol, qty=qty, side=side, type="limit", limit_price=str(limit_price), time_in_force=DEFAULT_TIF)


def open_positions_count(api: tradeapi.REST) -> int:
    try:
        positions = api.list_positions()
        return len(positions)
    except Exception:
        return 0


def close_position_if_exists(api: tradeapi.REST, symbol: str):
    try:
        api.close_position(symbol)
        logger.info(f"Requested close for existing {symbol} position")
    except Exception:
        pass


# =====================
# WEB SERVER
# =====================
app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "kill_switch": STATE.kill_switch, "trades_today": STATE.trades_today}), 200


@app.route("/webhook", methods=["POST"])
def webhook():
    # Optional shared secret verification
    if REQUIRE_SECRET:
        provided = request.headers.get("X-Webhook-Secret", "")
        if provided != WEBHOOK_SECRET:
            logger.warning("Unauthorized webhook attempt")
            return jsonify({"error": "unauthorized"}), 401

    try:
        payload_raw = request.get_data(as_text=True)
        payload_json = json.loads(payload_raw) if payload_raw else {}
        tv = TVPayload(**payload_json)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Invalid payload: {e}")
        return jsonify({"error": "invalid payload"}), 400

    symbol = tv.symbol.upper()
    action = tv.action.lower()

    if symbol not in ALLOWED_SYMBOLS:
        logger.info(f"Symbol {symbol} not in ALLOWED_SYMBOLS; skipping")
        return jsonify({"status": "ignored", "reason": "symbol not allowed"}), 200

    # Enqueue to avoid blocking webhook
    STATE.order_queue.put({"symbol": symbol, "action": action, "tv": tv.model_dump()})
    logger.info(f"Enqueued signal: {symbol} {action} {tv.strategy or ''} tf={tv.timeframe or ''}")
    return jsonify({"status": "accepted"}), 202


# =====================
# WORKER LOOP
# =====================

def worker_loop():
    api = get_api()
    logger.info("Worker started")

    while True:
        try:
            task = STATE.order_queue.get(timeout=1.0)
        except queue.Empty:
            # idle tick: also check daily reset periodically
            try:
                if is_new_trading_day():
                    reset_day(api)
            except Exception as e:
                logger.warning(f"Day reset check failed: {e}")
            continue

        try:
            check_global_risk(api)

            symbol = task["symbol"]
            action = task["action"]

            # Limit concurrent positions
            if action in ("buy", "sell"):
                if open_positions_count(api) >= MAX_CONCURRENT_POSITIONS:
                    logger.info("Max concurrent positions reached; skipping new entry")
                    continue

            # Position sizing (assume 0.5% stop distance baseline; tune per strategy)
            qty = position_sizing_qty(api, symbol, stop_pct=0.5)

            if action == "close":
                close_position_if_exists(api, symbol)
                logger.info(f"Closed (if existed): {symbol}")
                continue

            side = "buy" if action == "buy" else "sell"

            # If we are about to open an opposite direction, close existing first
            close_position_if_exists(api, symbol)

            order_req = build_order(api, symbol, side, qty)
            resp = safe_submit_order(api, **order_req)
            STATE.trades_today += 1

            logger.info(f"ORDER SENT: {order_req} | Resp: {getattr(resp, 'id', resp)}")
        except RiskError as re:
            logger.warning(f"Risk block: {re}")
        except Exception as e:
            logger.exception(f"Order processing error: {e}")
        finally:
            STATE.order_queue.task_done()


# =====================
# GRACEFUL SHUTDOWN
# =====================

def _sigterm(_signo, _frame):
    logger.info("Received shutdown signal. Exiting...")
    os._exit(0)

signal.signal(signal.SIGINT, _sigterm)
signal.signal(signal.SIGTERM, _sigterm)


# =====================
# MAIN
# =====================
if __name__ == "__main__":
    missing = [k for k in ("ALPACA_KEY", "ALPACA_SECRET") if not globals()[k]]
    if missing:
        logger.error(f"Missing required env vars: {missing}. Create a .env or export them.")
        sys.exit(1)

    # Start worker thread
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()

    # Start Flask app
    port = int(os.getenv("PORT", 5000))
    logger.info(f"Starting webhook server on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
