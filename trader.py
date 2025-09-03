from flask import Flask, request, jsonify
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("APCA_API_KEY_ID")
API_SECRET = os.getenv("APCA_API_SECRET_KEY")
BASE_URL = os.getenv("APCA_API_BASE_URL")

app = Flask(__name__)

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    action = data.get("action")
    symbol = data.get("symbol", "SPY")
    qty = int(data.get("size", 1))

    if action == "buy":
        api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='gtc')
        return jsonify({"status": "bought"})
    elif action == "sell":
        api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='gtc')
        return jsonify({"status": "sold"})
    else:
        return jsonify({"status": "ignored", "reason": "no valid action"})
