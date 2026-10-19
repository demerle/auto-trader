"""Trend-day afternoon gamma: a skewed 0DTE SPY straddle at 2pm on days that are already trending."""

from autotrader.options.helpers import units_for, vwap
from autotrader.options.strategy import Leg, OptionContext, OptionParams, OptionStrategy, Structure


class Params(OptionParams):
    min_move: float = 0.004  # SPY at least 0.4% from the open by 2pm
    vwap_share: float = 0.7  # and on the trend side of VWAP for 70% of the day's bars
    budget_pct: float = 0.30
    entry_et: int = 14 * 60
    entry_window: int = 15  # minutes
    take_profit: float = 0.80
    stop: float = -0.45
    exit_et: int = 15 * 60 + 15


class PowerHourStraddle(OptionStrategy):
    name = "power_hour_straddle"
    version = "1"
    underlyings = ("SPY",)
    description = "On trend days, buy 2 ATM 0DTE options with the trend + 1 against at 2pm; out by 3:15."
    thesis = ("Trend days tend to extend into the close as funds rebalance and late money chases, while "
              "same-day options are cheap in the last two hours because most of their time value is gone. "
              "A straddle tilted 2:1 toward the trend gains on continuation and is hedged against a sharp "
              "reversal; the position is small in time and its loss is capped at the premium.")
    Params = Params

    def on_tick(self, ctx: OptionContext):
        p = self.params
        et_min = ctx.et.hour * 60 + ctx.et.minute
        pos = ctx.positions.get("power")
        if pos is not None:
            if pos.pnl_pct >= p.take_profit or pos.pnl_pct <= p.stop or et_min >= p.exit_et:
                return {}
            return {"power": Structure(pos.legs, pos.qty, pos.note)}
        if ctx.closed_today or not (p.entry_et <= et_min < p.entry_et + p.entry_window) or "SPY" not in ctx.chains:
            return {}
        today = ctx.today_bars("SPY")
        if len(today) < 40:
            return {}
        open_, last = float(today["open"].iloc[0]), float(today["close"].iloc[-1])
        move = last / open_ - 1
        if abs(move) < p.min_move:
            return {}
        vw = vwap(today)
        typical = (today["high"] + today["low"] + today["close"]) / 3
        cum_vwap = (typical * today["volume"]).cumsum() / today["volume"].cumsum()
        side = (today["close"] > cum_vwap) if move > 0 else (today["close"] < cum_vwap)
        if vw is None or side.mean() < p.vwap_share:
            return {}
        chain = ctx.chains["SPY"]
        expiry = chain.expiry(0)
        if expiry is None or chain.dte(expiry) != 0:
            return {}
        c, pt = chain.atm(expiry, "C"), chain.atm(expiry, "P")
        if c is None or pt is None:
            return {}
        legs = ((Leg(str(c["symbol"]), 2), Leg(str(pt["symbol"]), 1)) if move > 0
                else (Leg(str(c["symbol"]), 1), Leg(str(pt["symbol"]), 2)))
        unit = sum(leg.ratio * float((c if leg.symbol == c["symbol"] else pt)["est_ask"]) for leg in legs)
        qty = min(units_for(ctx.equity * p.budget_pct, unit), 25)
        if qty < 1:
            return {}
        return {"power": Structure(legs, qty, f"trend day: SPY {move:+.2%} from the open, "
                                              f"{side.mean():.0%} of bars on the trend side of VWAP")}
