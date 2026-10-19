"""Trade the gap between what VIX1D prices for the rest of the day and how much SPY actually moves intraday."""

import math

from autotrader.options.helpers import iron_condor, net_mark, straddle, units_for
from autotrader.options.strategy import Leg, OptionContext, OptionParams, OptionStrategy, Structure


class Params(OptionParams):
    cheap: float = 0.85  # VIX1D below 85% of realized intraday vol: buy gamma
    rich: float = 1.20  # above 120%: sell a condor
    lookback_sessions: int = 5
    straddle_budget_pct: float = 0.30
    condor_risk_pct: float = 0.45
    short_delta: float = 0.10
    wing: float = 5.0
    entry_start_et: int = 10 * 60
    entry_end_et: int = 11 * 60
    long_take: float = 0.60  # straddle: +60%
    long_stop: float = -0.40
    short_take: float = 0.50  # condor: keep half the credit
    short_stop: float = -1.50
    exit_et: int = 15 * 60 + 10


class Vix1dGammaSwitch(OptionStrategy):
    name = "vix1d_gamma_switch"
    version = "1"
    underlyings = ("SPY",)
    description = ("Each morning: buy a 0DTE SPY straddle if VIX1D is cheap vs recent intraday realized vol, "
                   "sell a 0DTE iron condor if it is rich.")
    thesis = ("VIX1D prices SPY's move for the rest of the day; recent 5-minute realized vol says how much "
              "SPY actually moves intraday. When the two disagree, same-day options are mispriced, and 0DTE "
              "gamma turns that mispricing into large percentage moves: long straddles when the market "
              "underprices movement, short condors (defined risk) when it overprices it.")
    Params = Params

    @staticmethod
    def _intraday_vol(ctx: OptionContext, sessions: int) -> float | None:
        bars = ctx.bars.get("SPY")
        if bars is None or bars.empty:
            return None
        past = bars[bars.index < ctx.session_open]
        if past.empty:
            return None
        days = past.groupby(past.index.tz_convert("America/New_York").date)
        rets = []
        for _, g in list(days)[-sessions:]:
            r = (g["close"].astype(float).apply(math.log).diff()).dropna()  # within the day only
            rets += list(r)
        if len(rets) < 50:
            return None
        mean = sum(rets) / len(rets)
        var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
        return math.sqrt(var * 78 * 252) * 100  # annualized, in VIX points

    def on_tick(self, ctx: OptionContext):
        p = self.params
        et_min = ctx.et.hour * 60 + ctx.et.minute
        for tag in ("straddle", "condor"):
            pos = ctx.positions.get(tag)
            if pos is None:
                continue
            take, stop = (p.long_take, p.long_stop) if tag == "straddle" else (p.short_take, p.short_stop)
            if pos.pnl_pct >= take or pos.pnl_pct <= stop or et_min >= p.exit_et:
                return {}
            return {tag: Structure(pos.legs, pos.qty, pos.note)}
        if ctx.closed_today or not (p.entry_start_et <= et_min <= p.entry_end_et) or "SPY" not in ctx.chains:
            return {}
        realized = self._intraday_vol(ctx, p.lookback_sessions)
        implied = ctx.vix1d
        if not realized or not implied:
            return {}
        chain = ctx.chains["SPY"]
        expiry = chain.expiry(0)
        if expiry is None or chain.dte(expiry) != 0:
            return {}
        ratio = implied / realized
        why = f"VIX1D {implied:.1f} vs {realized:.1f} intraday realized ({ratio:.2f}x)"
        if ratio <= p.cheap:
            legs = straddle(chain, expiry)
            price = net_mark(chain, legs) if legs else None
            qty = min(units_for(ctx.equity * p.straddle_budget_pct, price or 0.0), 50)
            if legs and qty >= 1:
                return {"straddle": Structure(tuple(Leg(s, r) for s, r in legs), qty, f"long gamma: {why}")}
        elif ratio >= p.rich:
            legs = iron_condor(chain, expiry, p.short_delta, p.wing)
            if not legs:
                return {}
            credit = -(net_mark(chain, legs) or 0.0)
            strikes = [float(chain.row(s)["strike"]) for s, _ in legs]
            width = max(abs(strikes[0] - strikes[1]), abs(strikes[2] - strikes[3]))
            risk = (width - credit) * 100
            qty = min(int(ctx.equity * p.condor_risk_pct // risk), 50) if risk > 0 and credit > 0.05 else 0
            if qty >= 1:
                return {"condor": Structure(tuple(Leg(s, r) for s, r in legs), qty,
                                            f"short gamma: {why}; credit {credit:.2f}")}
        return {}
