"""Gold-scar: first-hour momentum in QQQ, ridden with in-the-money 0DTE options until the close."""

from autotrader.options.helpers import units_for
from autotrader.options.strategy import Leg, OptionContext, OptionParams, OptionStrategy, Structure


class Params(OptionParams):
    entry_et: int = 10 * 60 + 30  # decide once the 10:25-10:30 bar is complete
    entry_window: int = 10  # minutes; a late tick may still enter
    min_move: float = 0.002  # QQQ at least 0.2% from today's open by 10:30
    itm_strikes: int = 3  # buy 3 strikes in the money (~0.7-0.8 delta): little time value to bleed
    budget_pct: float = 0.20  # premium paid, as a fraction of equity
    max_contracts: int = 50  # per order (the risk limit); larger sizes are split across tags
    max_tags: int = 4
    stop_every_min: int = 15  # check the stop on quarter hours
    # Held until the engine flattens expiring contracts (15:20 ET), unless QQQ crosses back through the open.


class GoldScar(OptionStrategy):
    name = "gold_scar"
    version = "1"
    underlyings = ("QQQ",)
    description = ("At 10:30 ET, if QQQ is 0.2%+ away from its open, buy 3-strike-ITM 0DTE options in that "
                   "direction; exit if QQQ crosses back through the open, else hold to the 15:20 flatten.")
    thesis = ("Intraday momentum (Gao et al. 2018; Zarattini et al. 2024): a decisive first hour in an index "
              "tends to carry into the afternoon. In-the-money strikes keep the 0DTE volatility premium small "
              "(real 0DTE ATM IV runs ~1.7x VIX1D on SPY, ~2.2x on QQQ, so buyers pay up), and the open is the "
              "line where the thesis is wrong. Replayed through this engine on real QQQ 0DTE prints (Alpaca, "
              "Feb 2024 - Oct 2026, 142 weeks): +2.5% mean week, -2.9% median, 46% of weeks up, best +142%, "
              "worst -49%; positive in both halves of the sample but statistically weak (t ~ 1). Same-period "
              "rivals on real prints: odte_opening_drive -2.9%, vix1d_gamma_switch -3.0%, "
              "power_hour_straddle -3.7% mean week. budget_pct scales risk: 0.5 gave +10.5% mean, -12% median, "
              "best week +798%.")
    Params = Params

    def on_tick(self, ctx: OptionContext):
        p = self.params
        et_min = ctx.et.hour * 60 + ctx.et.minute
        today = ctx.today_bars("QQQ")
        if today.empty:
            return {t: Structure(pos.legs, pos.qty, pos.note) for t, pos in ctx.positions.items()}
        open_ = float(today["open"].iloc[0])
        last = float(today["close"].iloc[-1])
        if ctx.positions:
            long_calls = any(leg.symbol[-9] == "C" for pos in ctx.positions.values() for leg in pos.legs)
            crossed = last < open_ if long_calls else last > open_
            if et_min % p.stop_every_min == 0 and crossed:
                return {}  # momentum failed: everything out
            return {t: Structure(pos.legs, pos.qty, pos.note) for t, pos in ctx.positions.items()}
        if ctx.closed_today or not (p.entry_et <= et_min < p.entry_et + p.entry_window) or "QQQ" not in ctx.chains:
            return {}
        move = last / open_ - 1
        if abs(move) < p.min_move:
            return {}
        chain = ctx.chains["QQQ"]
        expiry = chain.expiry(0)
        if expiry is None or chain.dte(expiry) != 0:
            return {}
        right = "C" if move > 0 else "P"
        row = chain.atm(expiry, right, -p.itm_strikes if right == "C" else p.itm_strikes)
        if row is None:
            return {}
        total = units_for(ctx.equity * p.budget_pct, float(row["est_ask"]))
        out = {}
        note = (f"QQQ {move:+.2%} from the open at 10:30: {right} {float(row['strike']):g}, "
                f"out if it crosses {open_:.2f}")
        for i in range(p.max_tags):
            qty = min(total, p.max_contracts)
            if qty < 1:
                break
            out[f"scar-{i + 1}"] = Structure((Leg(str(row["symbol"]), 1),), qty, note)
            total -= qty
        return out
