"""Gold-scar: an ML read of the first 15 minutes across ten ETFs, traded with deep in-the-money 0DTE options."""

from autotrader.options.helpers import units_for
from autotrader.options.ml.model import predict
from autotrader.options.strategy import Leg, OptionContext, OptionParams, OptionStrategy, Structure


class Params(OptionParams):
    decide_et: int = 9 * 60 + 45  # decide once the 9:40-9:45 bar is complete
    entry_window: int = 10  # minutes; a late tick may still decide (the model takes any bar)
    threshold: float = 1.0  # |z| of the model's prediction (out-of-sample std = 1)
    trade: tuple[str, ...] = ("SPY", "QQQ")
    itm_strikes: int = 4  # deep in the money: mostly delta, little time value to bleed
    budget_pct: float = 0.20  # premium per trade, as a fraction of equity
    max_contracts: int = 50  # per order (the risk limit); larger sizes are split across tags
    max_tags: int = 4
    # Held until the engine flattens expiring contracts (15:20 ET).


class GoldScar(OptionStrategy):
    name = "gold_scar"
    version = "2"
    underlyings = ("SPY", "QQQ", "IWM")
    bar_symbols = ("DIA", "TLT", "HYG", "SMH", "XLF", "XLK", "GLD", "^VIX")
    description = ("At 9:45 ET a ridge + transformer ensemble reads the first 15 minutes of SPY, QQQ, IWM and "
                   "seven cross-asset ETFs; on a strong call it buys 4-strike-ITM 0DTE options in that direction "
                   "and holds them to the 15:20 flatten.")
    thesis = ("Intraday continuation is real but small, and it shows up across assets: tech, credit and bonds "
              "in the first minutes lead the index into the afternoon. A ridge model and three small causal "
              "transformers, trained walk-forward on Alpaca 5-minute bars since 2016, have out-of-sample IC "
              "+0.00 to +0.07 per year in 2022-2026 (negative in 2020). Deep in-the-money strikes turn the call "
              "into near-delta-1 exposure with little time value. Replayed through this engine on real 0DTE prints "
              "(Alpaca, Feb 2024 - Oct 2026, 142 weeks, each year traded by a model trained only on earlier "
              "years): +1.8% mean week, median 0 (no trade in 23% of weeks), 37% of weeks up (48% of the weeks "
              "it traded), best +67%, worst -39%, t ~ 1.2. Same-period rivals on real prints: gold_scar v1 "
              "+2.5% mean / -2.9% median, odte_opening_drive -2.9%, vix1d_gamma_switch -3.0%, "
              "power_hour_straddle -3.7% mean week.")
    Params = Params

    def on_tick(self, ctx: OptionContext):
        p = self.params
        if ctx.positions:
            return {t: Structure(pos.legs, pos.qty, pos.note) for t, pos in ctx.positions.items()}
        et_min = ctx.et.hour * 60 + ctx.et.minute
        if ctx.closed_today or not (p.decide_et <= et_min < p.decide_et + p.entry_window):
            return {}
        k = (et_min - 570) // 5 - 1  # the last complete bar
        z = predict(ctx.bars, ctx.vix, k, ctx.et.normalize().tz_localize(None))
        out = {}
        for u in p.trade:
            if u not in z or abs(z[u]) < p.threshold or u not in ctx.chains:
                continue
            chain = ctx.chains[u]
            expiry = chain.expiry(0)
            if expiry is None or chain.dte(expiry) != 0:
                continue
            right = "C" if z[u] > 0 else "P"
            row = chain.atm(expiry, right, -p.itm_strikes if right == "C" else p.itm_strikes)
            if row is None:
                continue
            total = units_for(ctx.equity * p.budget_pct, float(row["est_ask"]))
            note = f"model z {z[u]:+.2f} on {u} at {ctx.et:%H:%M}: {right} {float(row['strike']):g}"
            for i in range(p.max_tags):
                qty = min(total, p.max_contracts)
                if qty < 1:
                    break
                out[f"{u.lower()}-{i + 1}"] = Structure((Leg(str(row["symbol"]), 1),), qty, note)
                total -= qty
        return out
