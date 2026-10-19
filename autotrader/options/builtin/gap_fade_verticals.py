"""Fade big overnight gaps in QQQ and IWM with this week's debit verticals."""

from autotrader.options.helpers import gap, net_mark, units_for, vertical
from autotrader.options.strategy import Leg, OptionContext, OptionParams, OptionStrategy, Structure


class Params(OptionParams):
    min_gap: float = 0.010  # 1% gap from yesterday's close
    max_filled: float = 0.5  # skip if half the gap has already been filled
    budget_pct: float = 0.25  # per underlying
    width_pct: float = 0.012  # spread width as a fraction of the price
    take_profit_frac: float = 0.75  # close at 75% of the spread's max value
    stop: float = -0.60
    entry_start_et: int = 9 * 60 + 45
    entry_end_et: int = 10 * 60 + 30


class GapFadeVerticals(OptionStrategy):
    name = "gap_fade_verticals"
    version = "1"
    underlyings = ("QQQ", "IWM")
    description = "Debit verticals against >1% overnight gaps in QQQ/IWM, held up to Friday."
    thesis = ("Large index gaps are often driven by thin overnight trading and partially fill once the "
              "cash session's liquidity arrives. A debit vertical toward yesterday's close risks only its "
              "premium, pays up to ~3x if the gap fills, and doesn't need the fill to happen today.")
    Params = Params

    def on_tick(self, ctx: OptionContext):
        p = self.params
        et_min = ctx.et.hour * 60 + ctx.et.minute
        out = {}
        for tag, pos in ctx.positions.items():
            hit = pos.mark >= p.take_profit_frac * self._width(ctx, pos)
            if hit or pos.pnl_pct <= p.stop:
                continue
            out[tag] = Structure(pos.legs, pos.qty, pos.note)
        if not (p.entry_start_et <= et_min <= p.entry_end_et):
            return out
        done = {c.tag for c in ctx.closed_today} | set(out)
        for u in self.underlyings:
            tag = f"{u.lower()}-fade"
            if tag in done or u not in ctx.chains:
                continue
            today = ctx.today_bars(u)
            daily = ctx.daily.get(u)
            if today.empty or daily is None or daily.empty:
                continue
            g = gap(daily, float(today["open"].iloc[0]))
            if g is None or abs(g) < p.min_gap:
                continue
            prev = float(daily["close"].iloc[-1])
            last = float(today["close"].iloc[-1])
            filled = (float(today["open"].iloc[0]) - last) / (float(today["open"].iloc[0]) - prev)
            if filled > p.max_filled:
                continue
            chain = ctx.chains[u]
            expiry = chain.weekly()
            if expiry is None or chain.dte(expiry) == 0:
                expiry = chain.expiry(0, min_dte=1)
            if expiry is None:
                continue
            width = max(round(last * p.width_pct), 1)
            if g > 0:  # gapped up: buy a put spread toward yesterday's close
                legs = vertical(chain, expiry, "P", last, last - width)
            else:
                legs = vertical(chain, expiry, "C", last, last + width)
            if legs is None:
                continue
            price = net_mark(chain, legs)
            qty = min(units_for(ctx.equity * p.budget_pct, price or 0.0), 50)
            if qty >= 1:
                out[tag] = Structure(tuple(Leg(s, r) for s, r in legs), qty,
                                     f"{u} gapped {g:+.1%}; fading toward {prev:.2f}")
        return out

    @staticmethod
    def _width(ctx: OptionContext, pos) -> float:
        strikes = []
        for leg in pos.legs:
            for chain in ctx.chains.values():
                row = chain.row(leg.symbol)
                if row is not None:
                    strikes.append(float(row["strike"]))
        return (max(strikes) - min(strikes)) if len(strikes) == 2 else float("inf")
