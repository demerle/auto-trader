"""Lottery tickets on high-momentum single names: cheap OTM weeklies on volume-confirmed breakouts."""

from autotrader.options.helpers import units_for, vwap
from autotrader.options.strategy import Leg, OptionContext, OptionParams, OptionStrategy, Structure


class Params(OptionParams):
    volume_surge: float = 1.5  # today's volume so far vs the same time on recent days
    min_price: float = 0.20
    max_price: float = 1.00
    target_delta: float = 0.20
    bet_pct: float = 0.08  # of equity per ticket
    max_open: int = 4
    take_profit: float = 3.0  # +300%: sell the ticket
    stop: float = -0.70
    first_entry_et: int = 10 * 60 + 30
    last_entry_et: int = 14 * 60 + 30


class SingleNameLotto(OptionStrategy):
    name = "single_name_lotto"
    version = "1"
    underlyings = ("NVDA", "TSLA", "AMD", "PLTR")
    description = "Cheap ($0.20-1.00) OTM weekly calls/puts on volume-confirmed breakouts in hot single names."
    thesis = ("Momentum names move 3-5% in a day far more often than their options imply once a breakout "
              "gets volume behind it, and cheap OTM weeklies are where the convexity is: a ticket that "
              "pays 5-10x covers a string of small losses. Many small bets, tight -70% stops and letting "
              "winners run until Friday give a fat right tail, which is what wins a weekly competition.")
    Params = Params

    def _surge(self, ctx: OptionContext, u: str) -> float:
        bars = ctx.bars.get(u)
        if bars is None or bars.empty:
            return 0.0
        today = ctx.today_bars(u)
        if today.empty:
            return 0.0
        n = len(today)
        days = [g for d, g in bars[bars.index < ctx.session_open].groupby(bars.index[bars.index < ctx.session_open]
                                                                         .tz_convert("America/New_York").date)]
        past = [float(g["volume"].iloc[:n].sum()) for g in days[-5:] if len(g) >= n]
        base = sum(past) / len(past) if past else 0.0
        return float(today["volume"].sum()) / base if base > 0 else 0.0

    def on_tick(self, ctx: OptionContext):
        p = self.params
        et_min = ctx.et.hour * 60 + ctx.et.minute
        out = {}
        for tag, pos in ctx.positions.items():
            if pos.pnl_pct >= p.take_profit or pos.pnl_pct <= p.stop:
                continue
            out[tag] = Structure(pos.legs, pos.qty, pos.note)
        if not (p.first_entry_et <= et_min <= p.last_entry_et) or len(out) >= p.max_open:
            return out
        used = {t for t in out} | {c.tag for c in ctx.closed_today}
        for u in self.underlyings:
            if len(out) >= p.max_open or u not in ctx.chains:
                continue
            today = ctx.today_bars(u)
            daily = ctx.daily.get(u)
            if today.empty or daily is None or daily.empty:
                continue
            last = float(today["close"].iloc[-1])
            vw = vwap(today)
            prev_hi, prev_lo = float(daily["high"].iloc[-1]), float(daily["low"].iloc[-1])
            right = "C" if last > prev_hi and vw and last > vw else "P" if last < prev_lo and vw and last < vw else None
            tag = f"{u.lower()}-{right}"
            if right is None or tag in used or self._surge(ctx, u) < p.volume_surge:
                continue
            chain = ctx.chains[u]
            expiry = chain.weekly() if ctx.et.weekday() < 4 else chain.expiry(0)
            if expiry is None:
                continue
            side = chain.side(expiry, right)
            side = side[(side["mark"] >= p.min_price) & (side["mark"] <= p.max_price)]
            if side.empty:
                continue
            row = side.iloc[int((side["delta"].abs() - p.target_delta).abs().to_numpy().argmin())]
            qty = min(units_for(ctx.equity * p.bet_pct, float(row["est_ask"])), 50)
            if qty >= 1:
                out[tag] = Structure((Leg(str(row["symbol"]), 1),), qty,
                                     f"{u} {'breakout' if right == 'C' else 'breakdown'} at {last:.2f} on "
                                     f"{self._surge(ctx, u):.1f}x volume")
                used.add(tag)
        return out
