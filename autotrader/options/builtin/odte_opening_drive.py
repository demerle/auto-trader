"""0DTE opening drive: ride the first decisive break of the opening range with same-day options."""

from autotrader.options.helpers import opening_range, units_for
from autotrader.options.strategy import Leg, OptionContext, OptionParams, OptionStrategy, Structure


class Params(OptionParams):
    range_minutes: int = 15
    delta: float = 0.30
    budget_pct: float = 0.20  # of equity per entry (two underlyings -> up to 40% a day)
    stop: float = -0.50  # exit at -50% of premium
    target: float = 2.00  # take +200%
    last_entry_et: int = 12 * 60  # no new entries after noon (minutes since midnight ET)
    exit_et: int = 15 * 60  # out by 15:00 ET


class OdteOpeningDrive(OptionStrategy):
    name = "odte_opening_drive"
    version = "1"
    underlyings = ("SPY", "QQQ")
    description = "Buy ~30-delta 0DTE calls/puts on the first close outside the 15-minute opening range."
    thesis = ("Opening-range breaks on index ETFs often run for hours as overnight orders and "
              "trend followers pile in. Cheap same-day options turn a 0.5% move into a multi-bagger, "
              "and a hard -50% stop caps the many false breaks.")
    Params = Params

    def on_tick(self, ctx: OptionContext):
        p = self.params
        et_min = ctx.et.hour * 60 + ctx.et.minute
        out = {}
        done_today = {c.tag for c in ctx.closed_today}
        for u in self.underlyings:
            tag = f"{u.lower()}-drive"
            pos = ctx.positions.get(tag)
            if pos is not None:
                if pos.pnl_pct <= p.stop or pos.pnl_pct >= p.target or et_min >= p.exit_et:
                    continue  # leaving the tag out closes it
                out[tag] = Structure(pos.legs, pos.qty, pos.note)
                continue
            if tag in done_today or et_min > p.last_entry_et or u not in ctx.chains:
                continue
            today = ctx.today_bars(u)
            rng = opening_range(today, p.range_minutes)
            if rng is None or len(today) <= p.range_minutes // 5:
                continue
            hi, lo = rng
            last = float(today["close"].iloc[-1])
            right = "C" if last > hi else "P" if last < lo else None
            if right is None:
                continue
            chain = ctx.chains[u]
            expiry = chain.expiry(0)
            if expiry is None or chain.dte(expiry) != 0:
                continue
            row = chain.nearest_delta(expiry, right, p.delta)
            if row is None:
                continue
            qty = min(units_for(ctx.equity * p.budget_pct, float(row["est_ask"])), 50)
            if qty >= 1:
                side = "up" if right == "C" else "down"
                out[tag] = Structure((Leg(str(row["symbol"]), 1),), qty,
                                     f"{u} broke the opening range {side} at {last:.2f}")
        return out
