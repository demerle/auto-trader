"""`autotrader options ...`: the gauntlet's command line."""

import json
import logging

import pandas as pd
import typer

options_app = typer.Typer(help="The options gauntlet (weekly options competition).", no_args_is_help=True)


def _logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def gauntlet():
    from autotrader.broker.alpaca import AlpacaBroker
    from autotrader.config import get_settings
    from autotrader.data.archive import BarArchive
    from autotrader.db.session import session_factory
    from autotrader.options.config import load_gauntlet_config
    from autotrader.options.engine import Gauntlet
    from autotrader.risk.limits import load_limits

    cfg = load_gauntlet_config()
    return Gauntlet(session_factory(), AlpacaBroker.options_from_settings(cfg.account), BarArchive(), cfg,
                    load_limits(get_settings().risk_file))


@options_app.command("engine")
def engine(once: bool = typer.Option(False, help="Run one tick now and exit"),
           dry_run: bool = typer.Option(False, help="With --once: show intended orders, change nothing")) -> None:
    """Run the gauntlet engine (ticks, chasing, Monday reset, Friday cull)."""
    _logging()
    g = gauntlet()
    if once:
        typer.echo(str(g.tick(dry_run=dry_run)))
        return
    from autotrader.options.loop import run_forever

    run_forever(g)


@options_app.command("chain")
def chain(symbol: str, expiry: int = typer.Option(0, help="n-th expiry"), width: int = 5) -> None:
    """Fetch a CBOE chain and show the strikes around the money (nowcast at CBOE's own spot)."""
    from autotrader.options.cboe import fetch_chain
    from autotrader.options.helpers import Chain
    from autotrader.options.pricing import nowcast

    snap = fetch_chain(symbol)
    now = pd.Timestamp.now(tz="UTC")
    df = nowcast(snap.chain, snap.spot, snap.spot, snap.quote_time, max(now, snap.quote_time))
    c = Chain(snap.underlying, snap.spot, snap.quote_time, df, now.tz_convert("America/New_York").date())
    e = c.expiry(expiry)
    typer.echo(f"{snap.underlying} spot {snap.spot} | quotes as of {snap.quote_time.tz_convert('America/New_York'):%Y-%m-%d %H:%M} ET "
               f"| {len(snap.chain)} contracts | expiries {[str(x) for x in c.expiries()[:6]]}")
    if e is None:
        return
    for right in ("C", "P"):
        side = c.side(e, right)
        atm = side.iloc[(side["strike"] - c.spot).abs().argsort()[:1]].index[0] if len(side) else None
        rows = side.loc[max(atm - width, side.index[0]): atm + width] if atm is not None else side
        typer.echo(rows[["symbol", "bid", "ask", "mark", "iv", "delta", "oi", "volume"]].to_string(index=False))


@options_app.command("record")
def record() -> None:
    """Fetch and store one chain snapshot for every underlying in the universe."""
    from autotrader.options.archive import ChainArchive
    from autotrader.options.cboe import fetch_chain
    from autotrader.options.config import load_gauntlet_config

    cfg = load_gauntlet_config()
    arch = ChainArchive()
    for u in cfg.universe:
        try:
            snap = fetch_chain(u, cfg.chain.max_dte, cfg.chain.moneyness)
            typer.echo(f"{u}: {len(snap.chain)} contracts -> {arch.save(snap)}")
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"{u}: {exc}")


@options_app.command("smoke")
def smoke(target: str = typer.Argument(..., help="Builtin strategy name, or a path to a strategy file")) -> None:
    """Run the entry test (synthetic replay of the last sessions) on a strategy."""
    from pathlib import Path

    from autotrader.config import get_settings
    from autotrader.options import registry
    from autotrader.options.config import load_gauntlet_config
    from autotrader.options.smoke import smoke_test
    from autotrader.risk.limits import load_limits

    cfg, limits = load_gauntlet_config(), load_limits(get_settings().risk_file).options
    if Path(target).exists():
        meta = registry.static_meta(Path(target).read_text())
        res = smoke_test(target, cfg, limits, meta["underlyings"])
    else:
        res = smoke_test(registry.get(target), cfg, limits)
    typer.echo(json.dumps({"passed": res.passed, "reasons": res.reasons, **res.summary}, indent=1, default=str))


@options_app.command("seed")
def seed(names: str = typer.Option("", help="Comma-separated builtin names (default: all)"),
         skip_smoke: bool = False) -> None:
    """Enter builtin seed strategies into the gauntlet (smoke-tested first), up to the open seats."""
    from autotrader.config import get_settings
    from autotrader.db.session import session_factory
    from autotrader.options import competition, registry
    from autotrader.options.config import load_gauntlet_config
    from autotrader.options.roster import active_option_bots, create_option_bot, ensure_option_version
    from autotrader.options.smoke import smoke_test
    from autotrader.risk.limits import load_limits

    _logging()
    cfg, limits = load_gauntlet_config(), load_limits(get_settings().risk_file).options
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    builtins = [c for k, c in sorted(registry.discover().items()) if not registry.is_sandboxed(c)
                and (not wanted or c.name in wanted)]
    with session_factory()() as s:
        active = {b.strategy.name for b in active_option_bots(s)}
        seats = cfg.slots - len(active_option_bots(s))
        for cls in builtins:
            if cls.name in active:
                typer.echo(f"{cls.name}: already in")
                continue
            if seats <= 0:
                typer.echo("no open seats left")
                break
            if not skip_smoke:
                res = smoke_test(cls, cfg, limits)
                typer.echo(f"{cls.name}: smoke {'PASS' if res.passed else 'FAIL'} {res.reasons} "
                           f"(opened {res.summary.get('structures_opened')}, weeks "
                           f"{[w['ret'] for w in res.summary.get('weeks', [])]})")
                if not res.passed:
                    continue
            sv = ensure_option_version(s, cls.name, cls.version, author="seed")
            bot = create_option_bot(s, sv, cfg, notes=f"Seed strategy. Thesis: {cls.thesis}")
            competition.join_week(s, bot, pd.Timestamp.now(tz="UTC"), cfg.capital)
            seats -= 1
            typer.echo(f"  -> {bot.slug} enters the gauntlet with ${cfg.capital:,.0f}")
        s.commit()


@options_app.command("standings")
def standings() -> None:
    """This week's live standings."""
    from autotrader.db.session import session_factory
    from autotrader.options import views

    with session_factory()() as s:
        st = views.standings(s)
    typer.echo(f"Week of {st['week']} (day {st['day']}/{st['days']}), cut line below rank {st['cut_line']}")
    for r in st["rows"]:
        typer.echo(f"{r['rank']:>2}. {r['slug']:28} {r['ret'] * 100:+7.1f}%  ${r['equity']:>9,.0f}  "
                   f"today {r['day_pnl']:+9,.0f}  trades {r['trades']:>3}  open {r['open']}  {r['status']}")


@options_app.command("trades")
def trades(limit: int = 30) -> None:
    """Latest option orders across all gauntlet bots."""
    from autotrader.db.session import session_factory
    from autotrader.options import views

    with session_factory()() as s:
        for t in views.trade_feed(s, limit=limit):
            typer.echo(f"{t['time']}  {t['bot']:24} {t['intent']:5} {t['structure']:40} x{t['qty']:<3} "
                       f"{t['status']:9} @ {t['price'] if t['price'] is not None else '-'}  {t['reason']}")


@options_app.command("cull")
def cull(dry_run: bool = typer.Option(True, help="Preview the ranking without culling")) -> None:
    """Score this week now (normally automatic after the week's last session)."""
    from autotrader.data.archive import BarArchive
    from autotrader.db.session import session_factory
    from autotrader.options import competition
    from autotrader.options.config import load_gauntlet_config
    from autotrader.options.loop import practice

    cfg = load_gauntlet_config()
    now = pd.Timestamp.now(tz="UTC")
    with session_factory()() as s:
        out = competition.score_week(s, now, cfg, BarArchive(), apply=not dry_run,
                                     practice=practice(cfg, competition.week_of(now)))
        if not dry_run:
            s.commit()
    typer.echo(json.dumps(out, indent=1, default=str))


@options_app.command("kill")
def kill(reason: str = typer.Option("manual", help="Why")) -> None:
    """Options kill switch: cancel options orders and close every options position at market."""
    typer.echo(str(gauntlet().kill(reason)))


@options_app.command("unkill")
def unkill() -> None:
    gauntlet().unkill()
    typer.echo("options kill switch off")


@options_app.command("reconcile")
def reconcile() -> None:
    drift, stock = gauntlet().reconcile()
    typer.echo("no option drift" if not drift else "\n".join(f"{s}: ledger {a:+g} broker {b:+g}" for s, a, b in drift))
    if stock:
        typer.echo(f"STOCK POSITIONS in the options account: {stock}")
