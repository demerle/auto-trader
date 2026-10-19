"""Options bots in the database: strategy versions (code sha pinned) and bot rows."""

import hashlib
import inspect
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from autotrader.db.models import OPTIONS, Bot, BotStage, StrategyVersion
from autotrader.options import registry
from autotrader.options.config import GauntletConfig

INTERVAL = "opt"


def ensure_option_version(session: Session, name: str, version: str | None = None,
                          author: str = "human", metrics: dict | None = None) -> StrategyVersion:
    cls = registry.get(name, version)
    path = getattr(cls, "source_path", None) or inspect.getfile(cls)
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    sv = session.scalars(select(StrategyVersion).where(
        StrategyVersion.name == cls.name, StrategyVersion.version == cls.version)).first()
    if sv is None:
        sv = StrategyVersion(name=cls.name, version=cls.version, interval=INTERVAL, module=cls.__module__,
                             code_sha=sha, description=cls.description, author=author,
                             backtest_metrics=dict(metrics or {}, thesis=cls.thesis,
                                                   underlyings=list(cls.underlyings)))
        session.add(sv)
        session.flush()
    elif sv.code_sha != sha:
        raise ValueError(f"{cls.name}@{cls.version} source changed since it was registered; bump its version")
    elif sv.interval != INTERVAL:
        raise ValueError(f"{cls.name}@{cls.version} is already an arena strategy")
    return sv


def unique_slug(session: Session, base: str) -> str:
    slug, n = base, 2
    while session.scalars(select(Bot).where(Bot.slug == slug)).first():
        slug, n = f"{base}-{n}", n + 1
    return slug


def create_option_bot(session: Session, sv: StrategyVersion, cfg: GauntletConfig, notes: str = "",
                      slug: str | None = None) -> Bot:
    base = slug or sv.name.replace("_", "-") + (f"-v{sv.version}" if sv.version != "1" else "")
    bot = Bot(slug=unique_slug(session, base), strategy_id=sv.id, params={}, stage=BotStage.PAPER_PROBATION,
              account=cfg.account, group=OPTIONS, starting_capital=cfg.capital, capital=cfg.capital,
              notes=notes, stage_since=pd.Timestamp.now(tz="UTC").to_pydatetime())
    session.add(bot)
    session.flush()
    return bot


def active_option_bots(session: Session) -> list[Bot]:
    from autotrader.engine.arena import ACTIVE_STAGES

    return list(session.scalars(select(Bot).where(Bot.group == OPTIONS, Bot.stage.in_(ACTIVE_STAGES))
                                .order_by(Bot.id)).all())
