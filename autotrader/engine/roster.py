"""Sync the bot roster (config/bots.yaml) and strategy versions into the database."""

import hashlib
import inspect
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from autotrader.db.models import Bot, BotStage, StrategyVersion
from autotrader.strategies import registry


def ensure_strategy_version(session: Session, name: str, version: str | None = None) -> StrategyVersion:
    cls = registry.get(name, version)
    source = Path(inspect.getfile(cls)).read_bytes()
    sha = hashlib.sha256(source).hexdigest()
    sv = session.scalars(select(StrategyVersion).where(
        StrategyVersion.name == cls.name, StrategyVersion.version == cls.version)).first()
    if sv is None:
        sv = StrategyVersion(name=cls.name, version=cls.version, interval=cls.interval,
                             module=cls.__module__, code_sha=sha, description=cls.description)
        session.add(sv)
        session.flush()
    elif sv.code_sha != sha:
        raise ValueError(f"{cls.name}@{cls.version} source changed since it was registered; "
                         "bump its version instead of editing it in place")
    return sv


def sync_roster(session: Session, path: Path) -> list[str]:
    cfg = yaml.safe_load(path.read_text())
    account = cfg.get("account", "paper")
    out = []
    for entry in cfg["bots"]:
        sv = ensure_strategy_version(session, entry["strategy"], entry.get("version"))
        bot = session.scalars(select(Bot).where(Bot.slug == entry["slug"])).first()
        if bot is not None:
            out.append(f"exists  {bot.slug} ({sv.name}@{sv.version})")
            continue
        capital = float(entry.get("capital", 10_000))
        session.add(Bot(slug=entry["slug"], strategy_id=sv.id, params=entry.get("params", {}),
                        stage=BotStage.PAPER_PROBATION, account=account,
                        starting_capital=capital, capital=capital))
        out.append(f"created {entry['slug']} ({sv.name}@{sv.version}, ${capital:,.0f})")
    return out
