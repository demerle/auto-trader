"""Daily evaluation: score every paper bot, apply verdicts, queue research for open slots."""

import json
import logging
import math
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from autotrader.alerts import alert
from autotrader.db.models import ARENA, AgentJob, Bot, BotStage, Evaluation, utcnow
from autotrader.evaluator.decide import Verdict, decide
from autotrader.evaluator.policy import Policy
from autotrader.evaluator.record import PaperRecord, build_record

log = logging.getLogger(__name__)
EVALUATED_STAGES = (BotStage.PAPER_PROBATION, BotStage.PAPER_PROVEN)


@dataclass
class Outcome:
    bot: Bot
    record: PaperRecord
    verdict: Verdict


def clean_json(d: dict) -> dict:
    """JSON-safe copy: NaN/inf -> None."""
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in d.items()}


def _bottom_streak(session: Session, bot: Bot) -> int:
    evals = session.scalars(select(Evaluation).where(Evaluation.bot_id == bot.id)
                            .order_by(Evaluation.ts.desc()).limit(10)).all()
    n = 0
    for e in evals:
        if not (e.metrics or {}).get("league_bottom"):
            break
        n += 1
    return n


def autopsy(bot: Bot, rec: PaperRecord, verdict: Verdict) -> str:
    m = clean_json(rec.metrics)
    return (f"Bot {bot.slug} running {bot.strategy.name}@{bot.strategy.version} "
            f"({bot.strategy.interval}, params {json.dumps(bot.params)}) was eradicated: "
            f"{verdict.rationale}. Paper record: {rec.days} days, {rec.trades} closing trades, "
            f"metrics {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})}. "
            f"Strategy description: {bot.strategy.description}")


def evaluate_all(session: Session, policy: Policy, spy_returns: pd.Series | None,
                 apply: bool = True) -> list[Outcome]:
    bots = session.scalars(select(Bot).where(Bot.stage.in_(EVALUATED_STAGES), Bot.group == ARENA)
                           .order_by(Bot.id)).all()
    records = {b.id: build_record(session, b, spy_returns) for b in bots}

    league = sorted((b for b in bots if records[b.id].days >= 5 and records[b.id].metrics),
                    key=lambda b: records[b.id].metrics.get("sharpe", 0.0))
    er = policy.eradicate
    bottom = {b.id for b in league[:er.league_bottom_n]} if len(league) >= er.league_min_size else set()

    outcomes = []
    for bot in bots:
        rec = records[bot.id]
        verdict = decide(bot.stage, bot.strategy.interval, rec.days, rec.trades, rec.metrics, policy,
                         league_bottom=bot.id in bottom, prior_bottom_streak=_bottom_streak(session, bot))
        outcomes.append(Outcome(bot, rec, verdict))
        if apply:
            _apply(session, bot, rec, verdict, bot.id in bottom)
    return outcomes


def _apply(session: Session, bot: Bot, rec: PaperRecord, verdict: Verdict, bottom: bool) -> None:
    metrics = clean_json(rec.metrics | {"days": rec.days, "trades": rec.trades,
                                        "league_bottom": bottom})
    session.add(Evaluation(bot_id=bot.id, metrics=metrics, decision=verdict.action,
                           rationale=verdict.rationale))
    now = utcnow()
    if verdict.action == "eradicate":
        bot.stage, bot.stage_since, bot.target_qty = BotStage.ERADICATED, now, {}
        text = autopsy(bot, rec, verdict)
        bot.notes = text
        session.add(AgentJob(agent="researcher", model="", status="queued", bot_id=bot.id,
                             prompt=f"A slot opened. Research a replacement strategy.\n\nGraveyard entry:\n{text}"))
        alert(f"Eradicated {bot.slug}", f"{verdict.rationale}. Positions will be closed; "
              "a research job was queued for the open slot.")
    elif verdict.action == "flag_live":
        bot.stage, bot.stage_since = BotStage.PAPER_PROVEN, now
        alert(f"{bot.slug} is LIVE-ELIGIBLE", f"{verdict.rationale}. Promote with "
              f"`autotrader promote {bot.slug} --capital <cap>` (phase 7).", priority="high")
    elif verdict.action == "unflag":
        bot.stage, bot.stage_since = BotStage.PAPER_PROBATION, now
        alert(f"{bot.slug} lost live eligibility", verdict.rationale)
