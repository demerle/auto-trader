"""Arena state. Bars live in the Parquet archive; everything else lives here."""

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    JSON, DateTime, Enum, Float, ForeignKey, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    type_annotation_map = {datetime: DateTime(timezone=True), dict: JSON}


class BotStage(str, enum.Enum):
    RESEARCH = "research"
    BACKTEST_PASSED = "backtest_passed"
    PAPER_PROBATION = "paper_probation"
    PAPER_PROVEN = "paper_proven"  # live-eligible, awaiting the user's approval
    LIVE = "live"
    DEMOTED = "demoted"
    PAUSED = "paused"
    ERADICATED = "eradicated"


class StrategyVersion(Base):
    __tablename__ = "strategy_versions"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    version: Mapped[str] = mapped_column(String(20))
    interval: Mapped[str] = mapped_column(String(4))
    module: Mapped[str] = mapped_column(String(200))
    code_sha: Mapped[str] = mapped_column(String(64))
    author: Mapped[str] = mapped_column(String(80), default="human")  # or agent job id
    description: Mapped[str] = mapped_column(Text, default="")
    backtest_metrics: Mapped[dict] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Bot(Base):
    __tablename__ = "bots"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(80), unique=True)
    strategy_id: Mapped[int] = mapped_column(ForeignKey("strategy_versions.id"))
    params: Mapped[dict] = mapped_column(default=dict)
    stage: Mapped[BotStage] = mapped_column(Enum(BotStage), default=BotStage.RESEARCH)
    account: Mapped[str] = mapped_column(String(40), default="paper")
    starting_capital: Mapped[float] = mapped_column(Float, default=10_000.0)
    capital: Mapped[float] = mapped_column(Float, default=10_000.0)  # current allocation
    live_cap: Mapped[float | None] = mapped_column(Float, nullable=True)  # set only by the user
    stage_since: Mapped[datetime] = mapped_column(default=utcnow)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    notes: Mapped[str] = mapped_column(Text, default="")

    strategy: Mapped[StrategyVersion] = relationship()


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    bot_id: Mapped[int] = mapped_column(ForeignKey("bots.id"), index=True)
    client_order_id: Mapped[str] = mapped_column(String(64), unique=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    account: Mapped[str] = mapped_column(String(40))
    symbol: Mapped[str] = mapped_column(String(16))
    qty: Mapped[float] = mapped_column(Float)  # signed
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="new")
    reason: Mapped[dict] = mapped_column(default=dict)  # decision inputs, for replay
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    bot_id: Mapped[int] = mapped_column(ForeignKey("bots.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(16))
    qty: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    filled_at: Mapped[datetime] = mapped_column(default=utcnow)


class EquitySnapshot(Base):
    __tablename__ = "equity_snapshots"
    __table_args__ = (UniqueConstraint("bot_id", "ts"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    bot_id: Mapped[int] = mapped_column(ForeignKey("bots.id"), index=True)
    ts: Mapped[datetime]
    equity: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    positions: Mapped[dict] = mapped_column(default=dict)


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(primary_key=True)
    bot_id: Mapped[int] = mapped_column(ForeignKey("bots.id"), index=True)
    ts: Mapped[datetime] = mapped_column(default=utcnow)
    metrics: Mapped[dict] = mapped_column(default=dict)
    decision: Mapped[str] = mapped_column(String(40))  # keep / eradicate / flag_live / demote
    rationale: Mapped[str] = mapped_column(Text, default="")


class AgentJob(Base):
    __tablename__ = "agent_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    agent: Mapped[str] = mapped_column(String(40))  # researcher / auditor / reviewer / supervisor
    model: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(20), default="queued")
    prompt: Mapped[str] = mapped_column(Text)
    bot_id: Mapped[int | None] = mapped_column(ForeignKey("bots.id"), nullable=True)
    est_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    transcript_path: Mapped[str | None] = mapped_column(String(300), nullable=True)
    result: Mapped[dict] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
