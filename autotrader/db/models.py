"""Arena state. Bars live in the Parquet archive; everything else lives here."""

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    MetaData,
    JSON, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    })
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


ARENA = "arena"  # the long-term, statistically judged bots
OPTIONS = "options"  # the weekly options gauntlet


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
    group: Mapped[str] = mapped_column(String(16), default=ARENA, server_default=ARENA, index=True)
    starting_capital: Mapped[float] = mapped_column(Float, default=10_000.0)
    capital: Mapped[float] = mapped_column(Float, default=10_000.0)  # current allocation
    live_cap: Mapped[float | None] = mapped_column(Float, nullable=True)  # set only by the user
    stage_since: Mapped[datetime] = mapped_column(default=utcnow)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    notes: Mapped[str] = mapped_column(Text, default="")
    # Share quantities the bot is working towards; retried each tick until filled.
    target_qty: Mapped[dict] = mapped_column(default=dict)
    paused_until: Mapped[datetime | None] = mapped_column(nullable=True)

    strategy: Mapped[StrategyVersion] = relationship()


class Order(Base):
    """An order sent to the broker by the desk; may be shared by several bots."""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(64), unique=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    account: Mapped[str] = mapped_column(String(40))
    interval: Mapped[str] = mapped_column(String(4))
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    qty: Mapped[float] = mapped_column(Float)  # signed
    # Options: one order per structure. legs = [{symbol, ratio, filled_qty, filled_avg_price}].
    legs: Mapped[list | None] = mapped_column(JSON, nullable=True)
    structure: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chase: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    order_type: Mapped[str] = mapped_column(String(12))  # market / limit
    time_in_force: Mapped[str] = mapped_column(String(8))  # opg / day
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    ref_price: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(24), default="new")
    filled_qty: Mapped[float] = mapped_column(Float, default=0.0)  # unsigned, as recorded
    filled_notional: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    done_at: Mapped[datetime | None] = mapped_column(nullable=True)

    allocations: Mapped[list["OrderAllocation"]] = relationship(back_populates="order")


class OrderAllocation(Base):
    """A bot's share of a desk order."""

    __tablename__ = "order_allocations"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    bot_id: Mapped[int] = mapped_column(ForeignKey("bots.id"), index=True)
    qty: Mapped[float] = mapped_column(Float)  # signed
    filled_qty: Mapped[float] = mapped_column(Float, default=0.0)  # signed
    reason: Mapped[dict] = mapped_column(default=dict)  # decision inputs, for replay

    order: Mapped[Order] = relationship(back_populates="allocations")


class Fill(Base):
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_id: Mapped[int | None] = mapped_column(ForeignKey("orders.id"), index=True, nullable=True)
    bot_id: Mapped[int] = mapped_column(ForeignKey("bots.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(32))
    qty: Mapped[float] = mapped_column(Float)  # signed (shares, or option contracts)
    price: Mapped[float] = mapped_column(Float)  # per share (options: premium per share, x100)
    structure: Mapped[str | None] = mapped_column(String(64), nullable=True)  # options tag
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    internal: Mapped[bool] = mapped_column(Boolean, default=False)  # crossed between bots
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
    # What the job is for: {"interval": "1d", "reason": "..."} or {"submission_id": 3}.
    target: Mapped[dict] = mapped_column(default=dict)
    submission_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # auditor jobs
    session_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)  # actual, from opencode events
    tokens: Mapped[dict] = mapped_column(default=dict)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)


class Submission(Base):
    """A strategy an agent wrote and submitted for the gate, the audit and a paper slot."""

    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("agent_jobs.id"), index=True)
    parent_strategy_id: Mapped[int | None] = mapped_column(ForeignKey("strategy_versions.id"),
                                                          nullable=True)
    name: Mapped[str] = mapped_column(String(80))
    version: Mapped[str] = mapped_column(String(20))
    interval: Mapped[str] = mapped_column(String(4))
    draft_path: Mapped[str] = mapped_column(String(300))
    code_sha: Mapped[str] = mapped_column(String(64))
    source: Mapped[str] = mapped_column(Text)  # frozen copy of what was gated
    rationale: Mapped[str] = mapped_column(Text, default="")
    params: Mapped[dict] = mapped_column(default=dict)
    # submitted -> gate_failed | audit_pending -> rejected | approved -> deployed
    status: Mapped[str] = mapped_column(String(20), default="submitted")
    reasons: Mapped[list] = mapped_column(JSON, default=list)
    gate_metrics: Mapped[dict] = mapped_column(default=dict)
    holdout_metrics: Mapped[dict] = mapped_column(default=dict)
    n_trials: Mapped[int] = mapped_column(Integer, default=1)
    audit: Mapped[dict] = mapped_column(default=dict)
    bot_id: Mapped[int | None] = mapped_column(ForeignKey("bots.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class AgentNote(Base):
    __tablename__ = "agent_notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int | None] = mapped_column(ForeignKey("agent_jobs.id"), nullable=True)
    bot_id: Mapped[int | None] = mapped_column(ForeignKey("bots.id"), nullable=True)
    topic: Mapped[str] = mapped_column(String(120))
    body: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class ToolCall(Base):
    """Every MCP tool call an agent makes: audit trail and trial counting."""

    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("agent_jobs.id"), index=True)
    tool: Mapped[str] = mapped_column(String(40))
    args: Mapped[dict] = mapped_column(default=dict)
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    trials: Mapped[int] = mapped_column(Integer, default=0)  # backtests this call ran
    sharpes: Mapped[list] = mapped_column(JSON, default=list)  # their per-period Sharpes
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    ts: Mapped[datetime] = mapped_column(default=utcnow)


class SystemState(Base):
    """Small key/value store: kill switch, account high-water mark, last tick per interval."""

    __tablename__ = "system_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(default=dict)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)


class CashFlow(Base):
    """Capital moved into (+) or out of (-) a bot by the allocator."""

    __tablename__ = "cash_flows"

    id: Mapped[int] = mapped_column(primary_key=True)
    bot_id: Mapped[int] = mapped_column(ForeignKey("bots.id"), index=True)
    ts: Mapped[datetime] = mapped_column(default=utcnow)
    amount: Mapped[float] = mapped_column(Float)
    reason: Mapped[str] = mapped_column(Text, default="")


class Alert(Base):
    """Every alert sent (also pushed to Discord/ntfy), for the dashboard's feed."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(default=utcnow, index=True)
    title: Mapped[str] = mapped_column(String(200))
    message: Mapped[str] = mapped_column(Text, default="")
    priority: Mapped[str] = mapped_column(String(16), default="default")


class Report(Base):
    """The daily report, built after the evaluation; the supervisor agent adds commentary."""

    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[str] = mapped_column(String(10), unique=True)  # ET session date
    body_md: Mapped[str] = mapped_column(Text)
    data: Mapped[dict] = mapped_column(default=dict)
    commentary: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    posted_at: Mapped[datetime | None] = mapped_column(nullable=True)
    commentary_posted_at: Mapped[datetime | None] = mapped_column(nullable=True)


class AuditLog(Base):
    """Actions taken from the dashboard."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(default=utcnow)
    action: Mapped[str] = mapped_column(String(40))
    target: Mapped[str] = mapped_column(String(120), default="")
    detail: Mapped[dict] = mapped_column(default=dict)


class WeeklyResult(Base):
    """One options bot's week in the gauntlet: opened Monday, scored and maybe culled Friday."""

    __tablename__ = "weekly_results"
    __table_args__ = (UniqueConstraint("week", "bot_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    week: Mapped[str] = mapped_column(String(10), index=True)  # Monday's date (ET), e.g. 2026-10-26
    bot_id: Mapped[int] = mapped_column(ForeignKey("bots.id"), index=True)
    start_equity: Mapped[float] = mapped_column(Float)
    end_equity: Mapped[float | None] = mapped_column(Float, nullable=True)
    ret: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown: Mapped[float | None] = mapped_column(Float, nullable=True)
    trades: Mapped[int] = mapped_column(Integer, default=0)
    sessions: Mapped[int] = mapped_column(Integer, default=0)  # sessions it traded this week
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running/survived/culled/exempt/busted
    busted: Mapped[bool] = mapped_column(Boolean, default=False)
    autopsy: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime] = mapped_column(default=utcnow)
    scored_at: Mapped[datetime | None] = mapped_column(nullable=True)

    bot: Mapped[Bot] = relationship()
