"""Generated strategies run out of process and must behave exactly like in-process code."""

import re
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select

from autotrader.backtest.costs import ZERO_COSTS
from autotrader.backtest.engine import BacktestConfig, run_backtest
from autotrader.broker.fake import FakeBroker
from autotrader.data.calendar import bar_close_time
from autotrader.db.models import Bot, EquitySnapshot
from autotrader.engine.arena import Arena
from autotrader.sandbox import SandboxError, SandboxProcess, run_once, sandboxed_class, static_meta
from autotrader.sandbox.runner import _env
from autotrader.strategies.builtin.stoch_momentum import StochMomentum
from tests.conftest import SYMS
from tests.test_arena import LOOSE, FakeArchive, TestStoch, make_db

BUILTIN = Path(__file__).parent.parent / "autotrader" / "strategies" / "builtin"


def generated_copy(tmp_path: Path, builtin: str = "stoch_momentum", name: str = "stoch_sandboxed",
                   extra: str = "") -> Path:
    src = (BUILTIN / f"{builtin}.py").read_text()
    src = re.sub(r'^    universe = .*$', f"    universe = {SYMS!r}", src, count=1, flags=re.M)
    src = re.sub(r'^    name = .*$', f'    name = "{name}"', src, count=1, flags=re.M)
    path = tmp_path / f"{name}.py"
    path.write_text(src + extra)
    return path


BAD_ON_BAR = '''
from autotrader.strategies.base import Strategy


class Bad(Strategy):
    name = "bad"
    interval = "1d"
    universe = ("AAA",)

    def on_bar(self, ctx):
        {body}
'''


def bad_strategy(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "bad.py"
    path.write_text(BAD_ON_BAR.replace("{body}", body))
    return path


def test_static_meta_reads_literals_without_running_code(tmp_path):
    meta = static_meta(generated_copy(tmp_path).read_text())
    assert meta["name"] == "stoch_sandboxed" and meta["interval"] == "1d"
    assert meta["universe"] == SYMS
    with pytest.raises(ValueError, match="literal"):
        static_meta("from x import Strategy\nU='a'\nclass A(Strategy):\n    name = U\n"
                    "    interval = '1d'\n    universe = 'core'\n")


def test_sandboxed_backtest_equals_in_process(tmp_path, universe):
    path = generated_copy(tmp_path)
    params = {"max_positions": 2}
    expected = run_backtest(TestStoch(params), universe).metrics()
    got = run_once(path, "backtest", params=params, bars=universe)["metrics"]
    assert got == pytest.approx(expected, rel=1e-12, nan_ok=True)
    # Per-bar RPC through the proxy (the engine's path) matches too.
    proxy = sandboxed_class(path)(params)
    try:
        assert run_backtest(proxy, universe).metrics() == pytest.approx(expected, rel=1e-12,
                                                                         nan_ok=True)
    finally:
        proxy.close()


def test_engine_with_sandboxed_strategy_matches_backtest(tmp_path, history):
    sessions, _ = make_db(tmp_path)
    broker = FakeBroker()
    proxy = sandboxed_class(generated_copy(tmp_path))({"max_positions": 2})
    arena = Arena(sessions, broker, FakeArchive(history), LOOSE, ZERO_COSTS,
                  strategy_loader=lambda bot: proxy)
    days = history["AAA"].index
    try:
        for ts in days:
            broker.fill_all({s: df.loc[ts, "open"] for s, df in history.items()})
            arena.tick("1d", now=bar_close_time(ts, "1d"))
    finally:
        proxy.close()
    expected = run_backtest(TestStoch({"max_positions": 2}), history,
                            BacktestConfig(initial_cash=10_000), ZERO_COSTS)
    with sessions() as s:
        snaps = s.scalars(select(EquitySnapshot).order_by(EquitySnapshot.ts)).all()
    live = pd.Series([x.equity for x in snaps], index=days[: len(snaps)])
    assert live.to_numpy() == pytest.approx(expected.equity.to_numpy(), rel=1e-9)


def test_lint_failure_refuses_to_load(tmp_path):
    path = generated_copy(tmp_path, extra="\nimport os\n")
    with pytest.raises(SandboxError, match="import of 'os'"):
        SandboxProcess(path)


def test_hung_strategy_is_killed(tmp_path, universe):
    with SandboxProcess(bad_strategy(tmp_path, "while True:\n            pass")) as sb:
        with pytest.raises(SandboxError, match="timed out"):
            sb.call("on_bar", 1.0, ctx=None)
        assert not sb.alive


def test_memory_bomb_is_contained(tmp_path):
    body = "import numpy as np\n        return {'AAA': float(np.ones(10**10).sum())}"
    with SandboxProcess(bad_strategy(tmp_path, body), memory_mb=1024) as sb:
        with pytest.raises(SandboxError, match="MemoryError|died"):
            sb.call("on_bar", 30.0, ctx=None)


def test_prints_do_not_corrupt_protocol(tmp_path):
    with SandboxProcess(bad_strategy(tmp_path, "print('hi' * 1000)\n        return {'AAA': 0.5}")) as sb:
        assert sb.call("on_bar", 10.0, ctx=None)["targets"] == {"AAA": 0.5}


def test_bad_targets_rejected(tmp_path):
    with SandboxProcess(bad_strategy(tmp_path, "return {'AAA': float('nan')}")) as sb:
        with pytest.raises(SandboxError, match="bad target"):
            sb.call("on_bar", 10.0, ctx=None)


def test_worker_env_has_no_secrets(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "x")
    env = _env("/tmp/x")
    assert not any("ALPACA" in k or "DATABASE" in k for k in env)


def test_engine_pauses_bot_when_sandbox_fails(tmp_path, history):
    sessions, (bot_id,) = make_db(tmp_path)
    proxy = sandboxed_class(bad_strategy(tmp_path, "raise_it = 1 / 0"))()
    arena = Arena(sessions, FakeBroker(), FakeArchive(history), LOOSE, ZERO_COSTS,
                  strategy_loader=lambda bot: proxy)
    ts = history["AAA"].index[-1]
    report = arena.tick("1d", now=bar_close_time(ts, "1d"))
    proxy.close()
    assert "sandboxed strategy failed" in str(report)
    with sessions() as s:
        assert s.get(Bot, bot_id).paused_until is not None
