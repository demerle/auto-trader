import pytest

from autotrader.backtest.engine import run_backtest
from autotrader.strategies import registry
from autotrader.strategies.lint import lint_source
from autotrader.strategies.builtin.stoch_momentum import StochMomentum


def test_registry_discovers_builtin():
    assert registry.get("stoch_momentum") is StochMomentum
    assert "stoch_momentum@1" in registry.discover()


def test_stoch_momentum_trades_long_only(universe):
    res = run_backtest(StochMomentum({"max_positions": 2}), universe)
    assert res.trades, "expected some trades on 400 random-walk bars"
    assert all(q > 0 for q in res.final_positions.values())


def test_params_are_validated():
    with pytest.raises(Exception):
        StochMomentum({"not_a_param": 1})


@pytest.mark.parametrize("src", [
    "import os",
    "import requests",
    "from subprocess import run",
    "x = open('f')",
    "x = eval('1')",
    "import pandas as pd\ndf = pd.read_csv('x')",
    "def f(df):\n    df.to_csv('x')",
    "x = ().__class__.__subclasses__()",
    "from . import secrets",
    "import numpy as np\nnp.load('a')",
])
def test_lint_rejects_unsafe(src):
    assert lint_source(src)


def test_lint_accepts_builtin_style():
    src = (
        "import numpy as np\nimport pandas as pd\n"
        "from autotrader.indicators import rsi\n"
        "from autotrader.strategies.base import Strategy\n"
        "def f(df):\n    return df.close.to_numpy().mean()\n"
    )
    assert lint_source(src) == []


def test_builtin_strategies_pass_validation(universe):
    from autotrader.strategies.validation import validate

    for cls in registry.discover().values():
        assert validate(cls(), universe) == [], cls


def test_validation_catches_lookahead_features(universe):
    import pandas as pd

    from autotrader.strategies.base import Strategy
    from autotrader.strategies.validation import validate

    class Cheater(Strategy):
        name, interval, universe = "cheat", "1d", ("AAA",)

        def features(self, df):
            return pd.DataFrame({"fwd": df.close.shift(-1) / df.close - 1})

        def on_bar(self, ctx):
            f = ctx.features["AAA"].fwd.iloc[-1]
            return {"AAA": 1.0 if f > 0 else 0.0}

    problems = validate(Cheater(), {"AAA": universe["AAA"]})
    assert any("lookahead" in p for p in problems)
