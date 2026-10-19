"""Sandbox worker: loads one strategy file and serves requests over stdin/stdout.

Started by `runner.SandboxProcess` as `python -I -m autotrader.sandbox.worker` with
a scrubbed environment and resource limits. Requests are pickles from the trusted
parent; responses are JSON (DataFrames as Arrow IPC) so the parent never
unpickles anything this process produced.
"""

import importlib.util
import json
import math
import os
import pickle
import sys
import traceback
from pathlib import Path

import pandas as pd

from autotrader.backtest.engine import BacktestConfig, run_backtest
from autotrader.backtest.walkforward import walk_forward
from autotrader.sandbox.protocol import encode_frame, frame_length, read_exact, write_frame
from autotrader.strategies.base import Strategy
from autotrader.strategies.lint import lint_source
from autotrader.strategies.validation import validate


def _json_default(obj):
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    return str(obj)


def load_strategy_class(path: str, base: type = Strategy) -> type:
    source = Path(path).read_text()
    if problems := lint_source(source):
        raise ValueError("lint failed: " + "; ".join(problems))
    import hashlib

    mod_name = "sandboxed_" + hashlib.sha256(source.encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    found = [obj for obj in vars(module).values()
             if isinstance(obj, type) and issubclass(obj, base) and obj is not base
             and obj.__module__ == mod_name and not getattr(obj, "__abstractmethods__", None)]
    if len(found) != 1:
        raise ValueError(f"expected exactly one {base.__name__} subclass in {path}, found {len(found)}")
    return found[0]


def _strategy(state: dict, params) -> Strategy:
    return state["cls"](params if params is not None else state["params"])


def op_load(state: dict, req: dict) -> dict:
    cls = load_strategy_class(req["path"])
    strategy = cls(req.get("params") or {})
    state.update(cls=cls, strategy=strategy, params=strategy.params.model_dump(mode="json"))
    return {"meta": {"name": cls.name, "version": cls.version, "interval": cls.interval,
                     "universe": cls.universe, "description": cls.description,
                     "lookback": strategy.lookback, "symbols": list(strategy.symbols),
                     "params": state["params"]}}


def op_features(state: dict, req: dict) -> dict:
    out = state["strategy"].features(req["df"])
    if out is not None and not isinstance(out, pd.DataFrame):
        raise TypeError("features() must return a DataFrame or None")
    return {"df": encode_frame(out)}


def op_on_bar(state: dict, req: dict) -> dict:
    targets = state["strategy"].on_bar(req["ctx"])
    if targets is None:
        return {"targets": None}
    if not isinstance(targets, dict):
        raise TypeError("on_bar() must return a dict or None")
    out = {}
    for k, v in targets.items():
        v = float(v)
        if not isinstance(k, str) or not math.isfinite(v):
            raise ValueError(f"bad target {k!r}: {v!r}")
        out[k] = v
    return {"targets": out}


def op_validate(state: dict, req: dict) -> dict:
    return {"problems": validate(_strategy(state, req.get("params")), req["bars"])}


def op_backtest(state: dict, req: dict) -> dict:
    res = run_backtest(_strategy(state, req.get("params")), req["bars"],
                       BacktestConfig(**req.get("config", {})), start=req.get("start"),
                       end=req.get("end"))
    m = res.metrics()
    return {"metrics": m, "trades": len(res.trades),
            "start": res.equity.index[0] if len(res.equity) else None,
            "end": res.equity.index[-1] if len(res.equity) else None,
            "final_positions": res.final_positions, "params": res.params}


def op_walkforward(state: dict, req: dict) -> dict:
    wf = walk_forward(state["cls"], req["bars"], req.get("grid"), train_bars=req["train_bars"],
                      test_bars=req["test_bars"], base_params=req.get("params"),
                      prior_trial_sharpes=req.get("prior_trial_sharpes", ()))
    return {"metrics": wf.metrics, "n_trials": wf.n_trials, "trial_sharpes": wf.trial_sharpes,
            "folds": [{"test_start": f.test_start, "test_end": f.test_end,
                       "best_params": f.best_params, "train_sharpe": f.train_score,
                       "test_sharpe": f.test.metrics()["sharpe"] if len(f.test.equity) > 2 else None}
                      for f in wf.folds]}


def op_gate(state: dict, req: dict) -> dict:
    from autotrader.evaluator.gate import run_gate
    from autotrader.evaluator.policy import GateRules

    res = run_gate(state["cls"], req["bars"], GateRules.model_validate(req["rules"]),
                   req.get("grid"), req.get("params"), holdout_start=req.get("holdout_start"),
                   prior_trial_sharpes=req.get("prior_trial_sharpes", ()))
    return {"strategy": res.strategy, "passed": res.passed, "reasons": res.reasons,
            "metrics": res.metrics, "holdout": res.holdout}


# ----- option strategies (the gauntlet) -----

def op_opt_load(state: dict, req: dict) -> dict:
    from autotrader.options.strategy import OptionStrategy

    cls = load_strategy_class(req["path"], OptionStrategy)
    strategy = cls(req.get("params") or {})
    state.update(cls=cls, strategy=strategy, params=strategy.params.model_dump(mode="json"))
    return {"meta": {"name": cls.name, "version": cls.version, "underlyings": list(cls.underlyings),
                     "description": cls.description, "thesis": cls.thesis, "params": state["params"]}}


def op_opt_tick(state: dict, req: dict) -> dict:
    from autotrader.options.strategy import encode_targets

    return {"targets": encode_targets(state["strategy"].on_tick(req["ctx"]))}


def op_opt_simulate(state: dict, req: dict) -> dict:
    """Run the whole simulation in here, so the strategy never leaves the sandbox."""
    from autotrader.options.sim import simulate
    from autotrader.options.strategy import decode_targets, encode_targets

    strategy = state["cls"](req.get("params") or state["params"])
    h = req["history"]

    def on_tick(ctx):  # round-trip through the wire format so live and sim see the same validation
        return decode_targets(encode_targets(strategy.on_tick(ctx)))

    res = simulate(on_tick, tuple(strategy.underlyings), h["bars5m"], h["daily"], h["vix"], h["vix1d"],
                   req["start"], req["end"], req["cfg"], req["limits"], capital=req.get("capital"))
    return {"summary": res.summary(), "min_worst_equity": res.min_worst_equity,
            "trades": res.trades[-200:], "errors": res.errors}


OPS = {"opt_load": op_opt_load, "opt_tick": op_opt_tick, "opt_simulate": op_opt_simulate,
       "load": op_load, "features": op_features, "on_bar": op_on_bar, "validate": op_validate,
       "backtest": op_backtest, "walkforward": op_walkforward, "gate": op_gate}


def main() -> None:
    proto_in = sys.stdin.buffer
    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # whatever the strategy prints must not corrupt the protocol
    sys.stdout = sys.stderr
    state: dict = {}
    while True:
        try:
            header = read_exact(proto_in, 4)
        except EOFError:
            return
        req = pickle.loads(read_exact(proto_in, frame_length(header)))
        try:
            resp = {"ok": True, **OPS[req["op"]](state, req)}
        except Exception as exc:  # noqa: BLE001 - report every failure to the parent
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc(limit=6)}
        write_frame(proto_out, json.dumps(resp, default=_json_default).encode())


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()
