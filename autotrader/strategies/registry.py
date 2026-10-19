"""Discover and load strategy plugins by name and version.

Builtin strategies are imported. Generated (agent-written) strategies are never
imported here: they are registered from static metadata and run in a sandbox.
"""

import importlib
import inspect
import pkgutil
from pathlib import Path

from autotrader.strategies.base import Strategy
from autotrader.strategies.lint import lint_source

BUILTIN = "autotrader.strategies.builtin"
GENERATED_DIR = Path(__file__).parent / "generated"


class StrategyLintError(ValueError):
    pass


def _strategies_in(module) -> list[type[Strategy]]:
    return [
        obj for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, Strategy) and obj is not Strategy and not inspect.isabstract(obj)
        and obj.__module__ == module.__name__
    ]


def discover() -> dict[str, type[Strategy]]:
    """Map "name@version" -> Strategy class for every builtin and generated plugin."""
    from autotrader.sandbox.strategy import sandboxed_class

    found: dict[str, type[Strategy]] = {}

    def add(cls: type[Strategy]) -> None:
        key = f"{cls.name}@{cls.version}"
        if key in found:
            raise ValueError(f"duplicate strategy {key}")
        found[key] = cls

    pkg = importlib.import_module(BUILTIN)
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        source = (Path(pkg.__path__[0]) / f"{info.name}.py").read_text()
        if problems := lint_source(source):
            raise StrategyLintError(f"{BUILTIN}.{info.name}: " + "; ".join(problems))
        for cls in _strategies_in(importlib.import_module(f"{BUILTIN}.{info.name}")):
            add(cls)
    for path in sorted(GENERATED_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        if problems := lint_source(path.read_text()):
            raise StrategyLintError(f"generated/{path.name}: " + "; ".join(problems))
        add(sandboxed_class(path))
    return found


def is_sandboxed(cls: type[Strategy]) -> bool:
    return hasattr(cls, "source_path")


def get(name: str, version: str | None = None) -> type[Strategy]:
    """Look up a strategy; without a version, the highest version wins."""
    strategies = discover()
    if version is not None:
        return strategies[f"{name}@{version}"]
    matches = [cls for cls in strategies.values() if cls.name == name]
    if not matches:
        raise KeyError(f"unknown strategy '{name}'")
    return max(matches, key=lambda c: tuple(int(p) if p.isdigit() else p for p in c.version.split(".")))
