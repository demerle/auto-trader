"""Discover and load strategy plugins by name and version."""

import importlib
import inspect
import pkgutil
from pathlib import Path

from autotrader.strategies.base import Strategy
from autotrader.strategies.lint import lint_source

PACKAGES = ("autotrader.strategies.builtin", "autotrader.strategies.generated")


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
    found: dict[str, type[Strategy]] = {}
    for pkg_name in PACKAGES:
        pkg = importlib.import_module(pkg_name)
        for info in pkgutil.iter_modules(pkg.__path__):
            if info.name.startswith("_"):
                continue
            source = (Path(pkg.__path__[0]) / f"{info.name}.py").read_text()
            if problems := lint_source(source):
                raise StrategyLintError(f"{pkg_name}.{info.name}: " + "; ".join(problems))
            module = importlib.import_module(f"{pkg_name}.{info.name}")
            for cls in _strategies_in(module):
                key = f"{cls.name}@{cls.version}"
                if key in found:
                    raise ValueError(f"duplicate strategy {key}")
                found[key] = cls
    return found


def get(name: str, version: str | None = None) -> type[Strategy]:
    """Look up a strategy; without a version, the highest version wins."""
    strategies = discover()
    if version is not None:
        return strategies[f"{name}@{version}"]
    matches = [cls for cls in strategies.values() if cls.name == name]
    if not matches:
        raise KeyError(f"unknown strategy '{name}'")
    return max(matches, key=lambda c: tuple(int(p) if p.isdigit() else p for p in c.version.split(".")))
