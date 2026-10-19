"""A Strategy that runs its code in a sandbox worker.

`sandboxed_class(path)` builds a Strategy subclass from a file's *static*
metadata (literal class attributes, read with `ast`), so registering or listing
a generated strategy never executes it. Instances forward `features` and
`on_bar` to a persistent worker.
"""

import ast
import hashlib
from pathlib import Path
from typing import ClassVar

import pandas as pd

from autotrader.sandbox.protocol import decode_frame
from autotrader.sandbox.runner import SandboxProcess
from autotrader.strategies.base import Strategy, StrategyContext, Targets

META_FIELDS = ("name", "version", "interval", "universe", "description")
ON_BAR_TIMEOUT = 5.0
FEATURES_TIMEOUT = 60.0


def static_meta(source: str) -> dict:
    """Literal metadata of the single Strategy subclass in `source`."""
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and any(isinstance(b, ast.Name) and b.id == "Strategy" for b in n.bases)]
    if len(classes) != 1:
        raise ValueError(f"expected exactly one `class X(Strategy)`, found {len(classes)}")
    meta: dict = {}
    for stmt in classes[0].body:
        targets = stmt.targets if isinstance(stmt, ast.Assign) else (
            [stmt.target] if isinstance(stmt, ast.AnnAssign) and stmt.value is not None else [])
        for t in targets:
            if isinstance(t, ast.Name) and t.id in META_FIELDS:
                try:
                    meta[t.id] = ast.literal_eval(stmt.value)
                except ValueError:
                    raise ValueError(f"class attribute `{t.id}` must be a literal") from None
    for required in ("name", "interval", "universe"):
        if required not in meta:
            raise ValueError(f"strategy class must define `{required}` as a literal")
    if isinstance(meta["universe"], list):
        meta["universe"] = tuple(meta["universe"])
    meta["version"] = str(meta.get("version", "1"))
    meta.setdefault("description", "")
    return meta


class _Params(dict):
    def model_dump(self, **_) -> dict:
        return dict(self)


class SandboxedStrategy(Strategy):
    source_path: ClassVar[str]
    code_sha: ClassVar[str]

    def __init__(self, params=None):
        self._requested = dict(params or {})
        self._sb: SandboxProcess | None = None
        self._meta: dict | None = None

    def _worker(self) -> SandboxProcess:
        if self._sb is None or not self._sb.alive:
            self._sb = SandboxProcess(self.source_path, self._requested)
            self._meta = self._sb.meta
        return self._sb

    @property
    def params(self) -> _Params:  # type: ignore[override]
        self._worker()
        return _Params(self._meta["params"])

    @property
    def lookback(self) -> int:
        self._worker()
        return int(self._meta["lookback"])

    def features(self, df: pd.DataFrame) -> pd.DataFrame | None:
        return decode_frame(self._worker().call("features", FEATURES_TIMEOUT, df=df)["df"])

    def on_bar(self, ctx: StrategyContext) -> Targets:
        return self._worker().call("on_bar", ON_BAR_TIMEOUT, ctx=ctx)["targets"]

    def close(self) -> None:
        if self._sb is not None:
            self._sb.close()
            self._sb = None


def sandboxed_class(path: str | Path) -> type[SandboxedStrategy]:
    path = Path(path).resolve()
    source = path.read_text()
    meta = static_meta(source)
    attrs = {k: meta[k] for k in META_FIELDS} | {
        "source_path": str(path), "code_sha": hashlib.sha256(source.encode()).hexdigest(),
        "__module__": f"sandboxed.{path.stem}",
    }
    return type(f"Sandboxed_{meta['name']}", (SandboxedStrategy,), attrs)
