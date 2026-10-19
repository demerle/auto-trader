"""Option strategies: hand-written builtins (in-process) and agent-written files (sandboxed).

Generated strategies are never imported here; only their literal metadata is read
(with ast), and their code runs in a sandbox worker.
"""

import ast
import hashlib
import importlib
import pkgutil
from pathlib import Path
from typing import ClassVar

from autotrader.options import builtin
from autotrader.options.strategy import OptionContext, OptionStrategy, Targets, decode_targets
from autotrader.sandbox.runner import SandboxProcess
from autotrader.strategies.lint import lint_source

GENERATED_DIR = Path(__file__).parent / "generated"
META_FIELDS = ("name", "version", "underlyings", "description", "thesis")
TICK_TIMEOUT = 5.0


def static_meta(source: str) -> dict:
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
               and any(isinstance(b, ast.Name) and b.id == "OptionStrategy" for b in n.bases)]
    if len(classes) != 1:
        raise ValueError(f"expected exactly one `class X(OptionStrategy)`, found {len(classes)}")
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
    for required in ("name", "underlyings"):
        if required not in meta:
            raise ValueError(f"the strategy class must define `{required}` as a literal")
    meta["underlyings"] = tuple(meta["underlyings"])
    meta["version"] = str(meta.get("version", "1"))
    meta.setdefault("description", "")
    meta.setdefault("thesis", "")
    return meta


class SandboxedOptionStrategy(OptionStrategy):
    source_path: ClassVar[str]
    code_sha: ClassVar[str]

    def __init__(self, params=None):
        self._requested = dict(params or {})
        self._sb: SandboxProcess | None = None
        self._meta: dict | None = None

    def _worker(self) -> SandboxProcess:
        if self._sb is None or not self._sb.alive:
            self._sb = SandboxProcess(self.source_path, self._requested, load_op="opt_load")
            self._meta = self._sb.meta
        return self._sb

    @property
    def params(self):  # type: ignore[override]
        self._worker()
        return dict(self._meta["params"])

    def on_tick(self, ctx: OptionContext) -> Targets:
        return decode_targets(self._worker().call("opt_tick", TICK_TIMEOUT, ctx=ctx)["targets"])

    def close(self) -> None:
        if self._sb is not None:
            self._sb.close()
            self._sb = None


def sandboxed_class(path: str | Path) -> type[SandboxedOptionStrategy]:
    path = Path(path).resolve()
    source = path.read_text()
    meta = static_meta(source)
    attrs = {k: meta[k] for k in META_FIELDS} | {
        "source_path": str(path), "code_sha": hashlib.sha256(source.encode()).hexdigest(),
        "__module__": f"sandboxed_options.{path.stem}",
    }
    return type(f"Sandboxed_{meta['name']}", (SandboxedOptionStrategy,), attrs)


def is_sandboxed(cls) -> bool:
    return hasattr(cls, "source_path")


def source_of(cls) -> str:
    if is_sandboxed(cls):
        return Path(cls.source_path).read_text()
    import inspect

    return inspect.getsource(inspect.getmodule(cls))


def discover() -> dict[str, type[OptionStrategy]]:
    """name@version -> class, for builtins and generated files."""
    found: dict[str, type[OptionStrategy]] = {}
    for mod in pkgutil.iter_modules(builtin.__path__):
        path = Path(builtin.__path__[0]) / f"{mod.name}.py"
        if problems := lint_source(path.read_text()):
            raise ValueError(f"builtin option strategy {mod.name} fails lint: {problems}")
        module = importlib.import_module(f"{builtin.__name__}.{mod.name}")
        for obj in vars(module).values():
            if isinstance(obj, type) and issubclass(obj, OptionStrategy) and obj is not OptionStrategy \
                    and obj.__module__ == module.__name__:
                found[f"{obj.name}@{obj.version}"] = obj
    if GENERATED_DIR.exists():
        for path in sorted(GENERATED_DIR.glob("*.py")):
            if path.name.startswith("_") or lint_source(path.read_text()):
                continue
            cls = sandboxed_class(path)
            key = f"{cls.name}@{cls.version}"
            if key in found:
                raise ValueError(f"duplicate option strategy {key}")
            found[key] = cls
    return found


def get(name: str, version: str | None = None) -> type[OptionStrategy]:
    found = discover()
    if version is not None:
        return found[f"{name}@{version}"]
    matches = sorted((k for k in found if k.split("@")[0] == name),
                     key=lambda k: [int(x) if x.isdigit() else x for x in k.split("@")[1].split(".")])
    if not matches:
        raise KeyError(name)
    return found[matches[-1]]
