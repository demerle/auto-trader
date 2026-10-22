"""Static safety lint for strategy source code (defense in depth, not a sandbox).

Agent-written strategies must pass this before they are loaded at all; they
are additionally executed in a resource-limited subprocess at runtime.
"""

import ast

ALLOWED_IMPORTS = {
    "math", "statistics", "dataclasses", "typing", "collections", "itertools", "functools",
    "datetime", "numpy", "pandas", "pydantic",
    "autotrader.indicators", "autotrader.strategies.base", "autotrader.config.universes",
    "autotrader.options.strategy", "autotrader.options.helpers",
    "autotrader.options.ml",  # gold_scar's packaged direction model (read-only weights, numpy inference)
}
FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "breakpoint", "exit", "quit", "memoryview",
}
# I/O entry points on numpy/pandas objects.
FORBIDDEN_ATTRS = {
    "load", "save", "savez", "savez_compressed", "fromfile", "tofile", "loadtxt", "savetxt",
    "genfromtxt", "memmap", "system", "popen", "eval", "query",
}
ALLOWED_TO_ATTRS = {"to_numpy", "to_list", "to_dict", "to_frame", "to_series", "to_period",
                    "to_timestamp", "to_datetime", "to_timedelta", "to_numeric"}


def _module_allowed(module: str) -> bool:
    return any(module == ok or module.startswith(ok + ".") for ok in ALLOWED_IMPORTS)


def lint_source(source: str) -> list[str]:
    """Return a list of violations; empty means the source passed."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error: {exc}"]

    problems: list[str] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _module_allowed(alias.name):
                    problems.append(f"line {line}: import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module or not _module_allowed(node.module):
                problems.append(f"line {line}: import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"line {line}: use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("__") and attr.endswith("__"):
                problems.append(f"line {line}: dunder attribute '{attr}' is not allowed")
            elif attr in FORBIDDEN_ATTRS or attr.startswith("read_") or (
                attr.startswith("to_") and attr not in ALLOWED_TO_ATTRS
            ):
                problems.append(f"line {line}: attribute '{attr}' (I/O) is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append(f"line {line}: global/nonlocal state is not allowed")
    return problems
