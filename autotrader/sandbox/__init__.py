"""Run agent-written strategy code out of process, with resource limits.

Generated strategies are never imported into a trusted process. The worker gets
its inputs as pickles (trusted -> untrusted) and answers in JSON / Arrow, which
the parent can parse without executing anything the worker controls.
"""

from autotrader.sandbox.runner import SandboxError, SandboxProcess, run_once
from autotrader.sandbox.strategy import SandboxedStrategy, sandboxed_class, static_meta

__all__ = ["SandboxError", "SandboxProcess", "SandboxedStrategy", "run_once", "sandboxed_class",
           "static_meta"]
