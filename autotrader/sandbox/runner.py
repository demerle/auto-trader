"""Parent side of the sandbox: a resource-limited worker process per strategy file."""

import json
import os
import pickle
import resource
import selectors
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

from autotrader.sandbox.protocol import frame_length, write_frame

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY_MB = 3072


class SandboxError(RuntimeError):
    """The sandboxed code failed, timed out, or broke the protocol."""


def _limits(memory_mb: int, cpu_sec: int | None):
    def apply() -> None:
        os.setsid()
        mem = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # no file writes
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if cpu_sec:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec + 5))
    return apply


def _env(home: str) -> dict[str, str]:
    """Nothing from the parent's environment: no broker keys, no database URL."""
    return {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": home, "LANG": "C.UTF-8",
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}


class SandboxProcess:
    """One worker serving one strategy file. Use as a context manager or call close()."""

    def __init__(self, path: str | Path, params: dict | None = None, memory_mb: int = DEFAULT_MEMORY_MB,
                 cpu_sec: int | None = None, load_timeout: float = 60.0, load_op: str = "load"):
        self.path = str(Path(path).resolve())
        self._tmp = tempfile.TemporaryDirectory(prefix="sandbox-")
        self._stderr: deque[str] = deque(maxlen=60)
        self._proc = subprocess.Popen(
            [sys.executable, "-I", "-m", "autotrader.sandbox.worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=self._tmp.name, env=_env(self._tmp.name),
            preexec_fn=_limits(memory_mb, cpu_sec), close_fds=True,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self.meta: dict = self.call(load_op, load_timeout, path=self.path, params=params or {})["meta"]

    def _drain_stderr(self) -> None:
        for line in iter(self._proc.stderr.readline, b""):
            self._stderr.append(line.decode(errors="replace").rstrip())

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def stderr_tail(self, n: int = 20) -> str:
        return "\n".join(list(self._stderr)[-n:])

    def _read(self, n: int, deadline: float) -> bytes:
        out = self._proc.stdout
        fd = out.fileno()
        buf = bytearray()
        with selectors.DefaultSelector() as sel:
            sel.register(fd, selectors.EVENT_READ)
            while len(buf) < n:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not sel.select(remaining):
                    raise TimeoutError
                chunk = os.read(fd, min(n - len(buf), 1 << 20))
                if not chunk:
                    raise EOFError
                buf += chunk
        return bytes(buf)

    def call(self, op: str, timeout: float, **kwargs) -> dict:
        if not self.alive:
            raise SandboxError(f"sandbox for {self.path} is not running\n{self.stderr_tail()}")
        try:
            write_frame(self._proc.stdin, pickle.dumps({"op": op, **kwargs},
                                                        protocol=pickle.HIGHEST_PROTOCOL))
            deadline = time.monotonic() + timeout
            n = frame_length(self._read(4, deadline))
            resp = json.loads(self._read(n, deadline))
        except TimeoutError:
            self.close()
            raise SandboxError(f"{op} timed out after {timeout:.0f}s") from None
        except (EOFError, BrokenPipeError, ValueError) as exc:
            self.close()
            raise SandboxError(f"{op}: worker died ({exc!r}, exit {self._proc.returncode})\n"
                               f"{self.stderr_tail()}") from None
        if not isinstance(resp, dict) or not resp.get("ok"):
            err = resp.get("error", "bad response") if isinstance(resp, dict) else "bad response"
            raise SandboxError(f"{op}: {err}")
        return resp

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                os.killpg(self._proc.pid, 9)
            except ProcessLookupError:
                pass
            self._proc.wait(timeout=10)
        for stream in (self._proc.stdin, self._proc.stdout):
            try:
                stream.close()
            except OSError:
                pass
        self._tmp.cleanup()

    def __enter__(self) -> "SandboxProcess":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def run_once(path: str | Path, op: str, timeout: float = 900.0, params: dict | None = None,
             memory_mb: int = DEFAULT_MEMORY_MB, load_op: str = "load", **kwargs) -> dict:
    """Start a worker, run one op with a CPU limit, and tear it down."""
    with SandboxProcess(path, params, memory_mb=memory_mb, cpu_sec=int(timeout) + 60,
                        load_op=load_op) as sb:
        return sb.call(op, timeout, params=params, **kwargs)
