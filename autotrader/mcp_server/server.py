"""Per-job MCP server for opencode agents.

The orchestrator (trusted, holds the database credentials) serves each job's
tools over streamable HTTP on localhost, under an unguessable path. The agent
process gets only that URL: no database, no data files, no keys.
"""

import secrets
import socket
import threading
import time

import uvicorn
from mcp.server.mcpserver import MCPServer

from autotrader.mcp_server.tools import AgentTools

INSTRUCTIONS = """Autotrader arena tools. You research, write and test trading strategies.
Start with get_strategy_api. All code you write runs in a sandbox; every backtest you run counts
against your trial budget and deflates your score."""


def build_server(tools: AgentTools) -> MCPServer:
    server = MCPServer("autotrader", instructions=INSTRUCTIONS)
    for fn in tools.functions():
        server.add_tool(fn, name=fn.__name__, description=(fn.__doc__ or "").strip())
    return server


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class JobServer:
    """Serve one job's tools in a background thread until stop()."""

    def __init__(self, tools: AgentTools, host: str = "127.0.0.1"):
        self.token = secrets.token_urlsafe(24)
        self.host, self.port = host, _free_port()
        self.path = f"/{self.token}/mcp"
        app = build_server(tools).streamable_http_app(streamable_http_path=self.path, host=host)
        self._server = uvicorn.Server(uvicorn.Config(app, host=host, port=self.port,
                                                     log_level="warning", lifespan="on"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}{self.path}"

    def start(self, timeout: float = 15.0) -> "JobServer":
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._server.started:
            if time.monotonic() > deadline or not self._thread.is_alive():
                raise RuntimeError("MCP server failed to start")
            time.sleep(0.05)
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)

    def __enter__(self) -> "JobServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
