"""Run one agent job: a per-job MCP server, a deny-by-default opencode config, and
`opencode run` as an unprivileged process with a scrubbed environment."""

import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from autotrader.config import get_settings
from autotrader.db.models import AgentJob, utcnow
from autotrader.evaluator.policy import Policy
from autotrader.mcp_server.server import JobServer
from autotrader.mcp_server.tools import AgentTools
from autotrader.orchestrator.config import AgentsConfig

log = logging.getLogger(__name__)
PROMPTS = Path(__file__).parent / "prompts"


def permissions(role: str) -> list[dict]:
    """Last matching rule wins. The workspace is opencode's project directory, so
    `external_directory: deny` confines file tools to it."""
    def rule(action: str, effect: str, resource: str = "*") -> dict:
        return {"action": action, "resource": resource, "effect": effect}

    rules = [rule("*", "allow"),
             rule("shell", "deny"), rule("bash", "deny"),
             rule("external_directory", "deny"),
             rule("task", "deny"), rule("skill", "deny"), rule("browser", "deny"),
             rule("question", "deny"),  # nobody is there to answer
             rule("read", "deny", "*.env"), rule("read", "deny", "*.env.*")]
    if role not in ("researcher", "options_researcher"):
        rules += [rule("webfetch", "deny"), rule("websearch", "deny")]
    if role in ("auditor", "supervisor", "options_auditor"):
        rules += [rule("edit", "deny"), rule("write", "deny"), rule("patch", "deny")]
    return rules


def opencode_config(role: str, model: str, mcp_url: str, steps: int = 120) -> dict:
    return {
        "$schema": "https://opencode.ai/config.json",
        # Pin everything to the one configured model: the main model and the "small" model
        # opencode uses for background calls (titles, summaries, compaction).
        "model": model,
        "small_model": model,
        "permissions": permissions(role),
        "agents": {role: {"model": model, "mode": "primary", "steps": steps,
                          "description": f"autotrader {role}",
                          "system": (PROMPTS / f"{role}.md").read_text(),
                          "permissions": permissions(role)}},
        # opencode 2.0.32 only connects servers in the legacy `mcp.<name>` shape (not
        # `mcp.servers`). Its request timeout defaults to 5s; gate runs take minutes.
        "mcp": {"autotrader": {"type": "remote", "url": mcp_url, "oauth": False,
                               "timeout": 1_200_000}},
        "autoupdate": False,
        "share": "disabled",
    }


def build_prompt(job: AgentJob, workspace: Path) -> str:
    target = job.target or {}
    if job.agent == "researcher":
        head = (f"Target interval: {target.get('interval', '1d')}. "
                f"Slot reason: {target.get('reason', 'open slot')}.\n")
    elif job.agent in ("auditor", "options_auditor"):
        head = f"Audit submission_id {job.submission_id}.\n"
    elif job.agent == "options_researcher":
        head = (f"Options gauntlet seat: {target.get('reason') or 'replace ' + str(target.get('replaces'))}. "
                f"Week of {target.get('week', 'now')}.\n")
    else:
        head = f"Review bot `{target.get('slug')}`.\n"
    return f"{head}Your workspace (current directory): {workspace}\n\n{job.prompt}"


@dataclass
class RunResult:
    exit_code: int | None
    timed_out: bool
    cost_usd: float = 0.0
    tokens: dict = field(default_factory=dict)
    session_id: str | None = None
    error: str | None = None
    error_type: str | None = None


def parse_events(lines) -> RunResult:
    """Extract session id, cost and tokens from `opencode run --format json` events.

    Cost-bearing objects (steps, messages) can be re-emitted as they update, so the
    last value per object id is used.
    """
    by_id: dict[str, tuple[float, dict]] = {}
    anon: list[tuple[float, dict]] = []
    session_id, error, error_type = None, None, None

    def visit(obj) -> None:
        if isinstance(obj, dict):
            cost = obj.get("cost")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                tokens = obj.get("tokens") if isinstance(obj.get("tokens"), dict) else {}
                key = obj.get("id") or obj.get("messageID")
                if key:
                    by_id[str(key)] = (float(cost), tokens)
                else:
                    anon.append((float(cost), tokens))
                return
            for v in obj.values():
                visit(v)
        elif isinstance(obj, list):
            for v in obj:
                visit(v)

    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        session_id = session_id or ev.get("sessionID")
        if ev.get("type") == "error":
            err = ev.get("error") or {}
            error = err.get("message") if isinstance(err, dict) else str(err)
            error_type = err.get("type") if isinstance(err, dict) else None
        visit(ev)
    entries = list(by_id.values()) + anon
    tokens: dict[str, float] = {}
    for _, t in entries:
        for k, v in t.items():
            if isinstance(v, (int, float)):
                tokens[k] = tokens.get(k, 0) + v
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float)):
                        tokens[f"{k}_{k2}"] = tokens.get(f"{k}_{k2}", 0) + v2
    return RunResult(None, False, sum(c for c, _ in entries), tokens, session_id, error, error_type)


def session_cost(session_id: str, home: Path, xdg_config: Path) -> tuple[float, dict] | None:
    """Total cost and tokens of an opencode session, from `opencode session export`."""
    s = get_settings()
    try:
        out = subprocess.run(_priv_drop(s.agent_user) + [s.opencode_bin, "session", "export",
                                                         "--standalone", session_id],
                             env=_env(home, xdg_config), cwd=xdg_config.parent, capture_output=True,
                             text=True, timeout=120, check=True).stdout
        info = json.loads(out).get("info", {})
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        log.warning("could not read cost of session %s: %s", session_id, exc)
        return None
    tokens: dict[str, float] = {}
    for k, v in (info.get("tokens") or {}).items():
        if isinstance(v, dict):
            tokens |= {f"{k}_{k2}": v2 for k2, v2 in v.items() if isinstance(v2, (int, float))}
        elif isinstance(v, (int, float)):
            tokens[k] = v
    return float(info.get("cost") or 0.0), tokens


def _env(home: Path, xdg_config: Path) -> dict[str, str]:
    """No database URL, no broker keys: only what opencode needs."""
    env = {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": str(home),
           "LANG": "C.UTF-8", "XDG_CONFIG_HOME": str(xdg_config), "NO_COLOR": "1"}
    for k in ("XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME", "TZ"):
        if k in os.environ:
            env[k] = os.environ[k]
    return env


def _priv_drop(user: str | None) -> list[str]:
    if not user:
        return []
    return ["setpriv", f"--reuid={user}", f"--regid={user}", "--init-groups",
            "--inh-caps=-all", "--ambient-caps=-all", "--"]


def run_opencode(job: AgentJob, model: str, workspace: Path, mcp_url: str, timeout_sec: float,
                 log_path: Path) -> RunResult:
    s = get_settings()
    xdg = workspace / ".xdg"  # empty: the host's global opencode config must not apply
    xdg.mkdir(exist_ok=True)
    xdg.chmod(0o2770)  # opencode (the agent user) writes its config dir here
    (workspace / "opencode.json").write_text(json.dumps(opencode_config(job.agent, model, mcp_url),
                                                        indent=1))
    cmd = _priv_drop(s.agent_user) + [
        s.opencode_bin, "run", "--standalone", "--format", "json", "-m", model,
        "--agent", job.agent, "--title", f"autotrader-job-{job.id}", build_prompt(job, workspace)]
    home = Path(s.agent_home or os.environ.get("HOME", "/tmp"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with open(log_path, "w") as out:
        proc = subprocess.Popen(cmd, cwd=workspace, env=_env(home, xdg), stdout=out,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
        timed_out = False
        try:
            proc.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, 15)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, 9)
                proc.wait()
    res = parse_events(log_path.read_text(errors="replace").splitlines())
    res.exit_code, res.timed_out = proc.returncode, timed_out
    if res.session_id:  # `run --format json` carries no cost; the session export does
        cost = session_cost(res.session_id, home, xdg)
        if cost is not None:
            res.cost_usd, res.tokens = cost
    log.info("job %s (%s, %s) finished in %.0fs: exit %s, $%.4f", job.id, job.agent, model,
             time.monotonic() - started, proc.returncode, res.cost_usd)
    return res


AGENT_GROUP = "agents"


def agent_sessions(default: sessionmaker[Session]) -> sessionmaker[Session]:
    """Tools act through the least-privilege database role when one is configured."""
    url = get_settings().agent_database_url
    if not url or ":@" in url:  # unset, or no password configured
        return default
    from autotrader.db.session import session_factory

    return session_factory(url)


def prepare_workspace(job_id: int) -> Path:
    ws = (get_settings().workspace_dir / f"job_{job_id}").resolve()
    ws.mkdir(parents=True, exist_ok=True)
    try:  # in Docker: owned by us, group `agents`, so the agent user may write only here
        import grp

        os.chown(ws, -1, grp.getgrnam(AGENT_GROUP).gr_gid)
        ws.chmod(0o2770)
    except (KeyError, PermissionError):
        pass
    return ws


_installed_auth: dict[str, str] = {}


def install_auth() -> None:
    """Import the mounted OpenCode Go credential (from `autotrader agents auth-export`) into
    the agent user's opencode store. Re-imports only when the file changes."""
    import hashlib

    s = get_settings()
    src = s.opencode_auth_file
    if not (src and s.agent_home and src.exists()):
        return
    data = src.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if _installed_auth.get("sha") == digest:
        return
    tmp = s.agent_home / ".local" / "share" / "opencode" / "import.json"
    tmp.write_bytes(data)
    tmp.chmod(0o640)
    try:
        subprocess.run(_priv_drop(s.agent_user) + [s.opencode_bin, "auth", "import", "--standalone",
                                                   str(tmp)],
                       env=_env(s.agent_home, s.agent_home / ".config"), cwd=s.agent_home,
                       check=True, capture_output=True, timeout=60)
    finally:
        tmp.unlink(missing_ok=True)
    _installed_auth["sha"] = digest
    log.info("imported OpenCode Go credentials for the agent user")


def run_job(sessions: sessionmaker[Session], job_id: int, archive, cfg: AgentsConfig, policy: Policy,
            runner=run_opencode, now=None) -> AgentJob:
    """Claim, run and record one queued job. `runner` is swappable for tests."""
    with sessions() as s:
        job = s.get(AgentJob, job_id)
        if job.status != "queued":
            return job
        job.status, job.attempts, job.started_at = "running", job.attempts + 1, utcnow()
        job.model = cfg.model_for(job.agent, job.attempts)
        s.commit()
    log_path = get_settings().agent_log_dir / f"job_{job.id}.jsonl"
    try:
        ws = prepare_workspace(job.id)
        install_auth()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tools = AgentTools(job.id, job.agent, ws, agent_sessions(sessions), archive, cfg, policy,
                           now=now)
        with JobServer(tools) as server:
            res = runner(job, job.model, ws, server.url, cfg.timeouts_min.get(job.agent, 30) * 60,
                         log_path)
    except Exception as exc:  # noqa: BLE001 - record the failure, keep the orchestrator alive
        log.exception("job %s crashed", job.id)
        res = RunResult(None, False, error=f"{type(exc).__name__}: {exc}")
    with sessions() as s:
        job = s.get(AgentJob, job_id)
        job.cost_usd = (job.cost_usd or 0.0) + res.cost_usd
        job.tokens, job.session_id = res.tokens, res.session_id
        job.transcript_path = str(log_path)
        job.finished_at = utcnow()
        ok = res.exit_code == 0 and not res.timed_out and not res.error
        job.result = {"exit_code": res.exit_code, "timed_out": res.timed_out, "error": res.error,
                      "error_type": res.error_type, "attempt": job.attempts}
        job.status = "succeeded" if ok else "failed"
        s.commit()
        return job


def cleanup_workspace(job_id: int) -> None:
    shutil.rmtree(get_settings().workspace_dir / f"job_{job_id}" / ".xdg", ignore_errors=True)




def final_text(lines) -> str:
    """The agent's text output, best effort across event shapes."""
    texts = []
    for line in lines:
        try:
            ev = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        part = ev.get("part") if isinstance(ev.get("part"), dict) else ev
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            texts.append(part["text"])
        elif ev.get("type") == "error":
            texts.append(f"ERROR: {ev.get('error')}")
    return "\n".join(texts[-5:])
