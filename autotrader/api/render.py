"""Server-side rendering helpers: SVG line charts, a tiny markdown subset, number formats.

Charts are inline SVG so the dashboard needs no JavaScript libraries.
"""

import html
import re

import pandas as pd
from jinja2 import Undefined
from markupsafe import Markup

PALETTE = ("var(--accent)", "var(--muted)", "var(--warn)", "var(--good)")
DISTINCT = ("#2f6fde", "#e8590c", "#2b8a3e", "#ae3ec9", "#c92a2a", "#0c8599", "#e67700", "#5c7cfa")


def line_chart(series: list[tuple[str, list[tuple[str, float]]]], width: int = 760,
               height: int = 260, money: bool = True, percent: bool = False,
               time_fmt: str = "%Y-%m-%d", distinct: bool = False) -> Markup:
    """series: [(label, [(iso_ts, value), ...]), ...] sharing one time axis.
    distinct: every series a solid line in its own colour (for races between bots)."""
    pts = [(pd.Timestamp(t), v) for _, s in series for t, v in s if v is not None]
    if len(pts) < 2:
        return Markup('<p class="muted">Not enough data for a chart yet.</p>')
    t0 = min(p[0] for p in pts).value
    t1 = max(p[0] for p in pts).value
    lo = min(p[1] for p in pts)
    hi = max(p[1] for p in pts)
    if hi == lo:
        hi, lo = hi + 1, lo - 1
    pad_l, pad_r, pad_t, pad_b = 64, 12, 12, 28
    w, h = width - pad_l - pad_r, height - pad_t - pad_b

    def x(ts: pd.Timestamp) -> float:
        return pad_l + (ts.value - t0) / max(t1 - t0, 1) * w

    def y(v: float) -> float:
        return pad_t + (hi - v) / (hi - lo) * h

    fmt = (lambda v: f"{v * 100:+.0f}%") if percent else (lambda v: f"${v:,.0f}") if money \
        else (lambda v: f"{v:,.2f}")
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
           f'preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">']
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        out.append(f'<line class="grid" x1="{pad_l}" x2="{width - pad_r}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text class="axis" x="{pad_l - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{fmt(v)}</text>')
    for frac, anchor in ((0, "start"), (0.5, "middle"), (1, "end")):
        ts = pd.Timestamp(int(t0 + (t1 - t0) * frac), tz="UTC")
        if time_fmt != "%Y-%m-%d":
            ts = ts.tz_convert("America/New_York")
        out.append(f'<text class="axis" x="{pad_l + w * frac:.1f}" y="{height - 8}" '
                   f'text-anchor="{anchor}">{ts.strftime(time_fmt)}</text>')
    if percent and lo < 0 < hi:
        out.append(f'<line class="zero" x1="{pad_l}" x2="{width - pad_r}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>')
    legend = []
    for i, (label, s) in enumerate(series):
        clean = [(pd.Timestamp(t), v) for t, v in s if v is not None]
        if len(clean) < 2:
            continue
        color = DISTINCT[i % len(DISTINCT)] if distinct else PALETTE[i % len(PALETTE)]
        dashed = not distinct and i > 0
        d = " ".join(f"{'M' if j == 0 else 'L'}{x(t):.1f},{y(v):.1f}" for j, (t, v) in enumerate(clean))
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{2 if i == 0 or distinct else 1.5}"'
                   f'{" stroke-dasharray=\"4 3\"" if dashed else ""}><title>{html.escape(label)}</title></path>')
        legend.append(f'<span class="key"><i style="background:{color}"></i>{html.escape(label)}</span>')
    out.append("</svg>")
    return Markup(f'<figure class="chartbox">{"".join(out)}<figcaption>{"".join(legend)}</figcaption></figure>')


_BOLD = re.compile(r"\*\*(.+?)\*\*")


def _inline(text: str) -> str:
    return _BOLD.sub(r"<strong>\1</strong>", html.escape(text))


def markdown(text: str) -> Markup:
    """Headings (#, ##), bullets (-), numbered items, **bold**, paragraphs. Everything escaped."""
    out, in_ul, in_ol = [], False, False

    def close():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
        if in_ol:
            out.append("</ol>")
        in_ul = in_ol = False

    for line in (text or "").splitlines():
        s = line.rstrip()
        if m := re.match(r"^(#{1,3})\s+(.*)$", s):
            close()
            level = len(m.group(1)) + 1
            out.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
        elif m := re.match(r"^\s*[-*]\s+(.*)$", s):
            if not in_ul:
                close()
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{_inline(m.group(1))}</li>")
        elif m := re.match(r"^\s*\d+\.\s+(.*)$", s):
            if not in_ol:
                close()
                out.append("<ol>")
                in_ol = True
            out.append(f"<li>{_inline(m.group(1))}</li>")
        elif not s.strip():
            close()
        else:
            close()
            out.append(f"<p>{_inline(s)}</p>")
    close()
    return Markup("\n".join(out))


def _missing(v) -> bool:
    return v is None or isinstance(v, Undefined)


def money(v, signed: bool = False) -> str:
    if _missing(v):
        return "–"
    s = f"${abs(v):,.0f}"
    if v < 0:
        return "-" + s
    return ("+" + s) if signed else s


def pct(v, signed: bool = True) -> str:
    if _missing(v):
        return "–"
    return f"{v:+.2%}" if signed else f"{v:.2%}"


def fnum(v, nd: int = 2) -> str:
    if _missing(v):
        return "–"
    return f"{v:,.{nd}f}"


def when(v) -> str:
    """ISO timestamp -> 'YYYY-MM-DD HH:MM' in New York time."""
    if _missing(v) or not v:
        return "–"
    ts = pd.Timestamp(v)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    return f"{ts.tz_convert('America/New_York'):%Y-%m-%d %H:%M}"


def sign_class(v) -> str:
    if _missing(v):
        return ""
    return "pos" if v > 0 else "neg" if v < 0 else ""


def transcript(lines, max_body: int = 2500) -> list[dict]:
    """opencode `--format json` events -> readable entries (text, tool calls, errors).

    Parts can be re-emitted as they update, so the last version of each part id wins.
    """
    import json

    entries: dict[str, dict] = {}
    order: list[str] = []

    def put(key: str, entry: dict) -> None:
        if key not in entries:
            order.append(key)
        entries[key] = entry

    def clip(s) -> str:
        s = s if isinstance(s, str) else json.dumps(s, indent=1, default=str)
        return s if len(s) <= max_body else s[:max_body] + f"\n… ({len(s) - max_body} more chars)"

    for n, line in enumerate(lines):
        try:
            ev = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            if str(line).strip():
                put(f"raw{n}", {"kind": "raw", "title": "output", "body": clip(str(line))})
            continue
        if not isinstance(ev, dict):
            continue
        if ev.get("type") == "error":
            err = ev.get("error")
            msg = err.get("message") if isinstance(err, dict) else err
            put(f"err{n}", {"kind": "error", "title": (err or {}).get("type", "error")
                            if isinstance(err, dict) else "error", "body": clip(str(msg))})
            continue
        part = ev.get("part") if isinstance(ev.get("part"), dict) else None
        if part is None:
            continue
        key = str(part.get("id") or f"p{n}")
        kind = part.get("type")
        if kind == "text" and isinstance(part.get("text"), str):
            put(key, {"kind": "text", "title": "agent", "body": part["text"]})
        elif kind == "reasoning" and isinstance(part.get("text"), str):
            put(key, {"kind": "reasoning", "title": "thinking", "body": clip(part["text"])})
        elif kind == "tool":
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            body = "input: " + clip(state.get("input", {}))
            if state.get("output") is not None:
                body += "\n\noutput: " + clip(state.get("output"))
            if state.get("error"):
                body += "\n\nerror: " + clip(state.get("error"))
            put(key, {"kind": "tool", "title": f"{part.get('tool', 'tool')} ({state.get('status', '?')})",
                      "body": body})
    return [entries[k] for k in order]
