"""Alerts: always logged and stored for the dashboard, pushed to ntfy and/or a Discord
webhook when configured. Never raises."""

import logging
import time

import httpx

from autotrader.config import get_settings

log = logging.getLogger("autotrader.alerts")

DISCORD_CONTENT_MAX = 2000
DISCORD_EMBED_MAX = 4096
DISCORD_CHUNK = 3900  # per embed, leaving room under the 6000-char message total


def alert(title: str, message: str, priority: str = "default") -> None:
    log.warning("ALERT %s: %s", title, message)
    _record(title, message, priority)
    _deliver(title, message, priority)


def _record(title: str, message: str, priority: str) -> None:
    try:
        from autotrader.db.models import Alert
        from autotrader.db.session import session_factory

        with session_factory()() as s:
            s.add(Alert(title=title[:200], message=message, priority=priority))
            s.commit()
    except Exception:  # noqa: BLE001 - the feed is best effort
        log.debug("could not store alert", exc_info=True)


def _deliver(title: str, message: str, priority: str) -> None:
    s = get_settings()
    try:
        if s.ntfy_url:
            httpx.post(s.ntfy_url, content=message.encode(), timeout=10,
                       headers={"Title": title, "Priority": priority, "Tags": "chart_with_upwards_trend"})
        if s.discord_webhook:
            text = f"**{title}**\n{message}"
            if len(text) > DISCORD_CONTENT_MAX:
                text = text[:DISCORD_CONTENT_MAX - 1] + "…"
            post_discord(s.discord_webhook, {"content": text})
    except Exception:  # noqa: BLE001 - alerts are best effort
        log.exception("alert delivery failed")


def post_discord(url: str, payload: dict, tries: int = 3) -> None:
    """POST to a Discord webhook, waiting out rate limits (HTTP 429)."""
    for attempt in range(tries):
        r = httpx.post(url, json=payload, timeout=10)
        if r.status_code == 429 and attempt < tries - 1:
            try:
                wait = float(r.json().get("retry_after", 1.0))
            except ValueError:
                wait = 1.0
            time.sleep(min(max(wait, 0.0), 10.0))
            continue
        r.raise_for_status()
        return


def split_text(text: str, limit: int = DISCORD_CHUNK) -> list[str]:
    """Split on line boundaries into chunks of at most `limit` chars."""
    chunks, cur = [], ""
    for line in text.splitlines():
        while len(line) > limit:  # a single overlong line
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if cur and len(cur) + 1 + len(line) > limit:
            chunks.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    return chunks or [""]


def post_report(title: str, body_md: str, url: str | None = None) -> bool:
    """Post a long markdown text to Discord as embeds, one per message. False if not configured."""
    hook = get_settings().discord_webhook
    if not hook:
        return False
    chunks = split_text(body_md)
    for i, chunk in enumerate(chunks):
        embed: dict = {"description": chunk, "color": 0x2F81F7}
        if i == 0:
            embed["title"] = title[:256]
            if url:
                embed["url"] = url
        if len(chunks) > 1:
            embed["footer"] = {"text": f"{i + 1}/{len(chunks)}"}
        post_discord(hook, {"embeds": [embed]})
    return True
