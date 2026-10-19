"""Push alerts to ntfy and/or a Discord webhook; always logged. Never raises."""

import logging

import httpx

from autotrader.config import get_settings

log = logging.getLogger("autotrader.alerts")


def alert(title: str, message: str, priority: str = "default") -> None:
    log.warning("ALERT %s: %s", title, message)
    s = get_settings()
    try:
        if s.ntfy_url:
            httpx.post(s.ntfy_url, content=message.encode(), timeout=10,
                       headers={"Title": title, "Priority": priority, "Tags": "chart_with_upwards_trend"})
        if s.discord_webhook:
            httpx.post(s.discord_webhook, json={"content": f"**{title}**\n{message}"}, timeout=10)
    except Exception:  # noqa: BLE001 - alerts are best effort
        log.exception("alert delivery failed")
