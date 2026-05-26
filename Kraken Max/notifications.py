from __future__ import annotations

import json
import urllib.request
from typing import Any


def _post_json(url: str, payload: dict[str, Any]) -> bool:
    if not url or not url.startswith("http"):
        return False
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "KrakenMax/4.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def send_telegram(webhook_url: str, message: str) -> bool:
    """Telegram bot API: pass full URL like https://api.telegram.org/bot<TOKEN>/sendMessage?chat_id=<ID> as webhook_url with POST body, or use bot token + chat in algo params."""
    if "api.telegram.org" in webhook_url and "sendMessage" in webhook_url:
        return _post_json(webhook_url, {"text": message[:4000]})
    return _post_json(webhook_url, {"message": message[:4000]})


def send_discord(webhook_url: str, message: str) -> bool:
    return _post_json(webhook_url, {"content": message[:2000]})


class AlertManager:
    def __init__(self, algo) -> None:
        self.algo = algo
        self.telegram_url = ""
        self.discord_url = ""
        self._last_alert_key = ""
        try:
            self.telegram_url = str(algo.GetParameter("telegram_webhook") or "")
        except Exception:
            self.telegram_url = ""
        try:
            self.discord_url = str(algo.GetParameter("discord_webhook") or "")
        except Exception:
            self.discord_url = ""

    def notify(self, event: str, detail: str, *, dedupe_key: str | None = None) -> None:
        if dedupe_key and dedupe_key == self._last_alert_key:
            return
        self._last_alert_key = dedupe_key or ""
        msg = f"[Kraken Max] {event}\n{detail}"
        if hasattr(self.algo, "Debug"):
            self.algo.Debug(msg[:200])
        if self.telegram_url:
            send_telegram(self.telegram_url, msg)
        if self.discord_url:
            send_discord(self.discord_url, msg)
