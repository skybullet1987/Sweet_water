from __future__ import annotations

import json
import ssl
from datetime import date
from urllib.request import urlopen

try:  # pragma: no cover
    from AlgorithmImports import *  # noqa: F401,F403
except Exception:  # pragma: no cover
    pass


class StablecoinLiquidityOverlay:
    CACHE_KEY = "stablecoin_liquidity_v1"
    URLS = {
        "usdt": "https://api.coingecko.com/api/v3/coins/tether/market_chart?vs_currency=usd&days=14&interval=daily",
        "usdc": "https://api.coingecko.com/api/v3/coins/usd-coin/market_chart?vs_currency=usd&days=14&interval=daily",
    }

    def __init__(self, algo=None) -> None:
        self.algo = algo
        self._weekly_pct = 0.0
        self._last_update: date | None = None
        self._load_cache()

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))

    def _debug(self, msg: str) -> None:
        if self.algo is not None and hasattr(self.algo, "Debug"):
            self.algo.Debug(msg)

    def _load_cache(self) -> None:
        try:
            if self.algo is not None and hasattr(self.algo, "ObjectStore") and self.algo.ObjectStore.ContainsKey(self.CACHE_KEY):
                payload = json.loads(self.algo.ObjectStore.Read(self.CACHE_KEY))
                self._weekly_pct = float(payload.get("weekly_pct", 0.0))
                d = payload.get("last_update")
                if d:
                    y, m, dd = [int(x) for x in str(d).split("-")]
                    self._last_update = date(y, m, dd)
        except Exception:
            pass

    def _save_cache(self) -> None:
        try:
            if self.algo is not None and hasattr(self.algo, "ObjectStore"):
                self.algo.ObjectStore.Save(
                    self.CACHE_KEY, json.dumps({"weekly_pct": self._weekly_pct, "last_update": str(self._last_update)})
                )
        except Exception:
            pass

    @staticmethod
    def _fetch_json(url: str):
        with urlopen(url, timeout=4, context=ssl.create_default_context()) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _fetch_weekly_pct(self) -> float:
        totals = []
        for _, url in self.URLS.items():
            data = self._fetch_json(url)
            mc = [float(x[1]) for x in data.get("market_caps", [])]
            if len(mc) < 8:
                raise ValueError("insufficient market cap points")
            totals.append(mc)
        combined = [sum(vals) for vals in zip(*totals)]
        latest = combined[-1]
        week_ago = combined[-8]
        if week_ago <= 0:
            return 0.0
        return (latest / week_ago) - 1.0

    def update(self, now=None) -> None:
        today = getattr(now, "date", lambda: date.today())()
        if self._last_update == today:
            return
        try:
            self._weekly_pct = float(self._fetch_weekly_pct())
            self._last_update = today
            self._save_cache()
        except Exception as exc:  # pragma: no cover
            self._debug(f"stablecoin_liquidity fallback: {type(exc).__name__}:{exc!r}")
            self._last_update = today

    def multiplier(self) -> float:
        return self._clamp(1.0 + 5.0 * float(self._weekly_pct), 0.5, 1.5)
