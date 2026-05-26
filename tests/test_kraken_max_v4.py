from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KRAKEN_MAX = Path(__file__).resolve().parents[1] / "Kraken Max"
if str(KRAKEN_MAX) not in sys.path:
    sys.path.insert(0, str(KRAKEN_MAX))

from regime import AdvancedRegimeEngine, HurstRegimeModel, VarianceRatioRegimeModel  # noqa: E402
from execution import set_bracket_prices  # noqa: E402
from config import CONFIG  # noqa: E402
from data import load_open_interest_csv, load_funding_csv  # noqa: E402
from kraken_ops import AlertManager  # noqa: E402
from risk import erc_weights, allocate_erc_notionals  # noqa: E402
from core import FeatureCache  # noqa: E402
from risk import PositionRisk  # noqa: E402
from data import SentimentSnapshot, adjust_deployment_cap  # noqa: E402


def test_v4_config_flags():
    assert CONFIG.enable_brackets is True
    assert CONFIG.use_erc_sizing is True
    assert CONFIG.use_advanced_regime is True


def test_hurst_and_vr_update():
    h = HurstRegimeModel(window=100)
    for i in range(600):
        h.update("BTCUSD", 100.0 + i * 0.01)
    assert 0.0 <= h.hurst("BTCUSD") <= 1.0
    v = VarianceRatioRegimeModel(min_samples=50)
    for i in range(600):
        v.update("BTCUSD", 100.0 + np.sin(i / 10))
    assert v.regime("BTCUSD") in {"trend", "meanrev", "random"}


def test_advanced_regime_classify():
    engine = AdvancedRegimeEngine()
    for i in range(600):
        engine.update_btc_bar(100.0 + i * 0.02, 99.0)
    snap = SentimentSnapshot(fear_greed=0.6, btc_dominance=0.5, funding_proxy=0.3)
    reg = engine.classify_advanced(
        btc_features={"trend_quality": 0.05, "mom_21d": 0.08, "mom_7d": 0.03},
        breadth=0.65,
        median_rv=0.4,
        sentiment=snap,
    )
    assert reg.name in {"bull", "neutral", "bear", "chaos"}
    assert "H" in reg.micro_regime


def test_erc_weights_sum_to_one():
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.01], [0.01, 0.06, 0.02], [0.01, 0.02, 0.08]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    w = erc_weights(cov)
    assert abs(sum(w.values()) - 1.0) < 0.01


def test_allocate_erc_notionals():
    cache = FeatureCache()
    rng = np.random.default_rng(1)
    for t in ["BTCUSD", "ETHUSD", "SOLUSD"]:
        for i in range(80):
            c = 100 + i
            cache._bars[t].append(
                {"open": c, "high": c + 1, "low": c - 1, "close": c + rng.normal(0, 0.1), "volume": 1000}
            )
    out = allocate_erc_notionals(["BTCUSD", "ETHUSD", "SOLUSD"], cache, 1000.0, 0.9)
    assert abs(sum(out.values()) - 900.0) < 50.0


def test_bracket_prices_set():
    st = PositionRisk(100.0, __import__("datetime").datetime.now(__import__("datetime").timezone.utc), 2.0, 100.0)
    set_bracket_prices(st, 100.0, 2.0)
    assert st.stop_price is not None and st.take_profit_price is not None
    assert st.stop_price < 100.0 < st.take_profit_price


def test_open_interest_csv():
    df = load_open_interest_csv()
    assert not df.empty


def test_funding_csv_v4():
    df = load_funding_csv()
    assert not df.empty


def test_sentiment_oi_stress_cap():
    snap = SentimentSnapshot(fear_greed=0.6, btc_dominance=0.5, funding_proxy=0.2, open_interest_stress=0.5)
    cap = adjust_deployment_cap(0.9, snap, "bull")
    assert cap < 0.9


class _FakeAlgo:
    def GetParameter(self, key):
        return ""

    def Debug(self, msg):
        pass


def test_alert_manager_no_urls():
    mgr = AlertManager(_FakeAlgo())
    mgr.notify("TEST", "hello")
