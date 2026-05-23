"""Smoke tests for GoldDayTrading.

These tests do **not** hit the network or any LLM — they construct
synthetic OHLCV data and run the offline pipeline path so CI / a
plain `pytest` invocation always passes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from golddaytrading.agents.risk_manager import compute_guardrails
from golddaytrading.config import GDTConfig, load_config
from golddaytrading.dataflows.indicators import (
    compute_indicators,
    indicator_summary_block,
)
from golddaytrading.llm.client import LLMClient, build_client
from golddaytrading.sessions import classify_session, SESSIONS


def _make_synthetic_ohlcv(n: int = 240, freq_min: int = 15) -> pd.DataFrame:
    """Build a synthetic 15-minute OHLCV series with a gentle drift."""
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    idx = pd.date_range(
        end=end, periods=n, freq=f"{freq_min}min", tz="UTC"
    )
    rng = np.random.default_rng(42)
    drift = np.linspace(2300, 2380, n)
    noise = rng.normal(0, 4, n).cumsum() * 0.2
    close = drift + noise
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.5, 2.0, n)
    low = np.minimum(open_, close) - rng.uniform(0.5, 2.0, n)
    vol = rng.integers(100, 1000, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low,
         "Close": close, "Volume": vol},
        index=idx,
    )


def test_session_classifier_tiles_24h() -> None:
    """Every UTC hour must map to exactly one session."""
    for h in range(24):
        when = datetime(2026, 5, 23, h, 30, tzinfo=timezone.utc)
        s = classify_session(when)
        assert s.name in {sw.name for sw in SESSIONS}


def test_indicators_compute_on_synthetic_data() -> None:
    df = _make_synthetic_ohlcv()
    ind = compute_indicators(df)
    for k in ("ema20", "ema50", "ema200", "rsi14", "macd", "atr14",
              "vwap", "or", "pivots"):
        assert k in ind, f"missing indicator: {k}"
    block = indicator_summary_block(df, ind)
    assert "Last close" in block
    assert "RSI" in block
    assert "VWAP" in block


def test_offline_llm_client_returns_text() -> None:
    """The offline provider must produce a non-empty deterministic reply."""
    client = build_client("offline", "n/a")
    out = client.complete(
        "You are a test agent.",
        "- RSI(14): 55\n- MACD: bullish\n- DXY 1h: -0.10%",
    )
    assert isinstance(out, str) and out.strip()
    # No-key build_client falls back to offline regardless of provider.
    fallback = build_client("openai-fake-not-installed", "n/a")
    assert fallback.provider == "offline"


def test_load_config_defaults_to_offline_without_keys(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    cfg = load_config()
    assert cfg.llm_provider == "offline"


def test_risk_guardrails_skip_on_flat_plan() -> None:
    cfg = GDTConfig()
    rg = compute_guardrails(
        manager_report="Bias: FLAT. No trade.",
        cfg=cfg,
        upcoming_events=[],
    )
    assert not rg.approved
    assert rg.position_size_units is None


def test_risk_guardrails_size_long_setup() -> None:
    cfg = GDTConfig(
        account_usd=10_000, risk_per_trade_pct=0.5, min_rr=1.5,
    )
    plan = (
        "Bias: LONG\n"
        "- Entry trigger: 2350.00\n"
        "- Stop: 2345.00\n"
        "- TP1: 2360.00\n"
        "- TP2: 2370.00\n"
    )
    rg = compute_guardrails(plan, cfg, upcoming_events=[])
    assert rg.approved, rg.hard_block
    # 5-point risk per unit, $50 budget -> 10 units expected
    assert rg.position_size_units == pytest.approx(10.0, rel=1e-3)
    # R:R = (2360-2350) / (2350-2345) = 2.0
    assert rg.rr_ratio == pytest.approx(2.0, rel=1e-3)


def test_risk_guardrails_block_low_rr() -> None:
    cfg = GDTConfig(min_rr=2.0)
    plan = (
        "Bias: LONG\n"
        "- Entry trigger: 2350.00\n"
        "- Stop: 2345.00\n"
        "- TP1: 2356.00\n"           # only 1.2 R
        "- TP2: 2360.00\n"
    )
    rg = compute_guardrails(plan, cfg, upcoming_events=[])
    assert not rg.approved
    assert "R:R" in (rg.hard_block or "")


def test_llm_client_complete_falls_back_on_provider_error(monkeypatch) -> None:
    """A provider exception must degrade to the offline heuristic."""
    class Boom:
        @property
        def chat(self):
            class _Chat:
                @property
                def completions(self):
                    class _C:
                        def create(self, **_):
                            raise RuntimeError("boom")
                    return _C()
            return _Chat()

    client = LLMClient(provider="openai", model="gpt-4o", _impl=Boom())
    out = client.complete("Role: tester", "- RSI(14): 60")
    assert "Offline heuristic mode" in out
