"""Regression tests for the P0 bug fixes.

These cover:

1. ``risk_manager`` user-prompt construction must include account /
   risk / blackout context even when the deterministic guardrail
   produces a SKIP (None values for units / risk_dollars / R:R).
   The legacy code used a misplaced ``if/else`` ternary that, due to
   Python operator-precedence, stripped the entire prompt prefix
   when any field was ``None``.

2. ``_extract_level`` must parse thousands-separator and currency
   prefixes that LLMs love to emit (``$2,345.60``).

3. ``session_vwap`` must anchor to ``anchor_hour_utc`` (default
   22:00 UTC NY-close) and produce a sensible value when ``Volume``
   is all-zero (the yfinance ``XAUUSD=X`` case).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from golddaytrading.agents import risk_manager as rm
from golddaytrading.config import GDTConfig
from golddaytrading.dataflows.indicators import (
    compute_indicators,
    session_vwap,
)
from golddaytrading.llm.client import LLMClient


# ---------------------------------------------------------------------------
# P0.1 — risk_manager prompt construction never loses context
# ---------------------------------------------------------------------------


class _SpyLLM:
    """LLM stub that captures the (system, user) it was called with."""

    def __init__(self) -> None:
        self.system: str | None = None
        self.user: str | None = None

    def complete(self, system: str, user: str, **_: object) -> str:
        self.system = system
        self.user = user
        return "VERDICT: SKIP"


def test_risk_manager_prompt_includes_full_context_when_skip() -> None:
    """With a FLAT plan the guardrail returns Nones for all fields.

    The user prompt must still contain the account-size / risk-cap /
    upcoming-events sections (regression for the precedence bug).
    """
    cfg = GDTConfig(account_usd=12_345.0, risk_per_trade_pct=0.4,
                    daily_loss_limit_pct=2.5, min_rr=1.5)
    ctx = {
        "research_plan": "Bias: FLAT. No setup.",
        "upcoming_events": [],
        "calendar_block": "_(no calendar)_",
        "realised_daily_loss_usd": 0.0,
    }
    spy = _SpyLLM()
    rm.risk_manager(ctx, spy, cfg)

    assert spy.user is not None
    assert "Research-manager plan" in spy.user
    assert "Account & risk parameters" in spy.user
    assert "Account size: $12,345" in spy.user
    assert "Max risk per trade: 0.40%" in spy.user
    assert "Daily loss cap: 2.50%" in spy.user
    assert "Min R:R: 1.50" in spy.user
    # Even when the field is None it must show "N/A", not delete the
    # surrounding context.
    assert "- Position size (units): N/A" in spy.user
    assert "- Risk on the trade ($): N/A" in spy.user
    assert "- R:R (vs TP1): N/A" in spy.user
    assert "Upcoming events" in spy.user


def test_risk_manager_prompt_with_approved_setup() -> None:
    """When all guardrail fields populate, all numeric lines render."""
    cfg = GDTConfig(account_usd=10_000.0, risk_per_trade_pct=0.5, min_rr=1.5)
    plan = (
        "Bias: LONG\n"
        "- Entry trigger: 2350.00\n"
        "- Stop: 2345.00\n"
        "- TP1: 2360.00\n"
        "- TP2: 2370.00\n"
    )
    ctx = {
        "research_plan": plan,
        "upcoming_events": [],
        "calendar_block": "_(no calendar)_",
    }
    spy = _SpyLLM()
    rm.risk_manager(ctx, spy, cfg)
    assert spy.user is not None
    assert "Position size (units): 10.00" in spy.user
    assert "Risk on the trade ($): 50.00" in spy.user
    assert "R:R (vs TP1): 2.00" in spy.user
    assert "Hard block: none" in spy.user


# ---------------------------------------------------------------------------
# P0.2 — _extract_level handles commas, $, signed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, snippet, expected",
    [
        ("entry", "Entry trigger: 2345.6", 2345.6),
        ("entry", "**Entry trigger:** $2,345.60", 2345.60),
        ("stop", "- Stop: $2,340.00", 2340.0),
        ("stop", "Stop **2,345**", 2345.0),
        ("tp1", "TP1 = 2,360.00", 2360.0),
        ("tp2", "TP2 around 2370", 2370.0),
        ("entry", "Entry: ~2,345", 2345.0),
        ("stop", "Stop loss at $1,995.50", 1995.50),
    ],
)
def test_extract_level_parses_common_llm_formats(label: str, snippet: str,
                                                 expected: float) -> None:
    got = rm._extract_level(snippet, label)
    assert got is not None
    assert got == pytest.approx(expected, rel=1e-6)


def test_extract_level_returns_none_on_unrelated_text() -> None:
    assert rm._extract_level("FLAT — no trade rationale.", "entry") is None


def test_compute_guardrails_handles_dollar_comma_levels() -> None:
    """The pipeline-level integration: a plan with $2,345 must size."""
    cfg = GDTConfig(account_usd=10_000, risk_per_trade_pct=0.5, min_rr=1.5)
    plan = (
        "Bias: LONG\n"
        "- Entry trigger: $2,350.00\n"
        "- Stop: $2,345.00\n"
        "- TP1: $2,360.00\n"
        "- TP2: $2,370.00\n"
    )
    rg = rm.compute_guardrails(plan, cfg, upcoming_events=[])
    assert rg.approved, rg.hard_block
    assert rg.position_size_units == pytest.approx(10.0, rel=1e-3)
    assert rg.rr_ratio == pytest.approx(2.0, rel=1e-3)


# ---------------------------------------------------------------------------
# P0.3 — session VWAP anchoring + zero-volume fallback
# ---------------------------------------------------------------------------


def _build_two_session_df(anchor_hour: int) -> pd.DataFrame:
    """Synthetic 30m bars covering ~2 sessions on either side of anchor."""
    # Start the first session 1h before the anchor to ensure the
    # anchor crossing falls inside the index.
    start = datetime(2026, 5, 22, anchor_hour - 1, 0, tzinfo=timezone.utc)
    idx = pd.date_range(start, periods=20, freq="30min", tz="UTC")
    rng = np.random.default_rng(7)
    close = 2300 + np.linspace(0, 5, 20) + rng.normal(0, 0.2, 20).cumsum()
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low,
         "Close": close, "Volume": np.zeros(20)},
        index=idx,
    )


def test_session_vwap_resets_at_configured_anchor_hour() -> None:
    """The cumulative mean must restart on the bar that crosses anchor."""
    anchor = 22
    df = _build_two_session_df(anchor_hour=anchor)
    v = session_vwap(df, anchor_hour_utc=anchor)
    # First bar of the new session should equal that bar's typical price.
    new_session_mask = df.index.hour == anchor
    assert new_session_mask.any(), "test fixture didn't span the anchor"
    first_new = v[new_session_mask].iloc[0]
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    assert first_new == pytest.approx(typical[new_session_mask].iloc[0], rel=1e-9)


def test_session_vwap_zero_volume_falls_back_to_typical_mean() -> None:
    """With Volume==0 throughout, VWAP must be the running typical-price mean.

    Crucially it must NOT be pinned to the first bar (the bug that
    happened with the legacy ``Volume.replace(0, NaN).fillna(1)`` path).
    """
    df = _build_two_session_df(anchor_hour=22)
    assert (df["Volume"] == 0).all()
    v = session_vwap(df, anchor_hour_utc=22)

    # Hand-compute typical-price mean *within* the latest session.
    sid = (df.index - pd.Timedelta(hours=22)).normalize().asi8
    last_session_mask = sid == sid[-1]
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    expected_last = typical[last_session_mask].mean()
    assert v.iloc[-1] == pytest.approx(expected_last, rel=1e-9)


def test_session_vwap_with_real_volume_is_volume_weighted() -> None:
    """Hand-computable check: VWAP must follow sum(P*V)/sum(V), not simple mean."""
    # Single clean session (start at the anchor) with two halves that
    # differ enormously in both price and volume — so VWAP and simple
    # mean must be far apart.
    start = datetime(2026, 5, 22, 22, 0, tzinfo=timezone.utc)
    idx = pd.date_range(start, periods=10, freq="30min", tz="UTC")
    close = np.r_[np.full(5, 2300.0), np.full(5, 2400.0)]
    high = close + 0.0   # typical = (H+L+C)/3 == close when H=L=C
    low = close + 0.0
    vol = np.r_[np.full(5, 100.0), np.full(5, 1000.0)]
    df = pd.DataFrame(
        {"Open": close, "High": high, "Low": low,
         "Close": close, "Volume": vol},
        index=idx,
    )
    v = session_vwap(df, anchor_hour_utc=22)
    # Expected at last bar:
    #   (2300*100*5 + 2400*1000*5) / (100*5 + 1000*5)
    #   = 13_150_000 / 5_500 = 2390.909...
    expected = (2300 * 100 * 5 + 2400 * 1000 * 5) / (100 * 5 + 1000 * 5)
    assert v.iloc[-1] == pytest.approx(expected, rel=1e-9)
    # Simple mean would be 2350 — VWAP must clearly differ.
    assert abs(v.iloc[-1] - 2350.0) > 30.0


def test_compute_indicators_emits_new_indicator_keys() -> None:
    """P1.9 regression: bb20 / stoch / weekly_pivots / swings present."""
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    idx = pd.date_range(end=end, periods=400, freq="15min", tz="UTC")
    rng = np.random.default_rng(1)
    close = 2300 + np.linspace(0, 80, 400) + rng.normal(0, 4, 400).cumsum() * 0.2
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.5, 2.0, 400)
    low = np.minimum(open_, close) - rng.uniform(0.5, 2.0, 400)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low,
         "Close": close, "Volume": rng.integers(100, 1000, 400)},
        index=idx,
    )
    ind = compute_indicators(df)
    for k in ("bb20", "stoch", "swings", "weekly_pivots", "monthly_pivots",
              "avwap_swing_high", "avwap_swing_low",
              "last_swing_high", "last_swing_low"):
        assert k in ind, f"missing new indicator: {k}"
    # Bollinger and stochastic must produce numeric latest values.
    assert pd.notna(ind["bb20"].iloc[-1]["upper"])
    assert pd.notna(ind["stoch"].iloc[-1]["k"])
    # At least one swing must have been detected on 400 bars of drift.
    assert ind["swings"]["swing_high"].sum() > 0
    assert ind["swings"]["swing_low"].sum() > 0
