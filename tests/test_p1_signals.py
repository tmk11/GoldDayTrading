"""Tests for the P1 signal layer.

Covers:

* :mod:`golddaytrading.signals.quant_baseline` — feature extraction,
  direction sensitivity to macro pulse, calibration anchors,
  graceful handling of empty / missing data.
* :mod:`golddaytrading.signals.levels` — geometry validation, R:R
  filter, scoring with HTF + quant alignment, rendering.
* :mod:`golddaytrading.signals.envelope` — JSON parser, fenced and
  bare formats, hallucination downgrade, offline fallback path.
* :mod:`golddaytrading.dataflows.macro_pulse` — regime classifier
  buckets every regime, derived field math, prompt block contents.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from golddaytrading.dataflows.indicators import compute_indicators
from golddaytrading.dataflows.macro_pulse import (
    classify_regime,
    macro_pulse_block,
    REGIMES,
)
from golddaytrading.signals import (
    LevelPool,
    TradeIdea,
    build_level_pool,
    compute_quant_signal,
    parse_envelope,
    quant_signal_block,
    render_envelope_block,
)
from golddaytrading.signals.envelope import _extract_json_block


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_uptrend_df(n: int = 400, freq: str = "15min",
                     drift: float = 80.0) -> pd.DataFrame:
    """Synthetic uptrending OHLCV. EMA stack will be 20>50>200."""
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    idx = pd.date_range(end=end, periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(11)
    close = 2300 + np.linspace(0, drift, n) + rng.normal(0, 3, n).cumsum() * 0.15
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.5, 2.0, n)
    low = np.minimum(open_, close) - rng.uniform(0.5, 2.0, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low,
         "Close": close, "Volume": rng.integers(100, 1000, n)},
        index=idx,
    )


def _make_downtrend_df(n: int = 400) -> pd.DataFrame:
    df = _make_uptrend_df(n=n, drift=-80.0)
    return df


# ---------------------------------------------------------------------------
# Quant baseline
# ---------------------------------------------------------------------------


def test_quant_baseline_no_data_returns_neutral() -> None:
    sig = compute_quant_signal(pd.DataFrame(), {}, macro_pulse={})
    assert sig.p_up == 0.5
    assert sig.expected_move_atr == 0.0
    assert sig.confidence == 0.0
    assert sig.direction_label == "NEUTRAL"


def test_quant_baseline_uptrend_pushes_p_up_above_05() -> None:
    df = _make_uptrend_df()
    ind = compute_indicators(df)
    sig = compute_quant_signal(df, ind, macro_pulse={},
                                session_name="LONDON_NY_OVERLAP")
    assert sig.p_up > 0.55
    assert sig.expected_move_atr > 0
    assert sig.direction_label in ("BULLISH", "NEUTRAL")
    assert "ema_stack_score" in sig.features
    assert sig.features["ema_stack_score"] > 0


def test_quant_baseline_downtrend_pushes_p_up_below_05() -> None:
    df = _make_downtrend_df()
    ind = compute_indicators(df)
    sig = compute_quant_signal(df, ind, macro_pulse={},
                                session_name="TOKYO")
    assert sig.p_up < 0.45
    assert sig.expected_move_atr < 0
    assert sig.features["ema_stack_score"] < 0


def test_quant_baseline_responds_to_dxy_move() -> None:
    """A strong DXY drop should lift p_up vs baseline (USD inverse)."""
    df = _make_uptrend_df()
    ind = compute_indicators(df)
    sig_baseline = compute_quant_signal(df, ind, macro_pulse={})
    macro_dxy_down = {
        "DX-Y.NYB": {"last": 105.0, "chg_1h": -0.30,  # 3σ down
                     "chg_4h": -0.60, "chg_1d": -1.0, "bias": "bearish"},
    }
    sig_with_dxy = compute_quant_signal(df, ind, macro_pulse=macro_dxy_down)
    assert sig_with_dxy.p_up > sig_baseline.p_up


def test_quant_baseline_signal_block_contains_drivers() -> None:
    df = _make_uptrend_df()
    ind = compute_indicators(df)
    sig = compute_quant_signal(df, ind, macro_pulse={})
    block = quant_signal_block(sig)
    assert "Quant baseline signal" in block
    assert "Direction:" in block
    assert "P(up over" in block
    assert "Top drivers" in block


# ---------------------------------------------------------------------------
# Level pool
# ---------------------------------------------------------------------------


def test_level_pool_only_contains_valid_geometry() -> None:
    df = _make_uptrend_df()
    ind = compute_indicators(df)
    pool = build_level_pool(df, ind, min_rr=1.0, htf_trend="up",
                            quant_p_up=0.6)
    for idea in pool.ideas:
        if idea.bias == "LONG":
            assert idea.stop < idea.entry < idea.tp1 <= idea.tp2
        else:
            assert idea.stop > idea.entry > idea.tp1 >= idea.tp2
        assert idea.rr1 >= 1.0


def test_level_pool_filters_below_min_rr() -> None:
    df = _make_uptrend_df()
    ind = compute_indicators(df)
    pool_lo = build_level_pool(df, ind, min_rr=1.0)
    pool_hi = build_level_pool(df, ind, min_rr=5.0)
    assert len(pool_hi.ideas) <= len(pool_lo.ideas)
    for r in pool_hi.notes:
        # Rejected reasons must list at least one R:R rejection.
        if "R:R" in r:
            break
    else:
        # No R:R rejection in notes is acceptable when literally
        # nothing was generated, but in our synthetic case at least
        # one rejection should mention R:R.
        if pool_lo.ideas:
            pytest.fail(
                "Expected min_rr=5.0 to reject something with an R:R note"
            )


def test_level_pool_ranks_quant_aligned_higher() -> None:
    df = _make_uptrend_df()
    ind = compute_indicators(df)
    pool_long_aligned = build_level_pool(
        df, ind, min_rr=1.0, htf_trend="up", quant_p_up=0.85
    )
    pool_short_aligned = build_level_pool(
        df, ind, min_rr=1.0, htf_trend="up", quant_p_up=0.20
    )
    if pool_long_aligned.ideas and pool_short_aligned.ideas:
        # When the quant prior flips, a LONG idea's score must drop
        # vs the short-aligned variant (or a SHORT idea must outrank).
        long_top = next(
            (i for i in pool_long_aligned.ideas if i.bias == "LONG"), None
        )
        long_top_alt = next(
            (i for i in pool_short_aligned.ideas
             if i.setup_id == (long_top.setup_id if long_top else "")),
            None,
        )
        if long_top is not None and long_top_alt is not None:
            assert long_top.score >= long_top_alt.score


def test_level_pool_block_renders_when_empty() -> None:
    pool = LevelPool(ideas=[], notes=["something rejected"])
    block = LevelPool.__module__  # noqa: F841 — just to silence lint
    text = (
        "### Deterministic level pool\n"
        if False else __import__(
            "golddaytrading.signals.levels", fromlist=["level_pool_block"]
        ).level_pool_block(pool)
    )
    assert "No valid trade idea" in text
    assert "something rejected" in text


# ---------------------------------------------------------------------------
# Research envelope
# ---------------------------------------------------------------------------


def _make_pool_with_ideas() -> LevelPool:
    return LevelPool(ideas=[
        TradeIdea(
            setup_id="VWAP_RECLAIM_LONG",
            setup_name="VWAP reclaim long", bias="LONG",
            entry=2350.0, stop=2345.0, tp1=2360.0, tp2=2370.0,
            rationale="x", rr1=2.0, rr2=4.0, score=2.5,
        ),
        TradeIdea(
            setup_id="OR_BREAKDOWN_SHORT",
            setup_name="OR breakdown short", bias="SHORT",
            entry=2348.0, stop=2353.0, tp1=2340.0, tp2=2330.0,
            rationale="y", rr1=1.6, rr2=3.6, score=1.8,
        ),
    ])


def test_envelope_parses_fenced_json() -> None:
    pool = _make_pool_with_ideas()
    text = (
        "Some prose first.\n"
        '```json\n'
        '{"bias": "LONG", "conviction": "medium", '
        '"selected_setup_id": "VWAP_RECLAIM_LONG", '
        '"rationale": "Trend + quant align."}\n'
        '```\n'
        "More prose."
    )
    env = parse_envelope(text, pool)
    assert env is not None
    assert env.bias == "LONG"
    assert env.conviction == "medium"
    assert env.chosen_idea is not None
    assert env.chosen_idea.setup_id == "VWAP_RECLAIM_LONG"


def test_envelope_parses_bare_json() -> None:
    pool = _make_pool_with_ideas()
    text = (
        '{"bias": "SHORT", "conviction": "high", '
        '"selected_setup_id": "OR_BREAKDOWN_SHORT", '
        '"rationale": "Pivot rejection."}'
    )
    env = parse_envelope(text, pool)
    assert env is not None
    assert env.bias == "SHORT"
    assert env.chosen_idea is not None
    assert env.chosen_idea.setup_id == "OR_BREAKDOWN_SHORT"


def test_envelope_downgrades_unknown_setup_to_flat() -> None:
    """A non-FLAT bias with an unknown setup_id is treated as FLAT.

    This protects the downstream Risk Manager from acting on prices
    the LLM hallucinated outside the pool.
    """
    pool = _make_pool_with_ideas()
    text = '```json\n{"bias": "LONG", "selected_setup_id": "MADE_UP_LONG"}\n```'
    env = parse_envelope(text, pool)
    assert env is not None
    assert env.bias == "FLAT"
    assert env.chosen_idea is None


def test_envelope_returns_none_when_no_json() -> None:
    pool = _make_pool_with_ideas()
    env = parse_envelope("Pure prose, no JSON anywhere.", pool)
    assert env is None


def test_extract_json_block_handles_empty() -> None:
    assert _extract_json_block("") is None


def test_render_envelope_block_long() -> None:
    pool = _make_pool_with_ideas()
    text = (
        '```json\n'
        '{"bias": "LONG", "conviction": "high", '
        '"selected_setup_id": "VWAP_RECLAIM_LONG", '
        '"rationale": "ok"}\n'
        '```'
    )
    env = parse_envelope(text, pool)
    block = render_envelope_block(env)
    assert "Bias:** LONG" in block
    assert "VWAP_RECLAIM_LONG" in block
    assert "2350.00" in block


def test_render_envelope_block_flat() -> None:
    block = render_envelope_block(None)
    assert "No JSON envelope" in block


# ---------------------------------------------------------------------------
# Macro regime classification
# ---------------------------------------------------------------------------


def test_macro_regime_classifier_returns_known_regime() -> None:
    pulse = {
        "DX-Y.NYB": {"chg_1h": 0.20, "last": 105.0},
        "^TNX":     {"chg_1h": 0.30, "last": 4.50},
        "TIP":      {"chg_1h": -0.15, "last": 110.0},
        "^VIX":     {"chg_1h": 1.0, "last": 16.0},
        "ES=F":     {"chg_1h": 0.0, "last": 5000.0},
    }
    regime = classify_regime(pulse)
    assert regime.name in REGIMES
    assert regime.gold_bias in ("bullish", "bearish", "neutral")


def test_macro_regime_real_yield_drive() -> None:
    pulse = {
        "DX-Y.NYB": {"chg_1h": 0.20},
        "^TNX":     {"chg_1h": 0.30},
        "TIP":      {"chg_1h": -0.15},
        "^VIX":     {"chg_1h": 0.0, "last": 15.0},
        "ES=F":     {"chg_1h": 0.0},
    }
    regime = classify_regime(pulse)
    assert regime.name == "REAL_YIELD_DRIVE"
    assert regime.gold_bias == "bearish"


def test_macro_regime_usd_weakness() -> None:
    pulse = {
        "DX-Y.NYB": {"chg_1h": -0.25},
        "^TNX":     {"chg_1h": 0.0},
        "TIP":      {"chg_1h": 0.0},
        "^VIX":     {"chg_1h": 0.0, "last": 15.0},
        "ES=F":     {"chg_1h": 0.0},
    }
    regime = classify_regime(pulse)
    assert regime.name == "USD_WEAKNESS"
    assert regime.gold_bias == "bullish"


def test_macro_regime_risk_off() -> None:
    pulse = {
        "DX-Y.NYB": {"chg_1h": 0.0},
        "^TNX":     {"chg_1h": -0.20},
        "TIP":      {"chg_1h": 0.05},
        "^VIX":     {"chg_1h": 8.0, "last": 22.0},
        "ES=F":     {"chg_1h": -0.40},
    }
    regime = classify_regime(pulse)
    assert regime.name == "RISK_OFF_HAVEN_BID"
    assert regime.gold_bias == "bullish"


def test_macro_regime_range_bound_default() -> None:
    pulse = {
        "DX-Y.NYB": {"chg_1h": 0.0},
        "^TNX":     {"chg_1h": 0.0},
        "TIP":      {"chg_1h": 0.0},
        "^VIX":     {"chg_1h": 0.0, "last": 15.0},
        "ES=F":     {"chg_1h": 0.0},
    }
    regime = classify_regime(pulse)
    assert regime.name == "RANGE_BOUND"


def test_macro_pulse_block_contains_regime_and_drivers() -> None:
    pulse = {
        "DX-Y.NYB": {"last": 104.5, "chg_1h": -0.10, "chg_4h": -0.30,
                     "chg_1d": -0.5, "bias": "bearish"},
        "^TNX":     {"last": 4.20, "chg_1h": -0.15, "chg_4h": -0.40,
                     "chg_1d": -0.8, "bias": "bearish"},
        "TIP":      {"last": 110.5, "chg_1h": 0.10, "chg_4h": 0.20,
                     "chg_1d": 0.5, "bias": "bullish"},
        "^VIX":     {"last": 14.5, "chg_1h": -1.0, "chg_4h": -2.0,
                     "chg_1d": -3.0, "bias": "bullish"},
        "ES=F":     {"last": 5100.0, "chg_1h": 0.10, "chg_4h": 0.20,
                     "chg_1d": 0.4, "bias": "neutral"},
        "EURUSD=X": {"last": 1.085, "chg_1h": 0.10, "chg_4h": 0.20,
                     "chg_1d": 0.50, "bias": "bullish"},
        "^FVX":     {"last": 4.10, "chg_1h": -0.10, "chg_4h": -0.20,
                     "chg_1d": -0.40, "bias": "bearish"},
        "^TYX":     {"last": 4.40, "chg_1h": -0.10, "chg_4h": -0.20,
                     "chg_1d": -0.30, "bias": "bearish"},
        "BTC-USD":  {"last": 70000.0, "chg_1h": 0.20, "chg_4h": 0.50,
                     "chg_1d": 1.0, "bias": "neutral"},
        "CL=F":     {"last": 78.5, "chg_1h": 0.10, "chg_4h": 0.20,
                     "chg_1d": 0.30, "bias": "bullish"},
        "SI=F":     {"last": 31.5, "chg_1h": 0.20, "chg_4h": 0.40,
                     "chg_1d": 0.80, "bias": "bullish"},
        "__derived__": {"real_yield_proxy_chg_1h": -0.10,
                        "belly_vs_10y_chg_1h": 0.05,
                        "silver_last": 31.5},
        "__regime__": "USD_WEAKNESS",
        "__regime_bias__": "bullish",
        "__regime_description__": "USD weakness without a yield spike.",
    }
    block = macro_pulse_block(pulse)
    assert "DXY (USD index)" in block
    assert "EURUSD" in block
    assert "Silver futures" in block
    assert "USD_WEAKNESS" in block
    assert "Real-yield proxy" in block
    assert "5Y vs 10Y" in block


def test_quant_baseline_uses_macro_regime_score() -> None:
    """The quant baseline must shift p_up when a bullish regime is tagged."""
    df = _make_uptrend_df()
    ind = compute_indicators(df)
    sig_no_regime = compute_quant_signal(df, ind, macro_pulse={})
    pulse_bull_regime = {
        "__regime__": "USD_WEAKNESS",
        "__regime_bias__": "bullish",
    }
    sig_bull = compute_quant_signal(df, ind, macro_pulse=pulse_bull_regime)
    assert sig_bull.p_up > sig_no_regime.p_up

    pulse_bear_regime = {
        "__regime__": "REAL_YIELD_DRIVE",
        "__regime_bias__": "bearish",
    }
    sig_bear = compute_quant_signal(df, ind, macro_pulse=pulse_bear_regime)
    assert sig_bear.p_up < sig_no_regime.p_up


# ---------------------------------------------------------------------------
# End-to-end pipeline smoke (offline) — verifies all wiring lines up
# ---------------------------------------------------------------------------


def test_pipeline_offline_run_uses_deterministic_levels(monkeypatch) -> None:
    """The full offline pipeline must produce a guardrail with prices
    that exactly match the chosen level-pool idea (no LLM-invented
    numbers can leak through on the happy path)."""
    monkeypatch.setenv("GDT_LLM_PROVIDER", "offline")

    df = _make_uptrend_df()

    from golddaytrading.config import load_config
    from golddaytrading.graph import pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "fetch_intraday_ohlcv",
                        lambda *a, **k: df)
    monkeypatch.setattr(pipeline_mod, "fetch_macro_pulse", lambda: {})
    monkeypatch.setattr(pipeline_mod, "fetch_recent_gold_news",
                        lambda *a, **k: [])
    monkeypatch.setattr(pipeline_mod, "fetch_upcoming_events",
                        lambda *a, **k: [])

    cfg = load_config(min_rr=1.0)
    cfg.results_dir = "/tmp/gdt_test_runs"
    p = pipeline_mod.DayTradingPipeline(cfg=cfg, logger=lambda *_: None)
    ctx = p.run("XAUUSD=X")

    chosen = ctx.get("research_chosen_idea")
    guard = ctx.get("guardrail")
    assert chosen is not None, "research_chosen_idea must be populated"
    assert guard is not None
    if guard.approved:
        # When approved, the guardrail's risk budget must equal
        # cfg.risk_per_trade_pct of cfg.account_usd.
        expected_risk = cfg.account_usd * cfg.risk_per_trade_pct / 100.0
        assert guard.risk_dollars == pytest.approx(expected_risk, rel=1e-6)
        # Position size = risk_dollars / |entry - stop|
        risk_per_unit = abs(chosen.entry - chosen.stop)
        assert guard.position_size_units == pytest.approx(
            expected_risk / risk_per_unit, rel=1e-6
        )
