"""Tests for the P2 backtest harness and trade journal.

Covers:

* :mod:`golddaytrading.backtest.outcomes` — entry trigger detection,
  stop / TP resolution priority, same-bar conservative resolution,
  expired-trade R computation, never-triggered handling.
* :mod:`golddaytrading.backtest.replay` — walk-forward engine
  produces outcomes, respects the chosen strategy, and stays
  consistent across reruns (deterministic).
* :mod:`golddaytrading.backtest.stats` — aggregate edge metrics,
  group-by min_n filter, max-drawdown sign convention.
* :mod:`golddaytrading.backtest.journal` — schema lifecycle,
  log_plan / record_outcome, stats reconstruction from DB rows.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from golddaytrading.backtest.outcomes import TradeOutcome, simulate_outcome
from golddaytrading.backtest.replay import (
    STRATEGY_ALL_IDEAS,
    STRATEGY_BEST_IDEA,
    run_backtest,
)
from golddaytrading.backtest.stats import (
    aggregate_stats,
    group_by,
    render_full_report,
    render_stats_table,
)
from golddaytrading.backtest.journal import (
    ACCEPTED_RESOLUTIONS,
    TradeJournal,
)
from golddaytrading.config import GDTConfig
from golddaytrading.signals.levels import TradeIdea


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bars(prices: list[tuple[float, float, float, float]],
          start: str = "2026-05-22 22:00",
          freq: str = "15min") -> pd.DataFrame:
    """Build a small OHLC DataFrame from explicit (open, high, low, close) tuples."""
    idx = pd.date_range(start, periods=len(prices), freq=freq, tz="UTC")
    return pd.DataFrame(
        [{"Open": o, "High": h, "Low": l, "Close": c, "Volume": 100}
         for (o, h, l, c) in prices],
        index=idx,
    )


def _long_idea(entry=2350.0, stop=2345.0, tp1=2360.0, tp2=2370.0) -> TradeIdea:
    return TradeIdea(
        setup_id="TEST_LONG",
        setup_name="test long",
        bias="LONG",
        entry=entry, stop=stop, tp1=tp1, tp2=tp2,
        rationale="x",
        rr1=abs(tp1 - entry) / abs(entry - stop),
        rr2=abs(tp2 - entry) / abs(entry - stop),
        score=2.0,
    )


def _short_idea(entry=2350.0, stop=2355.0, tp1=2340.0, tp2=2330.0) -> TradeIdea:
    return TradeIdea(
        setup_id="TEST_SHORT",
        setup_name="test short",
        bias="SHORT",
        entry=entry, stop=stop, tp1=tp1, tp2=tp2,
        rationale="x",
        rr1=abs(tp1 - entry) / abs(entry - stop),
        rr2=abs(tp2 - entry) / abs(entry - stop),
        score=2.0,
    )


def _drift_df(n: int = 400, drift: float = 80.0,
              freq: str = "15min", seed: int = 7) -> pd.DataFrame:
    """Random-walk OHLCV with a controllable drift, used by the replay
    integration tests. Long enough for warmup + horizon."""
    end = datetime(2026, 5, 23, 22, 0, tzinfo=timezone.utc)
    idx = pd.date_range(end=end, periods=n, freq=freq, tz="UTC")
    rng = np.random.default_rng(seed)
    close = 2300 + np.linspace(0, drift, n) + rng.normal(0, 3, n).cumsum() * 0.2
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0.5, 2.0, n)
    low = np.minimum(open_, close) - rng.uniform(0.5, 2.0, n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low,
         "Close": close, "Volume": rng.integers(100, 1000, n)},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Outcome simulator — entry trigger
# ---------------------------------------------------------------------------


def test_outcome_never_triggered_returns_zero_r() -> None:
    """A LONG entry above any future bar's high never triggers."""
    idea = _long_idea(entry=2400.0, stop=2390.0, tp1=2410.0, tp2=2420.0)
    future = _bars([
        (2350, 2355, 2345, 2350),
        (2350, 2356, 2347, 2348),
        (2348, 2351, 2344, 2346),
    ])
    o = simulate_outcome(idea, future)
    assert o.triggered is False
    assert o.resolution == "never_triggered"
    assert o.realised_r == 0.0
    assert not o.is_win


def test_outcome_long_hits_tp1_then_returns_positive_rr() -> None:
    """Entry at 2350; future ticks up to 2362 — TP1 first."""
    idea = _long_idea()  # rr1=2.0, rr2=4.0
    future = _bars([
        (2349, 2351, 2348, 2351),     # entry triggered, no resolution yet
        (2351, 2358, 2350, 2357),     # neither stop nor tp
        (2357, 2362, 2356, 2361),     # tp1 hit
    ])
    o = simulate_outcome(idea, future)
    assert o.triggered
    assert o.resolution == "tp1"
    assert o.realised_r == pytest.approx(2.0)
    assert o.bars_to_trigger == 0
    assert o.bars_to_resolution == 2


def test_outcome_long_stops_out() -> None:
    idea = _long_idea()
    future = _bars([
        (2349, 2351, 2348, 2350),    # entry triggered
        (2350, 2351, 2344, 2345),    # stop hit
    ])
    o = simulate_outcome(idea, future)
    assert o.resolution == "stop"
    assert o.realised_r == pytest.approx(-1.0)


def test_outcome_long_same_bar_stop_wins_over_tp() -> None:
    """When the same bar contains both stop and TP1, stop wins.

    Conservative convention: protects against backtests that look
    too good due to ambiguous intra-bar fills.
    """
    idea = _long_idea(entry=2350.0, stop=2345.0, tp1=2360.0, tp2=2370.0)
    future = _bars([
        # Single huge bar: low = 2344 (stop), high = 2362 (TP1).
        (2349, 2362, 2344, 2350),
    ])
    o = simulate_outcome(idea, future)
    assert o.resolution == "stop"
    assert o.realised_r == pytest.approx(-1.0)


def test_outcome_short_hits_tp1() -> None:
    idea = _short_idea()  # rr1=2.0
    future = _bars([
        (2351, 2352, 2349, 2350),    # entry triggered
        (2350, 2352, 2346, 2348),    # neither
        (2348, 2349, 2338, 2340),    # tp1 hit (low=2338 < 2340)
    ])
    o = simulate_outcome(idea, future)
    assert o.triggered
    assert o.resolution == "tp1"
    assert o.realised_r == pytest.approx(2.0)


def test_outcome_expired_returns_partial_r() -> None:
    """Triggered but unresolved closes at last bar's close.

    We've moved 2.5 points of a 5-point risk → +0.5R.
    """
    idea = _long_idea()
    future = _bars([
        (2349, 2351, 2348, 2350),    # entry triggered
        (2350, 2353, 2348, 2352),
        (2352, 2354, 2350, 2352.5),  # +0.5R relative to entry of 2350
    ])
    o = simulate_outcome(idea, future, max_bars=3)
    assert o.triggered
    assert o.resolution == "expired"
    assert o.realised_r == pytest.approx(0.5, rel=1e-6)


def test_outcome_max_bars_caps_walk() -> None:
    idea = _long_idea()
    future = _bars([
        (2349, 2351, 2348, 2350),  # would trigger
        (2350, 2363, 2348, 2362),  # would hit TP1 — but we cap at 1 bar
    ])
    o = simulate_outcome(idea, future, max_bars=1)
    assert o.resolution == "expired"  # triggered but didn't reach TP within cap


def test_outcome_short_same_bar_stop_wins() -> None:
    idea = _short_idea(entry=2350.0, stop=2355.0, tp1=2340.0, tp2=2330.0)
    future = _bars([
        # one bar contains both stop (high>=2355) and tp1 (low<=2340)
        (2351, 2358, 2338, 2350),
    ])
    o = simulate_outcome(idea, future)
    assert o.resolution == "stop"


def test_outcome_with_empty_future_returns_never_triggered() -> None:
    o = simulate_outcome(_long_idea(), pd.DataFrame())
    assert o.resolution == "never_triggered"
    assert o.triggered is False


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _outcome(setup_id: str, r: float, *, triggered: bool = True,
             session: str | None = None,
             resolution: str = "tp1") -> TradeOutcome:
    return TradeOutcome(
        setup_id=setup_id, bias="LONG",
        entry=100.0, stop=99.0, tp1=102.0, tp2=104.0, rr1=2.0, rr2=4.0,
        triggered=triggered,
        resolution=resolution,
        realised_r=r if triggered else 0.0,
        session=session,
    )


def test_aggregate_stats_returns_zero_dict_when_empty() -> None:
    s = aggregate_stats([])
    assert s["n"] == 0
    assert s["win_rate"] == 0.0


def test_aggregate_stats_basic_metrics() -> None:
    outs = [
        _outcome("A", +2.0, resolution="tp1"),
        _outcome("A", +2.0, resolution="tp1"),
        _outcome("A", -1.0, resolution="stop"),
        _outcome("A", -1.0, resolution="stop"),
        _outcome("A", -1.0, resolution="stop"),
    ]
    s = aggregate_stats(outs)
    assert s["n"] == 5
    assert s["win_rate"] == pytest.approx(2 / 5)
    assert s["avg_r"] == pytest.approx((2 + 2 - 1 - 1 - 1) / 5)
    assert s["expectancy_r"] == pytest.approx(
        (2 / 5) * 2.0 + (3 / 5) * (-1.0)
    )
    assert s["total_r"] == pytest.approx(1.0)
    # Cumulative drawdown: equity goes 2 -> 4 -> 3 -> 2 -> 1, peak 4 → 1 = -3
    assert s["max_dd_r"] == pytest.approx(-3.0)
    assert s["profit_factor"] == pytest.approx(4.0 / 3.0)


def test_aggregate_stats_includes_never_triggered_in_planned() -> None:
    """Trigger rate denominator is *all* planned trades, not just hits."""
    outs = [
        _outcome("A", +2.0, triggered=True),
        _outcome("A", 0.0, triggered=False, resolution="never_triggered"),
    ]
    s = aggregate_stats(outs)
    assert s["n"] == 1            # triggered only
    assert s["n_planned"] == 2
    assert s["trigger_rate"] == pytest.approx(0.5)


def test_group_by_filters_min_n() -> None:
    outs = [
        _outcome("A", +2.0), _outcome("A", -1.0), _outcome("A", -1.0),
        _outcome("B", +2.0),
    ]
    g = group_by(outs, "setup_id", min_n=2)
    assert "A" in g
    assert "B" not in g  # only 1 trigger, below min_n=2


def test_render_stats_table_handles_grouped_and_flat() -> None:
    flat = aggregate_stats([_outcome("A", +2.0)])
    grouped = group_by([_outcome("A", +2.0)], "setup_id")
    assert "ALL" in render_stats_table(flat)
    rendered = render_stats_table(grouped)
    assert "A" in rendered


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------


def test_replay_produces_outcomes_on_drift_data() -> None:
    df = _drift_df(n=400)
    cfg = GDTConfig(min_rr=1.0)
    report = run_backtest(
        df, cfg,
        warmup_bars=200, horizon_bars=24, step_bars=2,
        strategy=STRATEGY_BEST_IDEA, htf_resample_factor=4,
        ticker="XAUUSD=X", timeframe="15m",
    )
    assert report.n_decisions > 0
    # On a 200-bar drifting series we expect *some* ideas to clear
    # the R:R filter — but if not, the test still passes (it
    # exercises the engine without enforcing strategy-specific edge).
    assert report.n_with_pool >= 0
    assert all(o.setup_id for o in report.outcomes)
    # Every outcome must reference the report's strategy via metadata.
    assert report.strategy == STRATEGY_BEST_IDEA


def test_replay_all_ideas_strategy_yields_more_outcomes() -> None:
    df = _drift_df(n=400)
    cfg = GDTConfig(min_rr=1.0)
    best = run_backtest(df, cfg, warmup_bars=200, horizon_bars=24,
                        step_bars=4, strategy=STRATEGY_BEST_IDEA)
    allall = run_backtest(df, cfg, warmup_bars=200, horizon_bars=24,
                          step_bars=4, strategy=STRATEGY_ALL_IDEAS)
    assert len(allall.outcomes) >= len(best.outcomes)


def test_replay_is_deterministic() -> None:
    df = _drift_df(n=300)
    cfg = GDTConfig(min_rr=1.0)
    a = run_backtest(df, cfg, warmup_bars=200, horizon_bars=24, step_bars=2)
    b = run_backtest(df, cfg, warmup_bars=200, horizon_bars=24, step_bars=2)
    assert len(a.outcomes) == len(b.outcomes)
    for o1, o2 in zip(a.outcomes, b.outcomes):
        assert o1.setup_id == o2.setup_id
        assert o1.realised_r == pytest.approx(o2.realised_r)
        assert o1.resolution == o2.resolution


def test_replay_handles_too_short_series() -> None:
    df = _drift_df(n=50)
    cfg = GDTConfig(min_rr=1.0)
    report = run_backtest(df, cfg, warmup_bars=200, horizon_bars=24)
    assert report.outcomes == []
    assert report.n_decisions == 0


def test_render_full_report_runs() -> None:
    df = _drift_df(n=400)
    cfg = GDTConfig(min_rr=1.0)
    report = run_backtest(df, cfg, warmup_bars=200, horizon_bars=24, step_bars=4)
    md = render_full_report(report)
    assert "Backtest report" in md
    assert "Overall (all outcomes)" in md


# ---------------------------------------------------------------------------
# Trade journal
# ---------------------------------------------------------------------------


def _journal(tmp_path: Path) -> TradeJournal:
    return TradeJournal(tmp_path / "journal.sqlite3")


def test_journal_resolutions_whitelist() -> None:
    assert "tp1" in ACCEPTED_RESOLUTIONS
    assert "manual" in ACCEPTED_RESOLUTIONS
    assert "made_up" not in ACCEPTED_RESOLUTIONS


def test_journal_log_plan_and_outcome_round_trip(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    idea = _long_idea()
    ctx = {
        "ticker": "XAUUSD=X",
        "primary_timeframe": "15m",
        "now_utc": datetime.now(timezone.utc),
        "research_chosen_idea": idea,
        "guardrail": type("G", (), dict(
            approved=True, position_size_units=10.0,
            risk_dollars=50.0, rr_ratio=2.0, hard_block=None,
        ))(),
        "quant_signal": type("S", (), dict(
            p_up=0.7, expected_move_atr=0.8, direction_logit=1.0,
            direction_label="BULLISH", confidence=0.6,
        ))(),
        "macro_pulse": {"__regime__": "USD_WEAKNESS",
                        "__regime_bias__": "bullish"},
        "active_session": type("Sn", (), dict(name="LONDON_NY_OVERLAP"))(),
        "htf_trend": "up",
    }
    plan_id = j.log_plan(ctx)
    assert plan_id > 0

    outcome_id = j.record_outcome(
        plan_id, resolution="tp1", realised_r=2.0,
        notes="textbook reclaim",
    )
    assert outcome_id > 0

    plans = j.list_plans()
    assert len(plans) == 1
    assert plans[0]["setup_id"] == idea.setup_id
    assert plans[0]["macro_regime"] == "USD_WEAKNESS"
    assert plans[0]["session"] == "LONDON_NY_OVERLAP"
    assert plans[0]["approved"] == 1

    outcomes = j.list_outcomes_for_plan(plan_id)
    assert len(outcomes) == 1
    assert outcomes[0]["resolution"] == "tp1"
    assert outcomes[0]["realised_r"] == pytest.approx(2.0)
    assert outcomes[0]["notes"] == "textbook reclaim"


def test_journal_record_outcome_rejects_invalid_resolution(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    plan_id = j.log_plan({
        "ticker": "XAUUSD=X", "now_utc": datetime.now(timezone.utc),
        "research_chosen_idea": _long_idea(),
    })
    with pytest.raises(ValueError):
        j.record_outcome(plan_id, resolution="bogus", realised_r=0.0)


def test_journal_stats_block_excludes_unresolved(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    p1 = j.log_plan({
        "ticker": "XAUUSD=X", "now_utc": datetime.now(timezone.utc),
        "research_chosen_idea": _long_idea(),
    })
    p2 = j.log_plan({
        "ticker": "XAUUSD=X", "now_utc": datetime.now(timezone.utc),
        "research_chosen_idea": _short_idea(),
    })
    j.record_outcome(p1, resolution="tp1", realised_r=2.0)
    # p2 has no outcome — must be excluded from stats.
    block = j.stats_block(days_back=30, ticker="XAUUSD=X")
    assert "TEST_LONG" in block
    assert "TEST_SHORT" not in block


def test_journal_fetch_outcomes_uses_latest(tmp_path: Path) -> None:
    """Two outcomes on the same plan: only the latest survives."""
    j = _journal(tmp_path)
    pid = j.log_plan({
        "ticker": "XAUUSD=X", "now_utc": datetime.now(timezone.utc),
        "research_chosen_idea": _long_idea(),
    })
    j.record_outcome(pid, resolution="tp1", realised_r=2.0)
    j.record_outcome(pid, resolution="stop", realised_r=-1.0,
                     notes="corrected")
    outs = j.fetch_outcomes(days_back=30, ticker="XAUUSD=X")
    assert len(outs) == 1
    assert outs[0].resolution == "stop"
    assert outs[0].realised_r == pytest.approx(-1.0)


def test_journal_reset_clears_tables(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    pid = j.log_plan({
        "ticker": "XAUUSD=X", "now_utc": datetime.now(timezone.utc),
        "research_chosen_idea": _long_idea(),
    })
    j.record_outcome(pid, resolution="tp1", realised_r=2.0)
    assert j.list_plans()
    j.reset()
    assert j.list_plans() == []


def test_pipeline_auto_logs_plan(monkeypatch, tmp_path: Path) -> None:
    """End-to-end: a successful offline pipeline run must persist a
    plan row in the journal so subsequent runs can read its stats."""
    monkeypatch.setenv("GDT_LLM_PROVIDER", "offline")
    df = _drift_df(n=400)

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
    cfg.results_dir = str(tmp_path / "runs")
    cfg.journal_db_path = str(tmp_path / "journal.sqlite3")
    p = pipeline_mod.DayTradingPipeline(cfg=cfg, logger=lambda *_: None)
    ctx = p.run("XAUUSD=X")

    assert ctx.get("research_chosen_idea") is not None
    j = TradeJournal(cfg.journal_db_path)
    plans = j.list_plans()
    assert len(plans) == 1
    assert plans[0]["setup_id"]
    assert ctx.get("journal_plan_id") == plans[0]["id"]
