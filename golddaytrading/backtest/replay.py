"""Walk-forward backtest replay engine.

Drives the deterministic half of the GoldDayTrading pipeline
(indicators → quant signal → level pool → top idea selection) over
a historical OHLCV DataFrame and records simulated outcomes for
each decision bar.

Why deterministic-only?

The LLM debate, research-manager synthesis, and risk-manager
verdict are non-deterministic and slow. A backtest that replays
those layers would (a) cost serious tokens, (b) fail to reproduce
across runs, and (c) hide the edge of the *foundation* (level pool
+ quant scorer) under LLM noise. Instead we measure the foundation
and let the journal (live runs) measure how the LLM layer adds or
subtracts value on top.

Macro replay
------------

Replaying historical macro pulses requires synced multi-asset
historical OHLCV (DXY, ^TNX, TIP, …) at the same cadence as the
primary TF. That data path is brittle on yfinance for intraday and
adds complexity. v1 of this module passes ``macro_pulse={}`` so the
quant signal uses only technicals during backtests. Live runs still
get the full macro pulse via the pipeline. This is documented as a
known limitation.

Higher-timeframe trend
----------------------

Optional: when ``htf_resample_factor`` is set (e.g. 4 for a 15m
primary TF → 1h HTF), each decision bar's HTF EMA stack is computed
from a simple Nx aggregation of the primary bars. This gives the
level-pool ranker the same HTF tie-breaker it gets in production.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from golddaytrading.backtest.outcomes import TradeOutcome, simulate_outcome
from golddaytrading.config import GDTConfig, load_config
from golddaytrading.dataflows.indicators import compute_indicators
from golddaytrading.sessions import classify_session
from golddaytrading.signals.levels import LevelPool, TradeIdea, build_level_pool
from golddaytrading.signals.quant_baseline import (
    QuantSignal,
    compute_quant_signal,
)


# Strategies the replay engine recognises.
STRATEGY_BEST_IDEA = "best_idea"           # only the top-ranked idea
STRATEGY_ALL_IDEAS = "all_ideas"           # every idea in the pool
STRATEGY_ALIGNED = "p_up_aligned"          # ideas whose bias matches quant prior


@dataclass
class BacktestReport:
    """Container for a backtest run's outcomes + run-level metadata."""

    outcomes: List[TradeOutcome] = field(default_factory=list)
    n_decisions: int = 0          # bars where the engine could choose
    n_with_pool: int = 0          # bars that produced at least one idea
    strategy: str = STRATEGY_BEST_IDEA
    timeframe: Optional[str] = None
    ticker: Optional[str] = None
    primary_bars: int = 0
    horizon_bars: int = 0
    warmup_bars: int = 0

    @property
    def trigger_rate(self) -> float:
        if not self.outcomes:
            return 0.0
        triggered = sum(1 for o in self.outcomes if o.triggered)
        return triggered / len(self.outcomes)

    def triggered_outcomes(self) -> List[TradeOutcome]:
        return [o for o in self.outcomes if o.triggered]


def _resample_to_htf(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    """Aggregate primary bars into a higher-timeframe DataFrame.

    Groups every ``factor`` consecutive bars and OHLC-aggregates.
    The resulting index uses the *last* timestamp in each group so
    the EMAs computed downstream don't peek into the future.
    """
    if df.empty or factor <= 1:
        return df
    n = len(df)
    if n < factor:
        return df.iloc[-1:].copy()
    g = np.arange(n) // factor
    agg = (
        df.groupby(g, sort=False)
          .agg({"Open": "first", "High": "max", "Low": "min",
                "Close": "last",
                "Volume": "sum" if "Volume" in df.columns else "max"})
    )
    last_ts = df.groupby(g, sort=False).apply(lambda x: x.index[-1])
    agg.index = last_ts.values
    return agg


def _htf_trend_label(htf_df: Optional[pd.DataFrame]) -> Optional[str]:
    if htf_df is None or htf_df.empty or len(htf_df) < 50:
        return None
    htf_ind = compute_indicators(htf_df)
    if not htf_ind:
        return None
    ema20 = float(htf_ind["ema20"].iloc[-1])
    ema50 = float(htf_ind["ema50"].iloc[-1])
    ema200 = float(htf_ind["ema200"].iloc[-1])
    if ema20 > ema50 > ema200:
        return "up"
    if ema20 < ema50 < ema200:
        return "down"
    return "chop"


def _ideas_for_strategy(
    pool: LevelPool, strategy: str, p_up: Optional[float]
) -> List[TradeIdea]:
    if strategy == STRATEGY_ALL_IDEAS:
        return list(pool.ideas)
    if strategy == STRATEGY_ALIGNED and p_up is not None:
        keep: List[TradeIdea] = []
        for idea in pool.ideas:
            if idea.bias == "LONG" and p_up >= 0.55:
                keep.append(idea)
            elif idea.bias == "SHORT" and p_up <= 0.45:
                keep.append(idea)
        return keep
    # Default: top-ranked only (matches what the live pipeline picks).
    return [pool.best] if pool.best is not None else []


def run_backtest(
    df: pd.DataFrame,
    cfg: Optional[GDTConfig] = None,
    *,
    warmup_bars: int = 200,
    step_bars: int = 1,
    horizon_bars: int = 32,
    strategy: str = STRATEGY_BEST_IDEA,
    htf_resample_factor: Optional[int] = 4,
    ticker: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> BacktestReport:
    """Run the deterministic walk-forward backtest.

    Parameters
    ----------
    df : pd.DataFrame
        Historical OHLCV at the primary timeframe, tz-aware index.
    cfg : Optional[GDTConfig]
        Risk / VWAP-anchor config. ``cfg.min_rr`` filters the pool.
    warmup_bars : int
        Number of bars consumed before the first decision (so EMA200
        and pivots have data to work with).
    step_bars : int
        Stride between decision bars; ``1`` evaluates every new bar.
    horizon_bars : int
        How many future bars to walk forward when resolving each
        idea. Should be longer than the median trade duration.
    strategy : str
        ``best_idea`` (default), ``all_ideas`` or ``p_up_aligned``.
    htf_resample_factor : Optional[int]
        Group every N primary bars into an HTF bar for trend
        detection. Set to ``None`` to disable HTF (the level-pool
        ranker then receives ``htf_trend=None``).
    """
    cfg = cfg or load_config()
    report = BacktestReport(
        strategy=strategy,
        timeframe=timeframe or getattr(cfg, "primary_timeframe", None),
        ticker=ticker or getattr(cfg, "ticker", None),
        primary_bars=len(df) if df is not None else 0,
        horizon_bars=horizon_bars,
        warmup_bars=warmup_bars,
    )

    if df is None or df.empty:
        return report

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("backtest input must be DatetimeIndex-ed OHLCV")

    n = len(df)
    if n <= warmup_bars + horizon_bars:
        return report

    last_decision_idx = n - horizon_bars - 1
    for end_idx in range(warmup_bars, last_decision_idx + 1, max(step_bars, 1)):
        history = df.iloc[: end_idx + 1]
        future = df.iloc[end_idx + 1: end_idx + 1 + horizon_bars]
        if future.empty:
            break

        ind = compute_indicators(
            history, vwap_anchor_hour_utc=cfg.vwap_anchor_hour_utc
        )
        if not ind:
            continue
        report.n_decisions += 1

        try:
            session_name = classify_session(history.index[-1]).name
        except Exception:
            session_name = None

        sig = compute_quant_signal(
            history, ind, macro_pulse={}, session_name=session_name
        )

        htf_trend: Optional[str] = None
        if htf_resample_factor and htf_resample_factor > 1:
            htf_df = _resample_to_htf(history, htf_resample_factor)
            htf_trend = _htf_trend_label(htf_df)

        pool = build_level_pool(
            history, ind,
            min_rr=cfg.min_rr,
            htf_trend=htf_trend,
            quant_p_up=sig.p_up,
        )
        if not pool.ideas:
            continue
        report.n_with_pool += 1

        chosen_ideas = _ideas_for_strategy(pool, strategy, sig.p_up)
        if not chosen_ideas:
            continue

        decision_ts = history.index[-1]
        for idea in chosen_ideas:
            outcome = simulate_outcome(
                idea, future, max_bars=horizon_bars,
                decision_ts=decision_ts.to_pydatetime(),
            )
            outcome.session = session_name
            outcome.htf_trend = htf_trend
            outcome.quant_p_up = float(sig.p_up)
            report.outcomes.append(outcome)

    return report
