"""Trade-outcome simulator.

Given a :class:`golddaytrading.signals.levels.TradeIdea` and the
DataFrame of bars *after* the decision timestamp, walk forward and
classify the outcome:

* ``never_triggered`` — entry level was never reached within horizon.
* ``stop``            — stop-loss hit first.
* ``tp1`` / ``tp2``   — first / second target hit.
* ``expired``         — entry triggered but neither stop nor target
  resolved within ``max_bars``; closed at the last bar's close.

Same-bar resolution
-------------------

When a single bar's range contains *both* the stop and a profit
target, we conservatively assume the **stop hits first**. This is
the industry-standard convention (it matches what would happen with
a stop-market order being triggered before a limit-take-profit) and
biases the backtest toward a fair / pessimistic estimate of the
strategy edge.

A LONG entry is considered "filled" the first bar where
``low <= entry <= high`` (i.e. the entry price falls inside the bar's
range). The SHORT case is symmetric. This handles both stop-buy
(entry above current) and limit-buy (entry below current) styles
without the caller having to specify the order type.

R-multiple convention
---------------------

``realised_r`` is in units of risk per unit (``|entry - stop|``).
A loss is ``-1.0R`` exactly (the stop). A TP1 / TP2 hit is
``+rr1`` / ``+rr2`` exactly (anchored to the deterministic level
pool's R:R). An ``expired`` outcome reports the partial R-multiple
based on the last bar's close.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from golddaytrading.signals.levels import TradeIdea


@dataclass
class TradeOutcome:
    """Result of walking forward one trade idea through ``future_bars``."""

    setup_id: str
    bias: str
    entry: float
    stop: float
    tp1: float
    tp2: float
    rr1: float
    rr2: float

    triggered: bool
    resolution: str          # "tp1" | "tp2" | "stop" | "expired" | "never_triggered"
    realised_r: float        # 0.0 when never triggered
    bars_to_trigger: Optional[int] = None
    bars_to_resolution: Optional[int] = None

    decision_ts: Optional[datetime] = None
    entry_ts: Optional[datetime] = None
    exit_ts: Optional[datetime] = None

    # Free-form context for stats group-by.
    session: Optional[str] = None
    htf_trend: Optional[str] = None
    quant_p_up: Optional[float] = None
    macro_regime: Optional[str] = None

    @property
    def is_win(self) -> bool:
        return self.triggered and self.realised_r > 0


def _safe(v: object, default: float = 0.0) -> float:
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _resolve_long_intrabar(
    bar_low: float, bar_high: float, idea: TradeIdea
) -> Optional[str]:
    """Same-bar resolution for an active LONG position.

    Returns ``"stop"``, ``"tp2"``, ``"tp1"`` or ``None`` (continue).
    Conservative: stop wins ties (touched the same bar as a TP).
    """
    if bar_low <= idea.stop:
        return "stop"
    if bar_high >= idea.tp2:
        return "tp2"
    if bar_high >= idea.tp1:
        return "tp1"
    return None


def _resolve_short_intrabar(
    bar_low: float, bar_high: float, idea: TradeIdea
) -> Optional[str]:
    """Same-bar resolution for an active SHORT position."""
    if bar_high >= idea.stop:
        return "stop"
    if bar_low <= idea.tp2:
        return "tp2"
    if bar_low <= idea.tp1:
        return "tp1"
    return None


def _r_for_resolution(idea: TradeIdea, resolution: str) -> float:
    if resolution == "stop":
        return -1.0
    if resolution == "tp1":
        return float(idea.rr1)
    if resolution == "tp2":
        return float(idea.rr2)
    return 0.0


def _expired_r(idea: TradeIdea, last_close: float) -> float:
    risk = abs(idea.entry - idea.stop)
    if risk <= 0:
        return 0.0
    if idea.bias == "LONG":
        return (last_close - idea.entry) / risk
    return (idea.entry - last_close) / risk


def simulate_outcome(
    idea: TradeIdea,
    future_bars: pd.DataFrame,
    max_bars: Optional[int] = None,
    decision_ts: Optional[datetime] = None,
) -> TradeOutcome:
    """Walk forward through ``future_bars`` and resolve the trade.

    Parameters
    ----------
    idea : TradeIdea
        The setup to simulate.
    future_bars : pd.DataFrame
        OHLCV bars *strictly after* the decision bar (no look-ahead).
        Indexed by tz-aware timestamps; must contain ``High`` /
        ``Low`` / ``Close`` columns.
    max_bars : Optional[int]
        Cap on bars to walk forward. Defaults to ``len(future_bars)``.
    decision_ts : Optional[datetime]
        Decision timestamp used for record-keeping only.
    """
    base = TradeOutcome(
        setup_id=idea.setup_id, bias=idea.bias,
        entry=idea.entry, stop=idea.stop, tp1=idea.tp1, tp2=idea.tp2,
        rr1=idea.rr1, rr2=idea.rr2,
        triggered=False, resolution="never_triggered", realised_r=0.0,
        decision_ts=decision_ts,
    )

    if future_bars is None or future_bars.empty:
        return base

    n = len(future_bars) if max_bars is None else min(max_bars, len(future_bars))
    if n <= 0:
        return base

    triggered = False
    entry_idx: Optional[int] = None
    entry_ts: Optional[datetime] = None

    for i in range(n):
        bar = future_bars.iloc[i]
        ts = future_bars.index[i]
        bar_h = _safe(bar.get("High"))
        bar_l = _safe(bar.get("Low"))
        bar_c = _safe(bar.get("Close"))

        if not triggered:
            # Entry fills when the bar's range covers ``idea.entry``.
            if bar_l <= idea.entry <= bar_h:
                triggered = True
                entry_idx = i
                entry_ts = ts
                # Same-bar resolution check.
                if idea.bias == "LONG":
                    res = _resolve_long_intrabar(bar_l, bar_h, idea)
                else:
                    res = _resolve_short_intrabar(bar_l, bar_h, idea)
                if res is not None:
                    return TradeOutcome(
                        setup_id=idea.setup_id, bias=idea.bias,
                        entry=idea.entry, stop=idea.stop,
                        tp1=idea.tp1, tp2=idea.tp2,
                        rr1=idea.rr1, rr2=idea.rr2,
                        triggered=True,
                        resolution=res,
                        realised_r=_r_for_resolution(idea, res),
                        bars_to_trigger=i,
                        bars_to_resolution=i,
                        decision_ts=decision_ts,
                        entry_ts=entry_ts,
                        exit_ts=ts,
                    )
            # Otherwise keep waiting for the entry trigger.
            continue

        # Already in the trade — check stop / TP on this bar.
        if idea.bias == "LONG":
            res = _resolve_long_intrabar(bar_l, bar_h, idea)
        else:
            res = _resolve_short_intrabar(bar_l, bar_h, idea)
        if res is not None:
            assert entry_idx is not None  # narrowed by ``triggered`` flag
            return TradeOutcome(
                setup_id=idea.setup_id, bias=idea.bias,
                entry=idea.entry, stop=idea.stop, tp1=idea.tp1, tp2=idea.tp2,
                rr1=idea.rr1, rr2=idea.rr2,
                triggered=True,
                resolution=res,
                realised_r=_r_for_resolution(idea, res),
                bars_to_trigger=entry_idx,
                bars_to_resolution=i,
                decision_ts=decision_ts,
                entry_ts=entry_ts,
                exit_ts=ts,
            )

    # End of horizon.
    if not triggered:
        return base

    # Triggered but unresolved — close at the last bar's close.
    last_idx = n - 1
    last_close = _safe(future_bars.iloc[last_idx].get("Close"))
    last_ts = future_bars.index[last_idx]
    return TradeOutcome(
        setup_id=idea.setup_id, bias=idea.bias,
        entry=idea.entry, stop=idea.stop, tp1=idea.tp1, tp2=idea.tp2,
        rr1=idea.rr1, rr2=idea.rr2,
        triggered=True,
        resolution="expired",
        realised_r=_expired_r(idea, last_close),
        bars_to_trigger=entry_idx,
        bars_to_resolution=last_idx,
        decision_ts=decision_ts,
        entry_ts=entry_ts,
        exit_ts=last_ts,
    )
