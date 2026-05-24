"""Deterministic trade-level pool generator.

LLMs are notoriously bad at producing *exact* prices: ask GPT-4 to
output a stop level and you get a number that *looks* round but is
not anchored to any real market structure. The Research Manager
prompt downstream therefore receives a curated **level pool** —
specific entry / stop / target triples derived deterministically
from the indicator block — and is asked to *select* the idea (or
return FLAT). This eliminates an entire class of hallucination.

We generate up to ~6 candidate trade ideas covering the canonical
intraday setups for gold:

* **VWAP reclaim** (long) and **VWAP rejection** (short) — mean
  reversion to / from session VWAP.
* **Opening-range breakout** (long) and **breakdown** (short) — first
  hour high/low + ATR cushion.
* **Pivot bounce** (long off S1) and **pivot rejection** (short off R1).
* **Bollinger mean reversion** (long off lower band when RSI < 25,
  short off upper band when RSI > 75) — fade extremes only.

Every idea is screened for:

* a positive entry/stop separation,
* a configurable minimum R:R against TP1,
* alignment with the higher-timeframe trend (we *bias* the ranking,
  not eliminate counter-trend ideas, since gold mean-reverts often).

The output is a :class:`LevelPool` — a sorted list of
:class:`TradeIdea` objects with a ``rationale`` so the LLM has the
*why* baked in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import math

import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TradeIdea:
    setup_id: str                 # stable id, e.g. "VWAP_RECLAIM_LONG"
    setup_name: str               # human-readable
    bias: str                     # "LONG" | "SHORT"
    entry: float
    stop: float
    tp1: float
    tp2: float
    rationale: str
    rr1: float = 0.0              # populated by build_level_pool
    rr2: float = 0.0
    score: float = 0.0            # ranking score
    tags: List[str] = field(default_factory=list)

    def risk_per_unit(self) -> float:
        return abs(self.entry - self.stop)


@dataclass
class LevelPool:
    ideas: List[TradeIdea]
    notes: List[str] = field(default_factory=list)

    @property
    def best(self) -> Optional[TradeIdea]:
        return self.ideas[0] if self.ideas else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe(x, default=None):
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _rr(entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    return abs(target - entry) / risk


def _validate(idea: TradeIdea) -> bool:
    """Reject ideas whose geometry is broken."""
    if any(v is None or not math.isfinite(v) for v in (idea.entry, idea.stop,
                                                        idea.tp1, idea.tp2)):
        return False
    if idea.entry == idea.stop:
        return False
    if idea.bias == "LONG":
        if not (idea.stop < idea.entry < idea.tp1 <= idea.tp2):
            return False
    elif idea.bias == "SHORT":
        if not (idea.stop > idea.entry > idea.tp1 >= idea.tp2):
            return False
    else:
        return False
    return True


# ---------------------------------------------------------------------------
# Setup builders
# ---------------------------------------------------------------------------


def _vwap_reclaim_long(close: float, vwap: float, atr: float, pv: dict,
                       last_swing_low: Optional[float]) -> Optional[TradeIdea]:
    """Long the reclaim of session VWAP from below."""
    if vwap is None or close >= vwap:
        return None
    entry = vwap
    stop_candidates = [last_swing_low, vwap - 1.5 * atr]
    stop = min([s for s in stop_candidates if s is not None and s < entry],
               default=None)
    if stop is None:
        return None
    # First target: pivot point P, second: R1.
    tp1 = pv.get("P") or (entry + 1.0 * atr)
    tp2 = pv.get("R1") or (entry + 2.0 * atr)
    return TradeIdea(
        setup_id="VWAP_RECLAIM_LONG",
        setup_name="VWAP reclaim long",
        bias="LONG",
        entry=entry, stop=stop, tp1=tp1, tp2=tp2,
        rationale=(
            "Price below session VWAP — long on the reclaim through "
            "VWAP, stop under the most recent swing low (or 1.5×ATR "
            "below VWAP), targets at the prev-day pivot and R1."
        ),
        tags=["mean_reversion", "vwap"],
    )


def _vwap_rejection_short(close: float, vwap: float, atr: float, pv: dict,
                          last_swing_high: Optional[float]
                          ) -> Optional[TradeIdea]:
    """Short the rejection at session VWAP from above."""
    if vwap is None or close <= vwap:
        return None
    entry = vwap
    stop_candidates = [last_swing_high, vwap + 1.5 * atr]
    stop = max([s for s in stop_candidates if s is not None and s > entry],
               default=None)
    if stop is None:
        return None
    tp1 = pv.get("P") or (entry - 1.0 * atr)
    tp2 = pv.get("S1") or (entry - 2.0 * atr)
    return TradeIdea(
        setup_id="VWAP_REJECTION_SHORT",
        setup_name="VWAP rejection short",
        bias="SHORT",
        entry=entry, stop=stop, tp1=tp1, tp2=tp2,
        rationale=(
            "Price above session VWAP — short on rejection at VWAP, "
            "stop above last swing high (or 1.5×ATR above VWAP), "
            "targets at the prev-day pivot and S1."
        ),
        tags=["mean_reversion", "vwap"],
    )


def _or_breakout_long(or_high: float, or_low: float, atr: float,
                      pv: dict) -> Optional[TradeIdea]:
    if or_high is None or or_low is None or atr <= 0:
        return None
    entry = or_high + 0.10 * atr
    stop = or_low
    tp1 = pv.get("R1") or (entry + 1.5 * atr)
    tp2 = pv.get("R2") or (entry + 3.0 * atr)
    return TradeIdea(
        setup_id="OR_BREAKOUT_LONG",
        setup_name="Opening-range breakout long",
        bias="LONG",
        entry=entry, stop=stop, tp1=tp1, tp2=tp2,
        rationale=(
            "Long on a clear breakout above the session opening "
            "range with a 0.1×ATR cushion. Invalidation = OR low. "
            "Targets at R1 then R2 (prev-day pivots)."
        ),
        tags=["breakout", "opening_range"],
    )


def _or_breakdown_short(or_high: float, or_low: float, atr: float,
                        pv: dict) -> Optional[TradeIdea]:
    if or_high is None or or_low is None or atr <= 0:
        return None
    entry = or_low - 0.10 * atr
    stop = or_high
    tp1 = pv.get("S1") or (entry - 1.5 * atr)
    tp2 = pv.get("S2") or (entry - 3.0 * atr)
    return TradeIdea(
        setup_id="OR_BREAKDOWN_SHORT",
        setup_name="Opening-range breakdown short",
        bias="SHORT",
        entry=entry, stop=stop, tp1=tp1, tp2=tp2,
        rationale=(
            "Short on a clear breakdown below the opening range "
            "with a 0.1×ATR cushion. Invalidation = OR high. "
            "Targets at S1 then S2."
        ),
        tags=["breakout", "opening_range"],
    )


def _pivot_bounce_long(close: float, pv: dict, atr: float
                       ) -> Optional[TradeIdea]:
    s1 = pv.get("S1")
    if s1 is None or close <= s1 or atr <= 0:
        return None
    entry = s1
    stop = (pv.get("S2") or (s1 - 1.5 * atr))
    tp1 = pv.get("P") or (entry + 1.0 * atr)
    tp2 = pv.get("R1") or (entry + 2.0 * atr)
    return TradeIdea(
        setup_id="PIVOT_BOUNCE_LONG",
        setup_name="Pivot S1 bounce long",
        bias="LONG",
        entry=entry, stop=stop, tp1=tp1, tp2=tp2,
        rationale=(
            "Long on a defended hold of the prev-day S1 pivot. "
            "Stop at S2 (or 1.5×ATR below S1), targets at the pivot "
            "and R1."
        ),
        tags=["pivot", "support"],
    )


def _pivot_rejection_short(close: float, pv: dict, atr: float
                           ) -> Optional[TradeIdea]:
    r1 = pv.get("R1")
    if r1 is None or close >= r1 or atr <= 0:
        return None
    entry = r1
    stop = (pv.get("R2") or (r1 + 1.5 * atr))
    tp1 = pv.get("P") or (entry - 1.0 * atr)
    tp2 = pv.get("S1") or (entry - 2.0 * atr)
    return TradeIdea(
        setup_id="PIVOT_REJECTION_SHORT",
        setup_name="Pivot R1 rejection short",
        bias="SHORT",
        entry=entry, stop=stop, tp1=tp1, tp2=tp2,
        rationale=(
            "Short on rejection of the prev-day R1 pivot. Stop at "
            "R2 (or 1.5×ATR above R1), targets at the pivot and S1."
        ),
        tags=["pivot", "resistance"],
    )


def _bb_meanrev_long(close: float, bb_lower: Optional[float],
                     rsi: float, atr: float, vwap: Optional[float],
                     last_swing_low: Optional[float]
                     ) -> Optional[TradeIdea]:
    if bb_lower is None or atr <= 0 or rsi > 30:
        return None
    if close > bb_lower * 1.001:  # already off the lower band
        return None
    entry = close + 0.05 * atr   # tag back inside band
    stop = (last_swing_low - 0.25 * atr) if last_swing_low is not None \
        else (bb_lower - 1.0 * atr)
    tp1 = vwap if vwap is not None else (entry + 1.0 * atr)
    tp2 = entry + 2.0 * atr
    if not (stop < entry < tp1 <= tp2):
        return None
    return TradeIdea(
        setup_id="BB_MEANREV_LONG",
        setup_name="Bollinger lower-band mean-reversion long",
        bias="LONG",
        entry=entry, stop=stop, tp1=tp1, tp2=tp2,
        rationale=(
            "Stretched short with RSI <= 30 at the lower Bollinger "
            "band — fade with a small ATR add and target VWAP."
        ),
        tags=["mean_reversion", "bollinger"],
    )


def _bb_meanrev_short(close: float, bb_upper: Optional[float],
                      rsi: float, atr: float, vwap: Optional[float],
                      last_swing_high: Optional[float]
                      ) -> Optional[TradeIdea]:
    if bb_upper is None or atr <= 0 or rsi < 70:
        return None
    if close < bb_upper * 0.999:
        return None
    entry = close - 0.05 * atr
    stop = (last_swing_high + 0.25 * atr) if last_swing_high is not None \
        else (bb_upper + 1.0 * atr)
    tp1 = vwap if vwap is not None else (entry - 1.0 * atr)
    tp2 = entry - 2.0 * atr
    if not (stop > entry > tp1 >= tp2):
        return None
    return TradeIdea(
        setup_id="BB_MEANREV_SHORT",
        setup_name="Bollinger upper-band mean-reversion short",
        bias="SHORT",
        entry=entry, stop=stop, tp1=tp1, tp2=tp2,
        rationale=(
            "Stretched long with RSI >= 70 at the upper Bollinger "
            "band — fade with a small ATR cushion and target VWAP."
        ),
        tags=["mean_reversion", "bollinger"],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_level_pool(
    df: pd.DataFrame,
    ind: Mapping[str, object],
    min_rr: float = 1.5,
    htf_trend: Optional[str] = None,
    quant_p_up: Optional[float] = None,
) -> LevelPool:
    """Generate, validate, and rank candidate intraday trade ideas.

    Parameters
    ----------
    df : pandas.DataFrame
        Primary-timeframe OHLCV. The last bar is treated as "now".
    ind : Mapping[str, object]
        Output of :func:`compute_indicators`.
    min_rr : float
        Minimum R:R against TP1 to admit an idea into the pool.
    htf_trend : Optional[str]
        ``"up"`` / ``"down"`` / ``"chop"`` from the higher-timeframe
        analysis. Used as a tie-breaker for ranking, not a hard filter
        — gold mean-reverts often even against the HTF trend.
    quant_p_up : Optional[float]
        ``QuantSignal.p_up`` from :mod:`quant_baseline`. When
        provided, ideas whose bias agrees with the quant prior get
        a score boost.
    """
    if df is None or df.empty or not ind:
        return LevelPool(ideas=[], notes=["no data"])

    close = _safe(df["Close"].iloc[-1])
    atr_v = _safe(ind["atr14"].iloc[-1], 1.0) or 1.0
    vwap = _safe(ind["vwap"].iloc[-1])
    rsi_v = _safe(ind["rsi14"].iloc[-1], 50.0)
    pv = ind.get("pivots") or {}
    or_block = ind.get("or") or {}
    or_high = _safe(or_block.get("or_high"))
    or_low = _safe(or_block.get("or_low"))

    bb_df = ind.get("bb20")
    bb_upper = bb_lower = None
    if isinstance(bb_df, pd.DataFrame) and not bb_df.empty:
        bb_upper = _safe(bb_df["upper"].iloc[-1])
        bb_lower = _safe(bb_df["lower"].iloc[-1])

    last_swing_high = _safe(ind.get("last_swing_high"))
    last_swing_low = _safe(ind.get("last_swing_low"))

    candidates: List[Optional[TradeIdea]] = [
        _vwap_reclaim_long(close, vwap, atr_v, pv, last_swing_low),
        _vwap_rejection_short(close, vwap, atr_v, pv, last_swing_high),
        _or_breakout_long(or_high, or_low, atr_v, pv),
        _or_breakdown_short(or_high, or_low, atr_v, pv),
        _pivot_bounce_long(close, pv, atr_v),
        _pivot_rejection_short(close, pv, atr_v),
        _bb_meanrev_long(close, bb_lower, rsi_v, atr_v, vwap, last_swing_low),
        _bb_meanrev_short(close, bb_upper, rsi_v, atr_v, vwap, last_swing_high),
    ]

    pool: List[TradeIdea] = []
    rejections: List[str] = []
    for idea in candidates:
        if idea is None:
            continue
        if not _validate(idea):
            rejections.append(f"{idea.setup_id} (geometry invalid)")
            continue
        idea.rr1 = _rr(idea.entry, idea.stop, idea.tp1)
        idea.rr2 = _rr(idea.entry, idea.stop, idea.tp2)
        if idea.rr1 < min_rr:
            rejections.append(
                f"{idea.setup_id} (R:R {idea.rr1:.2f} < min {min_rr:.2f})"
            )
            continue
        # Score: base R:R against TP1, +0.5 if HTF trend agrees, +0.5 if
        # quant prior agrees with the bias direction, -0.25 if it disagrees.
        score = idea.rr1
        if htf_trend == "up" and idea.bias == "LONG":
            score += 0.5
        elif htf_trend == "down" and idea.bias == "SHORT":
            score += 0.5
        elif htf_trend in ("up", "down"):
            score -= 0.25
        if quant_p_up is not None:
            if idea.bias == "LONG" and quant_p_up >= 0.55:
                score += 0.5 * (quant_p_up - 0.5) * 2
            elif idea.bias == "SHORT" and quant_p_up <= 0.45:
                score += 0.5 * (0.5 - quant_p_up) * 2
            elif (idea.bias == "LONG" and quant_p_up < 0.45) or \
                 (idea.bias == "SHORT" and quant_p_up > 0.55):
                score -= 0.25
        idea.score = score
        pool.append(idea)

    pool.sort(key=lambda i: i.score, reverse=True)
    return LevelPool(ideas=pool, notes=rejections)


def level_pool_block(pool: LevelPool, ticker: str = "") -> str:
    """Render the level pool as a markdown table for the RM prompt."""
    if not pool.ideas:
        return (
            "### Deterministic level pool\n"
            "_No valid trade idea passed the R:R + geometry screen._\n"
            + (
                "_Rejected: " + ", ".join(pool.notes) + "_\n"
                if pool.notes else ""
            )
        )

    rows = [
        "| # | Setup | Bias | Entry | Stop | TP1 | TP2 | R:R₁ | R:R₂ | Score |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, idea in enumerate(pool.ideas, 1):
        rows.append(
            f"| {i} | {idea.setup_name} | {idea.bias} "
            f"| {idea.entry:.2f} | {idea.stop:.2f} "
            f"| {idea.tp1:.2f} | {idea.tp2:.2f} "
            f"| {idea.rr1:.2f} | {idea.rr2:.2f} | {idea.score:.2f} |"
        )

    rationale_block = "\n".join(
        f"- **{i.setup_id}** — {i.rationale}" for i in pool.ideas
    )

    return (
        f"### Deterministic level pool"
        + (f" for `{ticker}`" if ticker else "")
        + "\n"
        + "\n".join(rows)
        + "\n\nRationale:\n"
        + rationale_block
        + "\n\n_Rule for the Research Manager: pick exactly one row "
        "(by `setup_id`) or return FLAT. Do not invent prices not in "
        "this table — every level here is anchored to the indicator "
        "snapshot above._\n"
    )
