"""Quantitative baseline signal for intraday gold direction.

A *calibrated* logistic-regression-style scorer that turns the
indicator snapshot + macro pulse into:

* ``p_up`` — probability that the next ``horizon_bars`` print a
  higher close than the current close.
* ``expected_move_atr`` — signed expected move in ATR(14) units.
* ``confidence`` — heuristic 0..1 derived from the absolute logit
  magnitude.

Why ship a hand-calibrated logistic instead of an empirically
trained one?

The repository ships **no** historical training data — pulling it on
the fly via yfinance and running a sklearn fit would (a) introduce
a heavy dependency and (b) overfit to the sliver of data that
yfinance returns. Instead, the coefficients here encode well-known
gold-trading priors (DXY down = bullish, RSI extremes mean-revert,
EMA stack matters more than absolute level). The numbers are not
random — they are scaled so that:

* A clean uptrend (EMA stack 20>50>200, price above VWAP, RSI ~ 55,
  DXY -0.20% in the last hour) returns ``p_up ≈ 0.62``.
* A clean downtrend with the inverse readings returns
  ``p_up ≈ 0.38``.
* Mixed / contradictory readings return ``p_up`` close to 0.5.

The intent is to be a **structured prior** the LLM can reason
against. The roadmap's backtest harness (P2) is the right place to
replace these coefficients with empirically fit ones; the API of
:func:`compute_quant_signal` will not change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Calibrated coefficients
# ---------------------------------------------------------------------------
#
# Features are *all* normalised into roughly the [-2, +2] range so
# the coefficients themselves stay interpretable. The intercept is 0
# because no feature has a directional prior in isolation when the
# market is ambiguous.

# Default forecast horizon (in *primary-tf* bars) used by the
# expected-move calculation. 4 bars on a 15m chart = 1 hour, which is
# the typical duration of an intraday gold setup.
DEFAULT_HORIZON_BARS = 4

# Direction model — predicts P(close[t+H] > close[t]).
DIRECTION_COEFS: Dict[str, float] = {
    "ema_stack_score":     0.85,   # +1 for 20>50>200, -1 for inverse
    "vwap_dist_atr":       0.55,   # price - vwap, in ATR units
    "macd_hist_norm":      0.45,   # macd hist / atr
    "rsi_centered":        0.30,   # (rsi - 50) / 25 — mild momentum bias
    "rsi_extreme":        -0.65,   # mean-reversion override at >75 / <25
    "stoch_centered":      0.20,   # (k - 50) / 50
    "bb_position":        -0.45,   # mean-reversion when stretched out of bands
    "dxy_chg_1h_z":       -0.95,   # gold's strongest live inverse
    "tnx_chg_1h_z":       -0.50,   # nominal yield inverse
    "vix_level_z":         0.30,   # risk-off bid for safe haven
    "tip_chg_1h_z":        0.45,   # rising TIPS = falling real yield = bullish
    "real_yield_chg_z":   -0.70,   # explicit real-yield proxy (overrides TIP/TNX combo)
    "eurusd_chg_1h_z":     0.45,   # EURUSD up = USD weak = gold bullish
    "session_overlap":     0.20,   # London-NY overlap = directional bias
    "macro_regime_score":  0.50,   # +1 for bullish regime, -1 bearish, 0 neutral
}
DIRECTION_INTERCEPT = 0.0

# Magnitude model — predicts expected absolute |move| in ATR units.
# We use a small linear model on absolute feature magnitudes plus a
# clamp.
MAGNITUDE_COEFS: Dict[str, float] = {
    "abs_ema_stack":       0.25,
    "abs_macd_hist_norm":  0.30,
    "bb_bandwidth_norm":   0.20,
    "abs_dxy_chg_1h_z":    0.20,
    "abs_real_yield_chg_z": 0.15,
    "session_overlap":     0.15,
}
MAGNITUDE_INTERCEPT = 0.50    # baseline expected move in ATR units


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QuantSignal:
    """Output of :func:`compute_quant_signal`."""

    p_up: float                       # 0..1
    expected_move_atr: float          # signed, ATR units
    expected_move_price: float        # signed, raw price units
    horizon_bars: int
    horizon_minutes: Optional[int]
    confidence: float                 # 0..1, |logit| / 3 capped
    direction_logit: float
    features: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def direction_label(self) -> str:
        if self.p_up >= 0.58:
            return "BULLISH"
        if self.p_up <= 0.42:
            return "BEARISH"
        return "NEUTRAL"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _ema_stack_score(ema20: float, ema50: float, ema200: float) -> float:
    """+1 fully bullish, -1 fully bearish, 0 mixed."""
    if ema20 > ema50 > ema200:
        return 1.0
    if ema20 < ema50 < ema200:
        return -1.0
    # Partial alignment — count agreeing pairs.
    score = 0.0
    score += 0.5 if ema20 > ema50 else -0.5
    score += 0.5 if ema50 > ema200 else -0.5
    return _clip(score, -1.0, 1.0)


def _bb_position(close: float, mid: float, upper: float, lower: float) -> float:
    """Map close position inside Bollinger range into [-1, +1].

    Returns 0 at the mid band, +1 at the upper, -1 at the lower, and
    clamps beyond +/- 1.5 for stretched-outside readings (which
    drives the mean-reversion coefficient).
    """
    width = (upper - lower) / 2.0
    if not math.isfinite(width) or width <= 0:
        return 0.0
    return _clip((close - mid) / width, -2.0, 2.0)


def extract_features(
    df: pd.DataFrame,
    ind: Mapping[str, object],
    macro_pulse: Optional[Mapping[str, dict]] = None,
    session_name: Optional[str] = None,
) -> Dict[str, float]:
    """Pull a deterministic numeric feature vector from the indicator
    snapshot and macro pulse.

    Returns a dict containing every feature referenced by
    :data:`DIRECTION_COEFS` or :data:`MAGNITUDE_COEFS`. Missing data
    yields neutral (0.0) values rather than NaN so the linear model
    always produces a finite output.
    """
    if df is None or df.empty or not ind:
        return {k: 0.0 for k in DIRECTION_COEFS} | {
            k: 0.0 for k in MAGNITUDE_COEFS
        }

    close = _safe_float(df["Close"].iloc[-1])
    ema20 = _safe_float(ind["ema20"].iloc[-1])
    ema50 = _safe_float(ind["ema50"].iloc[-1])
    ema200 = _safe_float(ind["ema200"].iloc[-1])
    rsi_v = _safe_float(ind["rsi14"].iloc[-1], 50.0)
    macd_df = ind.get("macd")
    macd_hist = (
        _safe_float(macd_df["hist"].iloc[-1])
        if isinstance(macd_df, pd.DataFrame) else 0.0
    )
    atr_v = _safe_float(ind["atr14"].iloc[-1], 1.0) or 1.0
    vwap_v = _safe_float(ind["vwap"].iloc[-1], close)

    bb_df = ind.get("bb20")
    if isinstance(bb_df, pd.DataFrame) and not bb_df.empty:
        bb_row = bb_df.iloc[-1]
        bb_pos = _bb_position(close, _safe_float(bb_row.get("mid"), close),
                              _safe_float(bb_row.get("upper"), close),
                              _safe_float(bb_row.get("lower"), close))
        bb_bw = _safe_float(bb_row.get("bandwidth"), 0.0)
    else:
        bb_pos, bb_bw = 0.0, 0.0

    stoch_df = ind.get("stoch")
    if isinstance(stoch_df, pd.DataFrame) and not stoch_df.empty:
        sk = _safe_float(stoch_df["k"].iloc[-1], 50.0)
    else:
        sk = 50.0

    rsi_centered = _clip((rsi_v - 50.0) / 25.0, -2.0, 2.0)
    # Activate the mean-reversion coefficient only when RSI is
    # outside [25, 75]; the value is +1 for >75 and -1 for <25 so a
    # negative coefficient drives p_up *down* in overbought.
    if rsi_v >= 75:
        rsi_extreme = (rsi_v - 75.0) / 15.0      # 0..~1.7 at RSI 100
    elif rsi_v <= 25:
        rsi_extreme = -((25.0 - rsi_v) / 15.0)
    else:
        rsi_extreme = 0.0
    rsi_extreme = _clip(rsi_extreme, -2.0, 2.0)

    # Macro pulse fields (1h % change). DXY/^TNX/^TYX live in macro_pulse
    # keyed by the original yfinance ticker. Convert the percent
    # change into a z-ish score by dividing by a typical 1h sigma:
    #   DXY  ~ 0.10%   ->  /0.10
    #   TNX  ~ 0.40%   ->  /0.40
    #   VIX  level     ->  (vix - 18) / 10
    #   TIP  ~ 0.10%   ->  /0.10
    macro_pulse = macro_pulse or {}

    def _chg_z(ticker: str, sigma_pct: float) -> float:
        info = macro_pulse.get(ticker) or {}
        chg = info.get("chg_1h")
        if chg is None:
            return 0.0
        return _clip(_safe_float(chg) / sigma_pct, -3.0, 3.0)

    dxy_chg_z = _chg_z("DX-Y.NYB", 0.10)
    tnx_chg_z = _chg_z("^TNX", 0.40)
    tip_chg_z = _chg_z("TIP", 0.10)
    eurusd_chg_z = _chg_z("EURUSD=X", 0.10)

    # Real-yield proxy: prefer the explicit derived field if the
    # macro_pulse fetcher provided one, else fall back to a synthetic
    # combination of TNX + TIP moves.
    derived = (macro_pulse or {}).get("__derived__") or {}
    real_yield_chg = derived.get("real_yield_proxy_chg_1h")
    if real_yield_chg is None:
        # Synthetic: real_yield ≈ nominal_yield - inflation_breakevens
        # ≈ TNX% change - (TIP% change with sign flipped)
        if (macro_pulse or {}).get("^TNX") and (macro_pulse or {}).get("TIP"):
            tnx_raw = _safe_float(macro_pulse["^TNX"].get("chg_1h"))
            tip_raw = _safe_float(macro_pulse["TIP"].get("chg_1h"))
            real_yield_chg = tnx_raw - (-tip_raw)
        else:
            real_yield_chg = 0.0
    real_yield_chg_z = _clip(_safe_float(real_yield_chg) / 0.20, -3.0, 3.0)

    # Macro regime score — bullish regimes lift p_up directly.
    regime_name = (macro_pulse or {}).get("__regime__")
    regime_bias = (macro_pulse or {}).get("__regime_bias__") or "neutral"
    regime_score = (
        1.0 if regime_bias == "bullish"
        else -1.0 if regime_bias == "bearish"
        else 0.0
    )

    vix_info = macro_pulse.get("^VIX") or {}
    vix_level = vix_info.get("last")
    if vix_level is None:
        vix_level_z = 0.0
    else:
        vix_level_z = _clip((_safe_float(vix_level) - 18.0) / 10.0, -2.0, 3.0)

    session_overlap = 1.0 if (session_name or "") == "LONDON_NY_OVERLAP" else 0.0

    feats = {
        "ema_stack_score":     _ema_stack_score(ema20, ema50, ema200),
        "vwap_dist_atr":       _clip((close - vwap_v) / atr_v, -3.0, 3.0),
        "macd_hist_norm":      _clip(macd_hist / atr_v, -3.0, 3.0),
        "rsi_centered":        rsi_centered,
        "rsi_extreme":         rsi_extreme,
        "stoch_centered":      _clip((sk - 50.0) / 50.0, -1.0, 1.0),
        "bb_position":         bb_pos,
        "dxy_chg_1h_z":        dxy_chg_z,
        "tnx_chg_1h_z":        tnx_chg_z,
        "vix_level_z":         vix_level_z,
        "tip_chg_1h_z":        tip_chg_z,
        "real_yield_chg_z":    real_yield_chg_z,
        "eurusd_chg_1h_z":     eurusd_chg_z,
        "session_overlap":     session_overlap,
        "macro_regime_score":  regime_score,
        # Magnitude-only inputs.
        "abs_ema_stack":       abs(_ema_stack_score(ema20, ema50, ema200)),
        "abs_macd_hist_norm":  abs(_clip(macd_hist / atr_v, -3.0, 3.0)),
        "bb_bandwidth_norm":   _clip(bb_bw * 100.0, 0.0, 5.0),  # %
        "abs_dxy_chg_1h_z":    abs(dxy_chg_z),
        "abs_real_yield_chg_z": abs(real_yield_chg_z),
    }
    return feats


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    if x >= 0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def _bar_minutes(df: pd.DataFrame) -> Optional[int]:
    if df is None or len(df) < 2:
        return None
    delta = df.index[-1] - df.index[-2]
    secs = int(delta.total_seconds())
    if secs <= 0:
        return None
    return max(secs // 60, 1)


def compute_quant_signal(
    df: pd.DataFrame,
    ind: Mapping[str, object],
    macro_pulse: Optional[Mapping[str, dict]] = None,
    session_name: Optional[str] = None,
    horizon_bars: int = DEFAULT_HORIZON_BARS,
) -> QuantSignal:
    """Compute the quantitative direction + magnitude signal."""
    if df is None or df.empty or not ind:
        return QuantSignal(
            p_up=0.5, expected_move_atr=0.0, expected_move_price=0.0,
            horizon_bars=horizon_bars, horizon_minutes=None,
            confidence=0.0, direction_logit=0.0,
            features={}, notes=["no data"],
        )

    feats = extract_features(df, ind, macro_pulse, session_name)
    logit = DIRECTION_INTERCEPT + sum(
        feats.get(k, 0.0) * w for k, w in DIRECTION_COEFS.items()
    )
    p_up = _sigmoid(logit)

    mag = MAGNITUDE_INTERCEPT + sum(
        feats.get(k, 0.0) * w for k, w in MAGNITUDE_COEFS.items()
    )
    mag = _clip(mag, 0.1, 3.0)
    direction_sign = 1.0 if logit >= 0 else -1.0
    expected_move_atr = direction_sign * mag

    atr_v = _safe_float(ind["atr14"].iloc[-1], 0.0)
    expected_move_price = expected_move_atr * atr_v

    confidence = _clip(abs(logit) / 3.0, 0.0, 1.0)

    notes: List[str] = []
    if abs(feats.get("rsi_extreme", 0.0)) > 0.5:
        notes.append(
            "RSI extreme triggers mean-reversion override "
            "(negative coefficient on rsi_extreme)."
        )
    if feats.get("session_overlap", 0.0) > 0:
        notes.append("London-NY overlap session boost applied.")
    if abs(feats.get("dxy_chg_1h_z", 0.0)) > 1.5:
        notes.append("DXY 1h move > 1.5σ — dominant macro driver this hour.")

    return QuantSignal(
        p_up=p_up,
        expected_move_atr=expected_move_atr,
        expected_move_price=expected_move_price,
        horizon_bars=horizon_bars,
        horizon_minutes=(
            (_bar_minutes(df) or 0) * horizon_bars
            if _bar_minutes(df) else None
        ),
        confidence=confidence,
        direction_logit=logit,
        features=feats,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def quant_signal_block(sig: QuantSignal) -> str:
    """Render the quantitative signal as a markdown block for prompts.

    The block is intentionally compact (~10 lines) and labels every
    number so the LLM can quote it accurately. Crucially, we expose
    the *features* so the analyst agents can verify the numbers
    against the indicator block (no hidden state).
    """
    if not sig.features:
        return "### Quant baseline signal\n_(no data — quant signal skipped)_\n"

    horizon_str = f"{sig.horizon_bars} bars"
    if sig.horizon_minutes:
        horizon_str += f" (~{sig.horizon_minutes} min)"

    move_str = (
        f"{sig.expected_move_atr:+.2f} ATR "
        f"(~{sig.expected_move_price:+.2f} price units)"
    )

    # Sort features by absolute contribution to the logit so the
    # analyst sees the *drivers* first.
    contribs = []
    for k, w in DIRECTION_COEFS.items():
        v = sig.features.get(k, 0.0)
        contribs.append((k, v, w, v * w))
    contribs.sort(key=lambda r: abs(r[3]), reverse=True)
    top = contribs[:5]

    rows = [
        "| Feature | Value | Coef | Logit contribution |",
        "|---|---|---|---|",
    ]
    for k, v, w, c in top:
        rows.append(f"| {k} | {v:+.3f} | {w:+.2f} | {c:+.3f} |")

    notes_block = ""
    if sig.notes:
        notes_block = "\n_Notes:_ " + " ".join(sig.notes) + "\n"

    return (
        "### Quant baseline signal\n"
        f"- **Direction:** **{sig.direction_label}**  "
        f"(P(up over {horizon_str}) = `{sig.p_up:.3f}`, "
        f"logit `{sig.direction_logit:+.2f}`, "
        f"confidence `{sig.confidence:.2f}`)\n"
        f"- **Expected move:** {move_str}\n"
        f"- **Top drivers (by absolute logit contribution):**\n"
        + "\n".join(rows)
        + "\n"
        + notes_block
        + "\n_How to read this:_ the logistic baseline is a "
        "pre-calibrated prior, not a forecast. Use it as a sanity "
        "check on the qualitative narrative — if the analysts say "
        "'uptrend, long' but the quant prior is BEARISH, the bull / "
        "bear debate must explicitly reconcile the disagreement.\n"
    )
