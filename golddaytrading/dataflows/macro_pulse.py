"""Intraday macro-pulse fetcher (DXY, real-yield proxy, VIX, ES).

Day-trading gold means watching a tight set of macro inverses on the
same intraday cadence as the gold tape — DXY ticks down, gold often
ticks up; ^TNX rips, gold takes a hit. This module pulls 1h candles
for the canonical drivers and renders a compact "what's leading"
block for the Macro Pulse Analyst.

Differences from the upstream daily macro tool:

* Hourly cadence, not daily — we want the *current* drift, not where
  things printed at NY close yesterday.
* Compact per-driver line (last value + 1h / 4h / 1d % change) so it
  fits in a few hundred tokens.
* No FRED dependency — FRED is daily and adds latency on intraday runs.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd

from golddaytrading.config import MACRO_PULSE_TICKERS
from golddaytrading.dataflows.intraday_data import fetch_intraday_ohlcv

logger = logging.getLogger(__name__)

# How a positive move in each driver affects gold (heuristic, used in
# the rendered block to give the LLM a quick "is this bullish?" tag).
_GOLD_BIAS = {
    "DX-Y.NYB": "bearish",      # strong USD => weaker gold
    "^TNX":     "bearish",      # higher nominal yields => weaker gold
    "^TYX":     "bearish",      # higher long yields => weaker gold
    "^VIX":     "bullish",      # risk-off => safe-haven gold bid
    "TIP":      "bullish",      # TIPS up => real yields down => gold up
    "ES=F":     "neutral",      # risk-on usually mild headwind, not direct
}


def _change_pct(series: pd.Series, bars_back: int) -> Optional[float]:
    if len(series) <= bars_back:
        return None
    prev = series.iloc[-(bars_back + 1)]
    last = series.iloc[-1]
    if prev == 0 or pd.isna(prev) or pd.isna(last):
        return None
    return float((last - prev) / prev * 100.0)


def fetch_macro_pulse() -> Dict[str, dict]:
    """Pull recent 1h candles for each macro driver and summarise.

    Returns a dict keyed by ticker; each value is
    ``{"last", "ts", "chg_1h", "chg_4h", "chg_1d", "bias"}``.
    Failed tickers map to a sentinel dict with ``"last": None``.
    """
    out: Dict[str, dict] = {}
    for tk in MACRO_PULSE_TICKERS:
        df = fetch_intraday_ohlcv(tk, timeframe="60m", bars=72)
        if df is None or df.empty:
            out[tk] = {
                "last": None, "ts": None,
                "chg_1h": None, "chg_4h": None, "chg_1d": None,
                "bias": _GOLD_BIAS.get(tk, "neutral"),
            }
            continue
        close = df["Close"]
        out[tk] = {
            "last": float(close.iloc[-1]),
            "ts": df.index[-1].strftime("%Y-%m-%d %H:%M UTC"),
            "chg_1h": _change_pct(close, 1),
            "chg_4h": _change_pct(close, 4),
            "chg_1d": _change_pct(close, 24),
            "bias": _GOLD_BIAS.get(tk, "neutral"),
        }
    return out


def macro_pulse_block(pulse: Dict[str, dict]) -> str:
    """Render the macro pulse as a markdown table for prompt injection."""
    if not pulse:
        return "_(macro pulse unavailable)_\n"

    def _fmt(v: Optional[float]) -> str:
        if v is None:
            return "  -- "
        sign = "+" if v >= 0 else ""
        return f"{sign}{v:.2f}%"

    rows = ["| Driver | Last | 1h | 4h | 1d | Gold bias if this is up |",
            "|---|---|---|---|---|---|"]
    label = {
        "DX-Y.NYB": "DXY (USD index)",
        "^TNX":     "10Y Treasury yield",
        "^TYX":     "30Y Treasury yield",
        "^VIX":     "VIX (vol)",
        "TIP":      "TIPS ETF (real-yield inverse)",
        "ES=F":     "S&P 500 futures",
    }
    for tk, info in pulse.items():
        last = f"{info['last']:.2f}" if info["last"] is not None else "  -- "
        rows.append(
            f"| {label.get(tk, tk)} | {last} "
            f"| {_fmt(info['chg_1h'])} | {_fmt(info['chg_4h'])} "
            f"| {_fmt(info['chg_1d'])} | {info['bias']} |"
        )
    return (
        "### Intraday macro pulse (1h cadence)\n"
        + "\n".join(rows)
        + "\n\n"
        "Reading: When DXY and yields move *down*, gold tends to "
        "move *up* on the same hour. A VIX spike with falling yields "
        "is the textbook safe-haven gold-bull regime.\n"
    )
