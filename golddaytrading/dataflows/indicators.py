"""Day-trading technical indicators.

We deliberately keep these as pure-pandas/numpy implementations so
the package has zero dependency on TA libraries (ta-lib, pandas-ta)
that often cause install pain. The set is intentionally compact —
just the indicators a gold day-trader actually reads off the screen:

* 20 / 50 / 200 EMA (trend filter on multiple horizons)
* RSI(14) (overbought/oversold, divergences)
* MACD(12,26,9) (momentum + signal cross)
* ATR(14) (volatility, stop sizing)
* Session VWAP (institutional reference, mean-reversion magnet)
* Opening Range (first hour of London / NY breakout level)
* Daily / weekly pivot points (S1/S2/R1/R2)

Each helper returns a Series (or DataFrame for multi-column outputs).
``compute_indicators`` runs the whole battery and returns a dict.
``indicator_summary_block`` renders the *latest* values into a
markdown block ready for the Technical Analyst prompt.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


# ---------- core helpers -----------------------------------------------------


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def macd(series: pd.Series,
         fast: int = 12, slow: int = 26, signal: int = 9
         ) -> pd.DataFrame:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "hist": hist})


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range. Expects columns ``High`` / ``Low`` / ``Close``."""
    h, l, c = df["High"], df["Low"], df["Close"]
    prev_close = c.shift(1)
    tr = pd.concat(
        [(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Anchored VWAP from the start of the most recent UTC day.

    Returns a series aligned with ``df.index`` whose values reset at
    each UTC-day boundary. For 24h FX-style instruments the
    "trading day" anchor is fuzzy; using UTC midnight is a reasonable
    default that aligns with most data vendors.
    """
    if "Volume" not in df.columns:
        df = df.assign(Volume=1.0)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].replace(0, np.nan).ffill().fillna(1.0)
    day_key = df.index.tz_convert("UTC").date if df.index.tz else df.index.date
    pv = (typical * vol).groupby(day_key).cumsum()
    cv = vol.groupby(day_key).cumsum()
    return pv / cv.replace(0, np.nan)


def opening_range(df: pd.DataFrame, minutes: int = 60) -> dict:
    """Return the high/low of the first ``minutes`` of the latest UTC day.

    Used by London-open and NY-open breakout strategies. When the
    timeframe is coarser than ``minutes``, this collapses to one bar.
    """
    if df.empty:
        return {"or_high": None, "or_low": None, "or_bars": 0}
    last_day = df.index[-1].date()
    todays = df[df.index.date == last_day]
    if todays.empty:
        return {"or_high": None, "or_low": None, "or_bars": 0}

    # Estimate the bar count covering ``minutes`` from the time delta
    # between consecutive bars.
    if len(todays) >= 2:
        bar_min = max(
            int((todays.index[1] - todays.index[0]).total_seconds() // 60), 1
        )
    else:
        bar_min = minutes
    n_bars = max(minutes // bar_min, 1)
    window = todays.head(n_bars)
    return {
        "or_high": float(window["High"].max()),
        "or_low": float(window["Low"].min()),
        "or_bars": int(len(window)),
    }


def daily_pivots(df: pd.DataFrame) -> dict:
    """Classic floor-trader pivots from the *previous* completed day.

    Expects an intraday-bar DataFrame (we group by date). When fewer
    than two distinct days are present we return ``None`` values.
    """
    if df.empty:
        return {"P": None, "R1": None, "S1": None, "R2": None, "S2": None}
    daily = df.resample("1D").agg(
        {"High": "max", "Low": "min", "Close": "last"}
    ).dropna()
    if len(daily) < 2:
        return {"P": None, "R1": None, "S1": None, "R2": None, "S2": None}
    prev = daily.iloc[-2]
    h, l, c = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    p = (h + l + c) / 3.0
    return {
        "P": p,
        "R1": 2 * p - l,
        "S1": 2 * p - h,
        "R2": p + (h - l),
        "S2": p - (h - l),
    }


# ---------- top-level battery ------------------------------------------------


def compute_indicators(df: pd.DataFrame) -> Dict[str, object]:
    """Run the whole indicator battery on an OHLCV DataFrame."""
    if df is None or df.empty:
        return {}
    close = df["Close"]
    out: Dict[str, object] = {
        "ema20":  ema(close, 20),
        "ema50":  ema(close, 50),
        "ema200": ema(close, 200),
        "rsi14":  rsi(close, 14),
        "macd":   macd(close),
        "atr14":  atr(df, 14),
        "vwap":   session_vwap(df),
        "or":     opening_range(df, 60),
        "pivots": daily_pivots(df),
    }
    return out


def indicator_summary_block(df: pd.DataFrame, ind: Dict[str, object]) -> str:
    """Render the latest reading of each indicator as a markdown block."""
    if df is None or df.empty or not ind:
        return "_(indicator computation skipped — no price data)_\n"

    last_close = float(df["Close"].iloc[-1])
    ema20 = float(ind["ema20"].iloc[-1])
    ema50 = float(ind["ema50"].iloc[-1])
    ema200 = float(ind["ema200"].iloc[-1])
    rsi_v = float(ind["rsi14"].iloc[-1])
    macd_df: pd.DataFrame = ind["macd"]  # type: ignore[assignment]
    macd_v = float(macd_df["macd"].iloc[-1])
    sig_v = float(macd_df["signal"].iloc[-1])
    hist_v = float(macd_df["hist"].iloc[-1])
    atr_v = float(ind["atr14"].iloc[-1])
    vwap_v = float(ind["vwap"].iloc[-1])

    or_block = ind["or"]
    pv = ind["pivots"]

    # Trend tag using EMA stack.
    if ema20 > ema50 > ema200:
        trend = "**uptrend** (EMA stack 20>50>200)"
    elif ema20 < ema50 < ema200:
        trend = "**downtrend** (EMA stack 20<50<200)"
    else:
        trend = "*mixed* (EMAs not aligned — chop / transition)"

    # VWAP relation.
    vwap_rel = "above" if last_close > vwap_v else "below"

    rsi_tag = (
        "overbought" if rsi_v >= 70
        else "oversold" if rsi_v <= 30
        else "neutral"
    )

    macd_tag = (
        "bullish (above signal, hist+)" if macd_v > sig_v and hist_v > 0
        else "bearish (below signal, hist-)" if macd_v < sig_v and hist_v < 0
        else "transitioning"
    )

    or_line = (
        f"- Opening range (first ~60min): "
        f"H `{or_block['or_high']:.2f}` / L `{or_block['or_low']:.2f}` "
        f"({or_block['or_bars']} bars)\n"
        if or_block.get("or_high") is not None else
        "- Opening range: _(insufficient intraday history today)_\n"
    )

    pv_line = (
        f"- Pivots (prev-day): "
        f"P `{pv['P']:.2f}` / R1 `{pv['R1']:.2f}` / R2 `{pv['R2']:.2f}` / "
        f"S1 `{pv['S1']:.2f}` / S2 `{pv['S2']:.2f}`\n"
        if pv.get("P") is not None else
        "- Pivots: _(need >=2 days of bars)_\n"
    )

    return (
        f"### Indicator snapshot\n"
        f"- Last close: `{last_close:.2f}`\n"
        f"- Trend: {trend}  |  EMA20 `{ema20:.2f}` · EMA50 `{ema50:.2f}` · EMA200 `{ema200:.2f}`\n"
        f"- RSI(14): `{rsi_v:.1f}` ({rsi_tag})\n"
        f"- MACD: `{macd_v:.2f}` / signal `{sig_v:.2f}` / hist `{hist_v:.2f}` — {macd_tag}\n"
        f"- ATR(14): `{atr_v:.2f}` — implied stop ~`{atr_v * 1.5:.2f}` (1.5×ATR)\n"
        f"- Session VWAP: `{vwap_v:.2f}` — price is **{vwap_rel}** VWAP\n"
        f"{or_line}"
        f"{pv_line}"
    )
