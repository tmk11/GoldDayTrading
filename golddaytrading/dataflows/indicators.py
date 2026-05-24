"""Day-trading technical indicators.

We deliberately keep these as pure-pandas/numpy implementations so
the package has zero dependency on TA libraries (ta-lib, pandas-ta)
that often cause install pain. The set covers the indicators a gold
day-trader actually reads off the screen:

* 20 / 50 / 200 EMA (trend filter on multiple horizons)
* RSI(14) (overbought/oversold, divergences)
* MACD(12,26,9) (momentum + signal cross)
* ATR(14) (volatility, stop sizing)
* Bollinger Bands(20, 2) (mean-reversion / squeeze detection)
* Stochastic(14, 3, 3) (overbought confirmation, divergence cross-check)
* Session VWAP anchored to the 5pm NY close (22:00 UTC)
* Anchored VWAP from the most recent significant swing high / low
  (institutional reference once a fresh leg starts)
* Opening Range (first hour of the *current trading day*)
* Daily / weekly / monthly pivot points (S1/S2/R1/R2 + Camarilla)
* Swing high / low fractal detection (auto market-structure)

Each helper returns a Series (or DataFrame for multi-column outputs).
``compute_indicators`` runs the whole battery and returns a dict.
``indicator_summary_block`` renders the *latest* values into a
markdown block ready for the Technical Analyst prompt.
"""

from __future__ import annotations

from typing import Dict, List, Optional

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


def bollinger_bands(series: pd.Series,
                    length: int = 20,
                    num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands — mid SMA(length) +/- num_std * rolling stdev.

    Adds a derived ``bandwidth`` column = (upper-lower)/mid which is
    the standard "squeeze" detector — sub-historical bandwidth often
    precedes intraday expansion.
    """
    mid = series.rolling(length, min_periods=length).mean()
    sd = series.rolling(length, min_periods=length).std(ddof=0)
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    bandwidth = (upper - lower) / mid.replace(0, np.nan)
    return pd.DataFrame(
        {"mid": mid, "upper": upper, "lower": lower, "bandwidth": bandwidth}
    )


def stochastic(df: pd.DataFrame,
               k_length: int = 14,
               k_smooth: int = 3,
               d_smooth: int = 3) -> pd.DataFrame:
    """Stochastic %K / %D oscillator.

    Returns columns ``k`` (smoothed %K) and ``d`` (signal). Used as a
    secondary momentum confirmation alongside RSI.
    """
    h = df["High"].rolling(k_length, min_periods=k_length).max()
    l = df["Low"].rolling(k_length, min_periods=k_length).min()
    rng = (h - l).replace(0, np.nan)
    raw_k = 100 * (df["Close"] - l) / rng
    k = raw_k.rolling(k_smooth, min_periods=1).mean()
    d = k.rolling(d_smooth, min_periods=1).mean()
    return pd.DataFrame({"k": k, "d": d})


def _session_id(idx: pd.DatetimeIndex, anchor_hour_utc: int) -> np.ndarray:
    """Integer session id that increments at each anchor crossing."""
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    shifted = idx - pd.Timedelta(hours=anchor_hour_utc)
    return shifted.normalize().asi8


def session_vwap(
    df: pd.DataFrame,
    anchor_hour_utc: int = 22,
) -> pd.Series:
    """Anchored VWAP from the most recent NY-close boundary.

    For 23h FX-style instruments the institutional convention for
    the "trading day" is the **5pm New York close** — i.e. 22:00 UTC
    (21:00 UTC during US daylight saving). We anchor the cumulative
    sums there so the resulting VWAP matches what desk traders see
    on Bloomberg / TradingView with the default "Session VWAP" tool.

    yfinance returns ``Volume == 0`` for spot FX pairs like
    ``XAUUSD=X``. Falling back to ``Volume = 1`` would silently turn
    this into a typical-price moving average. For those cases we
    instead return a typical-price-only mean that still resets at
    the session boundary — the closest honest approximation.
    """
    if df.empty:
        return pd.Series(dtype="float64", index=df.index)

    typical = (df["High"] + df["Low"] + df["Close"]) / 3.0
    session_id = _session_id(df.index, anchor_hour_utc)

    has_volume = (
        "Volume" in df.columns
        and df["Volume"].fillna(0).abs().sum() > 0
    )

    if has_volume:
        vol = df["Volume"].astype("float64").fillna(0.0)
        vol = vol.replace(0, np.nan).ffill().bfill().fillna(1.0)
        pv = (typical * vol).groupby(session_id).cumsum()
        cv = vol.groupby(session_id).cumsum()
        return pv / cv.replace(0, np.nan)

    grouper = pd.Series(session_id, index=df.index)
    cum_sum = typical.groupby(grouper).cumsum()
    cum_count = typical.groupby(grouper).cumcount() + 1
    return cum_sum / cum_count


def detect_swings(df: pd.DataFrame, left: int = 3, right: int = 3
                  ) -> pd.DataFrame:
    """Fractal swing-high / swing-low detection.

    A bar is a swing high if its high is strictly greater than every
    high in the ``left`` bars before it and the ``right`` bars after.
    Returns a DataFrame with boolean columns ``swing_high`` and
    ``swing_low``, aligned with ``df.index``. The ``right`` rightmost
    bars are always False (cannot be confirmed yet).
    """
    n = len(df)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    if n < left + right + 1:
        return pd.DataFrame(
            {"swing_high": swing_high, "swing_low": swing_low},
            index=df.index,
        )
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    for i in range(left, n - right):
        window_h = highs[i - left:i + right + 1]
        window_l = lows[i - left:i + right + 1]
        if highs[i] == window_h.max() and (window_h == highs[i]).sum() == 1:
            swing_high[i] = True
        if lows[i] == window_l.min() and (window_l == lows[i]).sum() == 1:
            swing_low[i] = True
    return pd.DataFrame(
        {"swing_high": swing_high, "swing_low": swing_low},
        index=df.index,
    )


def anchored_vwap(df: pd.DataFrame, anchor_idx: int) -> pd.Series:
    """Volume-weighted average price anchored at ``anchor_idx``.

    Returns a series aligned with ``df.index``; values before the
    anchor are ``NaN``. Same volume-fallback behaviour as
    :func:`session_vwap` for FX instruments without exchange volume.
    """
    n = len(df)
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    if anchor_idx < 0 or anchor_idx >= n:
        return out
    sub = df.iloc[anchor_idx:]
    typical = (sub["High"] + sub["Low"] + sub["Close"]) / 3.0
    has_volume = (
        "Volume" in sub.columns
        and sub["Volume"].fillna(0).abs().sum() > 0
    )
    if has_volume:
        vol = sub["Volume"].astype("float64").fillna(0.0)
        vol = vol.replace(0, np.nan).ffill().bfill().fillna(1.0)
        pv = (typical * vol).cumsum()
        cv = vol.cumsum()
        out.iloc[anchor_idx:] = (pv / cv.replace(0, np.nan)).to_numpy()
    else:
        out.iloc[anchor_idx:] = (
            typical.cumsum() / np.arange(1, len(sub) + 1)
        ).to_numpy()
    return out


def opening_range(df: pd.DataFrame, minutes: int = 60,
                  anchor_hour_utc: int = 22) -> dict:
    """High/low of the first ``minutes`` of the *current trading day*.

    The trading day is defined by ``anchor_hour_utc`` so this also
    aligns to the 5pm NY close convention by default. Used by
    London-open and NY-open breakout strategies. When the timeframe
    is coarser than ``minutes``, this collapses to one bar.
    """
    if df.empty:
        return {"or_high": None, "or_low": None, "or_bars": 0}
    sid = _session_id(df.index, anchor_hour_utc)
    last_session = sid[-1]
    mask = sid == last_session
    todays = df.loc[mask]
    if todays.empty:
        return {"or_high": None, "or_low": None, "or_bars": 0}

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


def _floor_pivots(h: float, l: float, c: float) -> dict:
    p = (h + l + c) / 3.0
    return {
        "P":  p,
        "R1": 2 * p - l,
        "S1": 2 * p - h,
        "R2": p + (h - l),
        "S2": p - (h - l),
    }


def _camarilla_pivots(h: float, l: float, c: float) -> dict:
    rng = h - l
    return {
        "C_R3": c + rng * 1.1 / 4.0,
        "C_R2": c + rng * 1.1 / 6.0,
        "C_R1": c + rng * 1.1 / 12.0,
        "C_S1": c - rng * 1.1 / 12.0,
        "C_S2": c - rng * 1.1 / 6.0,
        "C_S3": c - rng * 1.1 / 4.0,
    }


def daily_pivots(df: pd.DataFrame) -> dict:
    """Floor-trader and Camarilla pivots from the *previous* completed day."""
    empty = {"P": None, "R1": None, "S1": None, "R2": None, "S2": None,
             "C_R1": None, "C_R2": None, "C_R3": None,
             "C_S1": None, "C_S2": None, "C_S3": None}
    if df.empty:
        return empty
    daily = df.resample("1D").agg(
        {"High": "max", "Low": "min", "Close": "last"}
    ).dropna()
    if len(daily) < 2:
        return empty
    prev = daily.iloc[-2]
    h, l, c = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    out = _floor_pivots(h, l, c)
    out.update(_camarilla_pivots(h, l, c))
    return out


def weekly_pivots(df: pd.DataFrame) -> dict:
    """Floor-trader pivots from the *previous* completed ISO week."""
    empty = {"P": None, "R1": None, "S1": None, "R2": None, "S2": None}
    if df.empty:
        return empty
    weekly = df.resample("1W").agg(
        {"High": "max", "Low": "min", "Close": "last"}
    ).dropna()
    if len(weekly) < 2:
        return empty
    prev = weekly.iloc[-2]
    return _floor_pivots(
        float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    )


def monthly_pivots(df: pd.DataFrame) -> dict:
    """Floor-trader pivots from the *previous* completed calendar month."""
    empty = {"P": None, "R1": None, "S1": None, "R2": None, "S2": None}
    if df.empty:
        return empty
    monthly = df.resample("1ME").agg(
        {"High": "max", "Low": "min", "Close": "last"}
    ).dropna()
    if len(monthly) < 2:
        return empty
    prev = monthly.iloc[-2]
    return _floor_pivots(
        float(prev["High"]), float(prev["Low"]), float(prev["Close"])
    )


# ---------- top-level battery ------------------------------------------------


def compute_indicators(
    df: pd.DataFrame,
    vwap_anchor_hour_utc: int = 22,
) -> Dict[str, object]:
    """Run the whole indicator battery on an OHLCV DataFrame."""
    if df is None or df.empty:
        return {}
    close = df["Close"]

    swings = detect_swings(df, left=3, right=3)

    # Anchor a fresh VWAP from the most recent confirmed swing high
    # AND the most recent confirmed swing low — gives the analyst two
    # institutional reference lines once a leg is in motion.
    sh_idx = np.where(swings["swing_high"].to_numpy())[0]
    sl_idx = np.where(swings["swing_low"].to_numpy())[0]
    avwap_hi = (anchored_vwap(df, int(sh_idx[-1]))
                if len(sh_idx) else pd.Series(np.nan, index=df.index))
    avwap_lo = (anchored_vwap(df, int(sl_idx[-1]))
                if len(sl_idx) else pd.Series(np.nan, index=df.index))

    out: Dict[str, object] = {
        "ema20":  ema(close, 20),
        "ema50":  ema(close, 50),
        "ema200": ema(close, 200),
        "rsi14":  rsi(close, 14),
        "macd":   macd(close),
        "atr14":  atr(df, 14),
        "bb20":   bollinger_bands(close, 20, 2.0),
        "stoch":  stochastic(df, 14, 3, 3),
        "vwap":   session_vwap(df, anchor_hour_utc=vwap_anchor_hour_utc),
        "or":     opening_range(df, 60, anchor_hour_utc=vwap_anchor_hour_utc),
        "pivots": daily_pivots(df),
        "weekly_pivots": weekly_pivots(df),
        "monthly_pivots": monthly_pivots(df),
        "swings": swings,
        "avwap_swing_high": avwap_hi,
        "avwap_swing_low":  avwap_lo,
        "last_swing_high":  float(df["High"].iloc[sh_idx[-1]])
                            if len(sh_idx) else None,
        "last_swing_low":   float(df["Low"].iloc[sl_idx[-1]])
                            if len(sl_idx) else None,
    }
    return out


# ---------- summary block ---------------------------------------------------


def _bb_tag(close: float, bb_row: pd.Series) -> str:
    if pd.isna(bb_row.get("upper")) or pd.isna(bb_row.get("lower")):
        return "warming up"
    if close >= bb_row["upper"]:
        return "above upper band (stretched)"
    if close <= bb_row["lower"]:
        return "below lower band (stretched)"
    return "inside bands"


def _stoch_tag(k: float, d: float) -> str:
    if pd.isna(k) or pd.isna(d):
        return "warming up"
    if k >= 80:
        return f"overbought (k {k:.0f}/d {d:.0f})"
    if k <= 20:
        return f"oversold (k {k:.0f}/d {d:.0f})"
    return f"neutral (k {k:.0f}/d {d:.0f})"


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

    bb_df: pd.DataFrame = ind["bb20"]  # type: ignore[assignment]
    bb_row = bb_df.iloc[-1]
    bb_bw = bb_row.get("bandwidth")
    bb_bw_str = f"{bb_bw * 100:.2f}%" if pd.notna(bb_bw) else "—"

    stoch_df: pd.DataFrame = ind["stoch"]  # type: ignore[assignment]
    sk = float(stoch_df["k"].iloc[-1])
    sd = float(stoch_df["d"].iloc[-1])

    or_block = ind["or"]
    pv = ind["pivots"]
    wpv = ind.get("weekly_pivots", {}) or {}
    mpv = ind.get("monthly_pivots", {}) or {}
    avwap_hi_s: pd.Series = ind.get("avwap_swing_high")  # type: ignore[assignment]
    avwap_lo_s: pd.Series = ind.get("avwap_swing_low")  # type: ignore[assignment]
    avwap_hi = (
        float(avwap_hi_s.iloc[-1])
        if avwap_hi_s is not None and pd.notna(avwap_hi_s.iloc[-1]) else None
    )
    avwap_lo = (
        float(avwap_lo_s.iloc[-1])
        if avwap_lo_s is not None and pd.notna(avwap_lo_s.iloc[-1]) else None
    )

    if ema20 > ema50 > ema200:
        trend = "**uptrend** (EMA stack 20>50>200)"
    elif ema20 < ema50 < ema200:
        trend = "**downtrend** (EMA stack 20<50<200)"
    else:
        trend = "*mixed* (EMAs not aligned — chop / transition)"

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
        f"- Opening range (first ~60min of session): "
        f"H `{or_block['or_high']:.2f}` / L `{or_block['or_low']:.2f}` "
        f"({or_block['or_bars']} bars)\n"
        if or_block.get("or_high") is not None else
        "- Opening range: _(insufficient intraday history this session)_\n"
    )

    pv_line = (
        f"- Pivots (prev-day): "
        f"P `{pv['P']:.2f}` / R1 `{pv['R1']:.2f}` / R2 `{pv['R2']:.2f}` / "
        f"S1 `{pv['S1']:.2f}` / S2 `{pv['S2']:.2f}`\n"
        if pv.get("P") is not None else
        "- Pivots: _(need >=2 days of bars)_\n"
    )
    cam_line = (
        f"- Camarilla R/S levels: "
        f"R3 `{pv['C_R3']:.2f}` / R2 `{pv['C_R2']:.2f}` / R1 `{pv['C_R1']:.2f}` / "
        f"S1 `{pv['C_S1']:.2f}` / S2 `{pv['C_S2']:.2f}` / S3 `{pv['C_S3']:.2f}`\n"
        if pv.get("C_R1") is not None else ""
    )
    wpv_line = (
        f"- Weekly pivots: P `{wpv['P']:.2f}` / R1 `{wpv['R1']:.2f}` / "
        f"S1 `{wpv['S1']:.2f}`\n"
        if wpv.get("P") is not None else ""
    )
    mpv_line = (
        f"- Monthly pivots: P `{mpv['P']:.2f}` / R1 `{mpv['R1']:.2f}` / "
        f"S1 `{mpv['S1']:.2f}`\n"
        if mpv.get("P") is not None else ""
    )

    avwap_lines: List[str] = []
    if avwap_hi is not None and ind.get("last_swing_high") is not None:
        avwap_lines.append(
            f"- Anchored VWAP from last swing-high "
            f"(`{ind['last_swing_high']:.2f}`): `{avwap_hi:.2f}`"
        )
    if avwap_lo is not None and ind.get("last_swing_low") is not None:
        avwap_lines.append(
            f"- Anchored VWAP from last swing-low "
            f"(`{ind['last_swing_low']:.2f}`): `{avwap_lo:.2f}`"
        )
    avwap_block = ("\n".join(avwap_lines) + "\n") if avwap_lines else ""

    return (
        f"### Indicator snapshot\n"
        f"- Last close: `{last_close:.2f}`\n"
        f"- Trend: {trend}  |  EMA20 `{ema20:.2f}` · EMA50 `{ema50:.2f}` · EMA200 `{ema200:.2f}`\n"
        f"- RSI(14): `{rsi_v:.1f}` ({rsi_tag})\n"
        f"- Stochastic(14,3,3): {_stoch_tag(sk, sd)}\n"
        f"- MACD: `{macd_v:.2f}` / signal `{sig_v:.2f}` / hist `{hist_v:.2f}` — {macd_tag}\n"
        f"- ATR(14): `{atr_v:.2f}` — implied stop ~`{atr_v * 1.5:.2f}` (1.5×ATR)\n"
        f"- Bollinger(20,2): mid `{bb_row.get('mid', float('nan')):.2f}` "
        f"/ upper `{bb_row.get('upper', float('nan')):.2f}` "
        f"/ lower `{bb_row.get('lower', float('nan')):.2f}` "
        f"/ bandwidth {bb_bw_str} — {_bb_tag(last_close, bb_row)}\n"
        f"- Session VWAP (anchored 22:00 UTC NY-close): `{vwap_v:.2f}` — price is **{vwap_rel}** VWAP\n"
        f"{avwap_block}"
        f"{or_line}"
        f"{pv_line}"
        f"{cam_line}"
        f"{wpv_line}"
        f"{mpv_line}"
    )
