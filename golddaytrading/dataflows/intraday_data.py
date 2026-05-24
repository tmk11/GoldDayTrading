"""Intraday OHLCV fetching for the gold complex via yfinance.

Day-traders need 1m/5m/15m/1h candles, not the daily candles the
upstream TradingAgents framework was built for. yfinance limits the
intraday history per timeframe (e.g. 7 days for 1m, 60 days for
5m/15m, 730 days for 1h) so we map each requested timeframe to a
sensible ``period`` window automatically.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# yfinance ``interval`` -> default ``period`` to request. The period
# is chosen so we get at least ~200 bars of history for indicators.
_TF_TO_PERIOD = {
    "1m":  "5d",
    "2m":  "5d",
    "5m":  "30d",
    "15m": "60d",
    "30m": "60d",
    "60m": "180d",
    "1h":  "180d",
    "4h":  "730d",
    "1d":  "2y",
}

# Aliases the user might pass.
_TF_NORMALISE = {
    "1h": "60m",
    "4h": "1h",        # yfinance has no 4h; we resample below
    "1H": "60m",
    "60M": "60m",
}


def _normalise_tf(tf: str) -> str:
    return _TF_NORMALISE.get(tf, tf)


def fetch_intraday_ohlcv(
    ticker: str,
    timeframe: str = "15m",
    bars: int = 200,
) -> Optional[pd.DataFrame]:
    """Fetch the most recent ``bars`` of OHLCV at ``timeframe``.

    Returns a DataFrame indexed by timezone-aware UTC timestamps with
    columns ``[Open, High, Low, Close, Volume]``. Returns ``None``
    when yfinance is unreachable or the response is empty.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        logger.warning("yfinance not installed: %s", exc)
        return None

    requested_tf = timeframe
    interval = _normalise_tf(timeframe)

    # yfinance does not support 4h directly; pull 1h and resample.
    resample_to_4h = requested_tf in ("4h", "4H")
    fetch_interval = "60m" if resample_to_4h else interval

    period = _TF_TO_PERIOD.get(fetch_interval, "60d")

    try:
        df = yf.download(
            ticker,
            period=period,
            interval=fetch_interval,
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=10,
            multi_level_index=False,
        )
    except Exception as exc:
        logger.warning("yfinance history failed for %s @ %s: %s",
                       ticker, fetch_interval, exc)
        return None

    if df is None or df.empty:
        return None

    # Standardise column casing.
    df = df.rename(columns=str.title)
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[keep].copy()

    # yfinance sometimes returns naive index; force UTC.
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    if resample_to_4h:
        df = (
            df.resample("4h", label="right", closed="right")
              .agg({"Open": "first", "High": "max", "Low": "min",
                    "Close": "last", "Volume": "sum"})
              .dropna(subset=["Open", "Close"])
        )

    return df.tail(bars)


def latest_price_block(ticker: str, df: pd.DataFrame | None) -> str:
    """Render a one-line current-price summary."""
    if df is None or df.empty:
        return f"_(price feed unavailable for {ticker})_\n"
    last = df.iloc[-1]
    ts = df.index[-1].strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"**{ticker}** last close: `{last['Close']:.2f}` "
        f"(O `{last['Open']:.2f}` / H `{last['High']:.2f}` / "
        f"L `{last['Low']:.2f}`)  — {ts}\n"
    )


def ohlcv_summary_block(ticker: str, timeframe: str, df: pd.DataFrame | None,
                        tail_rows: int = 8) -> str:
    """Render the last few candles as a compact markdown table.

    The Technical Analyst gets this *plus* the indicator block, so
    the LLM can sanity-check claims against raw bars without us
    spending tokens on the whole 200-bar history.
    """
    if df is None or df.empty:
        return f"### Recent {timeframe} bars for {ticker}\n_(no data)_\n"

    rows = df.tail(tail_rows)
    header = "| Time (UTC) | Open | High | Low | Close | Volume |\n"
    sep = "|---|---|---|---|---|---|\n"
    body = "".join(
        f"| {ts.strftime('%m-%d %H:%M')} "
        f"| {r['Open']:.2f} | {r['High']:.2f} | "
        f"{r['Low']:.2f} | {r['Close']:.2f} | "
        f"{int(r['Volume']) if 'Volume' in r and pd.notna(r['Volume']) else 0} |\n"
        for ts, r in rows.iterrows()
    )
    return (
        f"### Recent {timeframe} bars for {ticker} (last {len(rows)})\n"
        f"{header}{sep}{body}\n"
    )
