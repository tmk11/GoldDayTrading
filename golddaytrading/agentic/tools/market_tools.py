"""Tool function dành cho Technical Agent.

Mỗi tool wrap một fetcher / tính toán đã có sẵn trong
``golddaytrading.dataflows`` hoặc ``golddaytrading.signals`` và
**chỉ trả về string markdown** đã render xong. LLM đọc thẳng được,
đỡ phải decode JSON trung gian.

Lý do thiết kế:

* Tool nhận arg primitive (``str``, ``int``) → JSON-friendly với
  function-calling của OpenAI.
* Output luôn là ``str`` → ToolMessage của LangGraph đỡ phiền với
  custom serializer.
* Mọi tool đều **idempotent / read-only** — không có side effect
  ngoài cache HTTP của yfinance.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import tool

from golddaytrading.dataflows.indicators import (
    compute_indicators,
    indicator_summary_block,
)
from golddaytrading.dataflows.intraday_data import (
    fetch_intraday_ohlcv,
    latest_price_block,
    ohlcv_summary_block,
)
from golddaytrading.sessions import classify_session, session_summary_block
from golddaytrading.signals.levels import build_level_pool, level_pool_block

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Live price
# ---------------------------------------------------------------------------


@tool
def get_live_price(ticker: str = "XAUUSD=X") -> str:
    """Lấy giá close mới nhất + bar cuối cùng cho `ticker`.

    Trả về một dòng markdown có timestamp UTC. Dùng khi cần xác nhận
    giá hiện tại trước khi đặt entry/stop. Không lấy nhiều history
    — đó là việc của ``get_intraday_ohlcv``.
    """
    df = fetch_intraday_ohlcv(ticker, timeframe="5m", bars=2)
    if df is None or df.empty:
        return f"_(không lấy được giá cho {ticker})_"
    return latest_price_block(ticker, df)


# ---------------------------------------------------------------------------
# OHLCV + indicators
# ---------------------------------------------------------------------------


@tool
def get_intraday_ohlcv(
    ticker: str = "XAUUSD=X",
    timeframe: str = "15m",
    bars: int = 200,
    tail_rows: int = 8,
) -> str:
    """Lấy OHLCV intraday + indicator snapshot.

    Trả về 2 block markdown nối tiếp:

    1. Bảng nến gần nhất (``tail_rows`` dòng).
    2. Indicator snapshot: EMA stack, RSI, MACD, ATR, Bollinger,
       VWAP, opening range, daily / weekly / monthly pivots,
       Camarilla, anchored VWAP từ swing high/low.

    Tham số:
        ticker:    symbol yfinance (XAUUSD=X / GC=F / GLD…).
        timeframe: 1m | 5m | 15m | 30m | 1h | 4h | 1d.
        bars:      số nến cần fetch (mặc định 200 — đủ cho EMA200).
        tail_rows: số nến hiển thị trong bảng tóm tắt (mặc định 8).
    """
    df = fetch_intraday_ohlcv(ticker, timeframe=timeframe, bars=bars)
    if df is None or df.empty:
        return f"_(không lấy được OHLCV cho {ticker} @ {timeframe})_"

    ind = compute_indicators(df)
    parts = [
        ohlcv_summary_block(ticker, timeframe, df, tail_rows=tail_rows),
        indicator_summary_block(df, ind),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Higher timeframe trend tag
# ---------------------------------------------------------------------------


@tool
def get_higher_timeframe_trend(
    ticker: str = "XAUUSD=X",
    higher_timeframe: str = "1h",
    bars: int = 120,
) -> str:
    """Tag trend khung cao hơn (``up`` / ``down`` / ``chop``).

    Dùng khung 1h hoặc 4h để xác nhận trend macro trước khi vào lệnh
    ở 5m/15m. Trả về JSON-like string + lý do ngắn:

        ``{ "tag": "up", "ema20": 2350.1, "ema50": 2342.7, "ema200": 2310.4 }``

    Quy tắc tag (deterministic, không LLM):

    * ``up``   nếu EMA20 > EMA50 > EMA200 trên khung cao.
    * ``down`` nếu EMA20 < EMA50 < EMA200.
    * ``chop`` các trường hợp còn lại.
    """
    df = fetch_intraday_ohlcv(ticker, timeframe=higher_timeframe, bars=bars)
    if df is None or df.empty:
        return '{"tag":"unknown","reason":"no_data"}'
    ind = compute_indicators(df)
    e20 = float(ind["ema20"].iloc[-1])
    e50 = float(ind["ema50"].iloc[-1])
    e200 = float(ind["ema200"].iloc[-1])
    if e20 > e50 > e200:
        tag = "up"
    elif e20 < e50 < e200:
        tag = "down"
    else:
        tag = "chop"
    return (
        f'{{"tag":"{tag}","ema20":{e20:.2f},"ema50":{e50:.2f},'
        f'"ema200":{e200:.2f},"timeframe":"{higher_timeframe}"}}'
    )


# ---------------------------------------------------------------------------
# Active session
# ---------------------------------------------------------------------------


@tool
def get_active_session() -> str:
    """Trả về session FX/gold đang active (UTC).

    Dùng để Technical Agent biết liệu thị trường đang trong
    LONDON_NY_OVERLAP (high liquidity, ưu tiên breakout) hay TOKYO
    (thin, ưu tiên mean-reversion).
    """
    return session_summary_block()


# ---------------------------------------------------------------------------
# Deterministic level pool
# ---------------------------------------------------------------------------


@tool
def get_level_pool(
    ticker: str = "XAUUSD=X",
    timeframe: str = "15m",
    bars: int = 200,
    min_rr: float = 1.5,
    htf_trend: Optional[str] = None,
) -> str:
    """Sinh **level pool deterministic** (entry/stop/tp) từ indicator.

    Trả về bảng markdown các trade idea ứng cử (VWAP reclaim,
    opening-range breakout, pivot bounce, Bollinger mean-reversion…)
    đã được lọc theo ``min_rr`` và xếp hạng theo score.

    **Quan trọng:** các giá trong bảng này là *deterministic* — anchor
    vào market structure thật. Risk Manager phải **chọn 1 row** theo
    ``setup_id`` thay vì bịa giá riêng.

    :param htf_trend:  ``"up"`` / ``"down"`` / ``"chop"`` để bias
                       ranking; nếu None thì tự lấy từ khung 1h.
    """
    df = fetch_intraday_ohlcv(ticker, timeframe=timeframe, bars=bars)
    if df is None or df.empty:
        return f"_(không sinh được level pool cho {ticker})_"
    ind = compute_indicators(df)

    if htf_trend is None:
        try:
            df_htf = fetch_intraday_ohlcv(ticker, timeframe="1h", bars=120)
            if df_htf is not None and not df_htf.empty:
                ind_htf = compute_indicators(df_htf)
                e20 = float(ind_htf["ema20"].iloc[-1])
                e50 = float(ind_htf["ema50"].iloc[-1])
                e200 = float(ind_htf["ema200"].iloc[-1])
                if e20 > e50 > e200:
                    htf_trend = "up"
                elif e20 < e50 < e200:
                    htf_trend = "down"
                else:
                    htf_trend = "chop"
        except Exception as exc:  # pragma: no cover
            logger.warning("HTF trend infer failed: %s", exc)

    pool = build_level_pool(df, ind, min_rr=min_rr, htf_trend=htf_trend)
    return level_pool_block(pool, ticker=ticker)


# ---------------------------------------------------------------------------
# Bundle export
# ---------------------------------------------------------------------------


TECH_TOOLS = [
    get_live_price,
    get_intraday_ohlcv,
    get_higher_timeframe_trend,
    get_active_session,
    get_level_pool,
]
