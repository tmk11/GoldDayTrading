"""Data Gatherer Node — pure Python, không gọi LLM.

Chức năng: nạp một lần các dữ liệu nền tảng vào ``AgentState`` để
các node LLM phía sau không phải tự fetch lại từ tool — vừa tiết
kiệm token, vừa đảm bảo mọi agent đều nhìn cùng một snapshot vĩ mô.

Cụ thể, node này:

1. Pull OHLCV primary + higher timeframe qua yfinance.
2. Tính indicator snapshot.
3. Render markdown block: bảng OHLCV gần nhất, indicator snapshot
   khung primary và khung HTF.
4. Pull macro pulse và lịch kinh tế 24h tới.
5. Pull news RSS gold-relevant.
6. Phân loại session.
7. Đóng gói tất cả vào ``MarketDataSnapshot`` + các block markdown
   trên ``AgentState``.

Tool-calling của các agent về sau **vẫn còn ý nghĩa** — chúng có
thể gọi thêm để verify/ refresh dữ liệu khi Risk Manager yêu cầu
re-fetch. Nhưng baseline data đã sẵn sàng từ vòng đầu.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from golddaytrading.agentic.state import (
    AgentState,
    MacroEvent,
    MarketDataSnapshot,
)
from golddaytrading.dataflows.econ_calendar import (
    calendar_block,
    fetch_upcoming_events,
)
from golddaytrading.dataflows.gold_news import (
    fetch_recent_gold_news,
    gold_news_block,
)
from golddaytrading.dataflows.indicators import (
    compute_indicators,
    indicator_summary_block,
)
from golddaytrading.dataflows.intraday_data import (
    fetch_intraday_ohlcv,
    ohlcv_summary_block,
)
from golddaytrading.dataflows.macro_pulse import (
    fetch_macro_pulse,
    macro_pulse_block,
)
from golddaytrading.sessions import classify_session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _htf_trend_from_indicators(ind: Dict[str, Any]) -> str:
    if not ind:
        return "chop"
    try:
        e20 = float(ind["ema20"].iloc[-1])
        e50 = float(ind["ema50"].iloc[-1])
        e200 = float(ind["ema200"].iloc[-1])
    except (KeyError, IndexError, ValueError, TypeError):
        return "chop"
    if e20 > e50 > e200:
        return "up"
    if e20 < e50 < e200:
        return "down"
    return "chop"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def data_gatherer_node(state: AgentState) -> Dict[str, Any]:
    """Thu thập dữ liệu nền tảng cho cả run.

    Trả về dict update gồm:
    - ``market_data``: :class:`MarketDataSnapshot`
    - ``macro_events``: List[:class:`MacroEvent`]
    - ``macro_pulse_block`` / ``calendar_block`` / ``news_block``
    - ``debate_history``: 1 message ``role=system`` ghi log đã gather.
    """
    asset = state.asset
    primary_tf = "15m"          # tham số cứng v1; có thể tham số hoá sau
    higher_tf = "1h"
    bars_primary = 200
    bars_higher = 120

    errors: List[str] = []

    # -------- 1. OHLCV primary + indicators --------
    df_p = fetch_intraday_ohlcv(asset, timeframe=primary_tf, bars=bars_primary)
    if df_p is None or df_p.empty:
        errors.append(f"yfinance không trả OHLCV cho {asset}@{primary_tf}.")
        ind_p: Dict[str, Any] = {}
    else:
        ind_p = compute_indicators(df_p)

    # -------- 2. OHLCV higher timeframe --------
    df_h = fetch_intraday_ohlcv(asset, timeframe=higher_tf, bars=bars_higher)
    if df_h is None or df_h.empty:
        errors.append(f"yfinance không trả OHLCV cho {asset}@{higher_tf}.")
        ind_h: Dict[str, Any] = {}
    else:
        ind_h = compute_indicators(df_h)

    # -------- 3. Render block --------
    ohlcv_block = (
        ohlcv_summary_block(asset, primary_tf, df_p, tail_rows=8)
        if df_p is not None else f"_(no OHLCV @{primary_tf})_\n"
    )
    indicator_block = (
        indicator_summary_block(df_p, ind_p)
        if df_p is not None and ind_p else "_(indicators skipped)_\n"
    )
    higher_indicator_block = (
        indicator_summary_block(df_h, ind_h)
        if df_h is not None and ind_h else "_(higher tf indicators skipped)_\n"
    )

    # -------- 4. Snapshot scalar --------
    last_price = last_open = last_high = last_low = last_close = last_vol = None
    last_bar_time = None
    indicators_dict: Dict[str, Any] = {}
    if df_p is not None and not df_p.empty:
        last_row = df_p.iloc[-1]
        last_open = float(last_row["Open"])
        last_high = float(last_row["High"])
        last_low = float(last_row["Low"])
        last_close = float(last_row["Close"])
        last_price = last_close
        last_vol = float(last_row.get("Volume") or 0.0)
        ts = df_p.index[-1]
        # đảm bảo timezone-aware
        last_bar_time = (
            ts.to_pydatetime().astimezone(timezone.utc)
            if hasattr(ts, "to_pydatetime") else ts
        )
        if ind_p:
            try:
                indicators_dict = {
                    "ema20": float(ind_p["ema20"].iloc[-1]),
                    "ema50": float(ind_p["ema50"].iloc[-1]),
                    "ema200": float(ind_p["ema200"].iloc[-1]),
                    "rsi14": float(ind_p["rsi14"].iloc[-1]),
                    "atr14": float(ind_p["atr14"].iloc[-1]),
                    "vwap": float(ind_p["vwap"].iloc[-1]),
                    "macd_hist": float(ind_p["macd"]["hist"].iloc[-1]),
                }
            except Exception as exc:  # pragma: no cover
                logger.warning("Không trích được indicator scalar: %s", exc)

    htf_trend = _htf_trend_from_indicators(ind_h)
    session = classify_session()

    market_data = MarketDataSnapshot(
        primary_timeframe=primary_tf,
        higher_timeframe=higher_tf,
        last_price=last_price,
        last_open=last_open,
        last_high=last_high,
        last_low=last_low,
        last_close=last_close,
        last_volume=last_vol,
        last_bar_time_utc=last_bar_time,
        indicators=indicators_dict,
        ohlcv_block=ohlcv_block,
        indicator_block=indicator_block,
        higher_indicator_block=higher_indicator_block,
        active_session=session.name,
        htf_trend=htf_trend,  # type: ignore[arg-type]
    )

    # -------- 5. Macro pulse --------
    try:
        pulse = fetch_macro_pulse()
        macro_pulse_md = macro_pulse_block(pulse)
    except Exception as exc:  # pragma: no cover
        logger.warning("Macro pulse fetch lỗi: %s", exc)
        macro_pulse_md = "_(macro pulse unavailable)_\n"
        errors.append(f"macro pulse: {exc}")

    # -------- 6. Calendar --------
    macro_events: List[MacroEvent] = []
    try:
        events = fetch_upcoming_events(hours_ahead=24)
        calendar_md = calendar_block(events)
        for ev in events:
            try:
                macro_events.append(MacroEvent(
                    title=ev.title,
                    country=ev.country or "USD",
                    impact=ev.impact if ev.impact in ("High", "Medium", "Low") else "Low",
                    when_utc=ev.when_utc,
                    forecast=ev.forecast,
                    previous=ev.previous,
                ))
            except Exception:
                # bỏ qua event lỗi schema, không phá run
                continue
    except Exception as exc:  # pragma: no cover
        logger.warning("Calendar fetch lỗi: %s", exc)
        calendar_md = "_(calendar unavailable)_\n"
        errors.append(f"calendar: {exc}")

    # -------- 7. News RSS --------
    try:
        news = fetch_recent_gold_news(hours_back=12, per_feed_limit=5)
        news_md = gold_news_block(news, hours_back=12)
    except Exception as exc:  # pragma: no cover
        logger.warning("News fetch lỗi: %s", exc)
        news_md = "_(news unavailable)_\n"
        errors.append(f"news: {exc}")

    # -------- 8. Build state update --------
    from golddaytrading.agentic.state import DebateMessage  # tránh import vòng

    log_msg = DebateMessage(
        role="system",
        iteration=state.iteration,
        content=(
            f"Đã thu thập dữ liệu nền tảng: {asset} @ {primary_tf}, "
            f"HTF {higher_tf} trend={htf_trend}, session={session.name}, "
            f"events={len(macro_events)} high-impact, "
            f"errors={len(errors)}."
        ),
    )

    return {
        "market_data": market_data,
        "macro_events": macro_events,
        "macro_pulse_block": macro_pulse_md,
        "calendar_block": calendar_md,
        "news_block": news_md,
        "debate_history": [log_msg],
        "errors": errors,
    }
