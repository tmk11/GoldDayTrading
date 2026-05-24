"""Tool function dành cho Macro & News Agent.

Wrap các fetcher trong ``golddaytrading.dataflows`` thành tool có
chữ ký primitive và output markdown. Macro & News Agent sẽ:

1. ``get_macro_pulse()`` để xem DXY/US10Y/VIX/EURUSD/realyield…
2. ``get_econ_calendar(hours_ahead)`` để biết blackout window 5'
   trước / 15' sau mỗi sự kiện high-impact.
3. ``get_gold_news(hours_back)`` để lấy headline RSS gần nhất.

Tool ``query_historical_context`` (Graph RAG) được tách sang
``memory_tools.py`` cho rõ ràng — vẫn được Macro Agent bind.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from golddaytrading.dataflows.econ_calendar import (
    calendar_block,
    fetch_upcoming_events,
)
from golddaytrading.dataflows.gold_news import (
    fetch_recent_gold_news,
    gold_news_block,
)
from golddaytrading.dataflows.macro_pulse import (
    fetch_macro_pulse,
    macro_pulse_block,
)

logger = logging.getLogger(__name__)


@tool
def get_macro_pulse() -> str:
    """Snapshot vĩ mô intraday (1h cadence).

    Trả về bảng markdown gồm: DXY, EURUSD, US 5/10/30Y yields, TIPS,
    VIX, ES, WTI crude, BTC, Silver — mỗi dòng có Δ 1h / 4h / 1d
    cộng với gold-bias tương ứng. Cuối block là **regime tag** đã
    được phân loại deterministic (REAL_YIELD_DRIVE / USD_WEAKNESS /
    RISK_OFF_HAVEN_BID / GROWTH_SCARE / RISK_ON / RANGE_BOUND).

    Dùng để Macro Agent kết luận xem driver vĩ mô đang nghiêng về
    LONG/SHORT cho gold ở khung intraday.
    """
    pulse = fetch_macro_pulse()
    return macro_pulse_block(pulse)


@tool
def get_econ_calendar(hours_ahead: int = 24) -> str:
    """Lịch kinh tế USD/EUR high-impact ``hours_ahead`` giờ tới.

    Output là bảng markdown sắp xếp theo thời gian, có cột "In"
    đếm ngược tới sự kiện. **Risk Manager bắt buộc enforce blackout**
    5' trước và 15' sau mỗi sự kiện ``Impact = High``.

    Tham số:
        hours_ahead:  cửa sổ nhìn về phía trước (mặc định 24).
                      Đặt 6-8 cho intraday gần; 48-72 nếu cần
                      planning trước cuối tuần.
    """
    events = fetch_upcoming_events(hours_ahead=hours_ahead)
    return calendar_block(events)


@tool
def get_gold_news(hours_back: int = 12, per_feed_limit: int = 5) -> str:
    """Headline gold-relevant từ các RSS công khai trong ``hours_back`` giờ.

    Nguồn: Investing.com (Commodities, Economy), Mining.com (gold
    tag), Bloomberg Markets. Không cần API key.

    Tham số:
        hours_back:     bao nhiêu giờ về trước (mặc định 12).
        per_feed_limit: tối đa headline mỗi feed (mặc định 5).
    """
    news = fetch_recent_gold_news(
        hours_back=hours_back, per_feed_limit=per_feed_limit
    )
    return gold_news_block(news, hours_back=hours_back)


NEWS_TOOLS = [
    get_macro_pulse,
    get_econ_calendar,
    get_gold_news,
]
