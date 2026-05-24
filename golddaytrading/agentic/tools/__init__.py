"""Tập hợp các tool function dùng trong LangGraph workflow.

Tool được phân nhóm theo agent sẽ gọi:

* ``market_tools``  — Technical Agent: live price, OHLCV, indicators.
* ``news_tools``    — Macro & News Agent: macro pulse, calendar, RSS.
* ``memory_tools``  — Macro & News Agent: Graph RAG retrieval.

Mỗi tool là một plain function được decorate ``@tool`` từ
``langchain_core.tools``, nhận arg primitive (str/int/float) và trả
về **string markdown** (không trả pandas DataFrame qua biên giới
tool — vừa khó serialize, vừa tốn token).

Các node sẽ import ``TECH_TOOLS`` / ``MACRO_TOOLS`` để bind LLM.
"""

from golddaytrading.agentic.tools.market_tools import (
    get_live_price,
    get_intraday_ohlcv,
    get_higher_timeframe_trend,
    get_active_session,
    get_level_pool,
    TECH_TOOLS,
)
from golddaytrading.agentic.tools.news_tools import (
    get_macro_pulse,
    get_econ_calendar,
    get_gold_news,
    NEWS_TOOLS,
)
from golddaytrading.agentic.tools.memory_tools import (
    query_historical_context,
    MEMORY_TOOLS,
    get_graph_rag,
    set_graph_rag,
)

#: Tool dành cho Technical Agent.
TECHNICAL_AGENT_TOOLS = list(TECH_TOOLS)

#: Tool dành cho Macro & News Agent (gồm cả memory).
MACRO_NEWS_AGENT_TOOLS = list(NEWS_TOOLS) + list(MEMORY_TOOLS)

__all__ = [
    "TECHNICAL_AGENT_TOOLS",
    "MACRO_NEWS_AGENT_TOOLS",
    "get_live_price",
    "get_intraday_ohlcv",
    "get_higher_timeframe_trend",
    "get_active_session",
    "get_level_pool",
    "get_macro_pulse",
    "get_econ_calendar",
    "get_gold_news",
    "query_historical_context",
    "get_graph_rag",
    "set_graph_rag",
]
