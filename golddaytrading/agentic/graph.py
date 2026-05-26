"""Lắp ráp LangGraph StateGraph cho workflow agentic.

Topology
--------

::

    START
      |
      v
    data_gatherer  (pure Python, không LLM)
      |
      v
    technical_agent  <----------+
      |                         |
      v                         |  (route_back)
    macro_news_agent  <-------+ |
      |                       | |
      v                       | |
    risk_manager  ------------+-+
      |
      v  (action=finalize)
    memory_consolidator   ← Self-Learning hook (Graph RAG ingest)
      |
      v
    END

Các edge tĩnh:
* START → data_gatherer
* data_gatherer → technical_agent
* technical_agent → macro_news_agent
* macro_news_agent → risk_manager
* memory_consolidator → END

Edge **conditional** sau Risk Manager — quyết định bằng
:func:`risk_router` đọc ``state.next_node``:

* ``"END"``                → memory_consolidator → END
* ``"technical_agent"``    → technical_agent (vòng phản biện mới).
* ``"macro_news_agent"``   → macro_news_agent.

Đây là pattern **(A) Explicit Node Selection** — Risk Manager LLM
*chỉ định tường minh* node nào quay lại, không heuristic Python.
``memory_consolidator`` chen vào *trước* END để mọi run đều ingest
1 episode vào Graph RAG (Self-Learning loop).
"""

from __future__ import annotations

import logging
from typing import Optional

from golddaytrading.agentic.nodes import (
    bear_case_agent_node,
    bull_case_agent_node,
    data_gatherer_node,
    macro_news_agent_node,
    memory_consolidator_node,
    risk_manager_node,
    technical_agent_node,
)
from golddaytrading.agentic.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional router
# ---------------------------------------------------------------------------

def debate_router(state: AgentState) -> str:
    """Decide whether the lightweight bull/bear debate should run."""
    tech = state.agent_outputs.get("technical")
    macro = state.agent_outputs.get("macro_news")
    reasons = []

    confidences = [out.confidence for out in (tech, macro) if out is not None]
    if any(0.50 <= confidence <= 0.75 for confidence in confidences):
        reasons.append("signal confidence is between 50 and 75")

    if tech and macro and tech.bias != macro.bias and "NEUTRAL" not in (tech.bias, macro.bias):
        reasons.append("technical and macro agents disagree")

    if _price_near_key_level(state):
        reasons.append("price is near key support/resistance")

    if any(event.impact == "High" and 0 <= event.minutes_until() <= 240 for event in state.macro_events):
        reasons.append("high-impact news is detected")

    if reasons:
        state.debate_required = True
        state.debate_reason = "; ".join(reasons)
        logger.info("Bull/bear debate enabled: %s", state.debate_reason)
        return "bull_case_agent"
    state.debate_required = False
    state.debate_reason = ""
    return "risk_manager"

def _price_near_key_level(state: AgentState) -> bool:
    md = state.market_data
    tech = state.agent_outputs.get("technical")
    if not md or md.last_price is None or not tech or not tech.key_levels:
        return False
    atr = (md.indicators or {}).get("atr14") or 0.0
    tolerance = max(float(md.last_price) * 0.0015, float(atr or 0.0) * 0.5, 1.0)
    for key, level in tech.key_levels.items():
        if any(token in key.lower() for token in ("entry", "stop", "tp", "support", "resistance", "level")):
            if abs(float(level) - float(md.last_price)) <= tolerance:
                return True
    return False


def risk_router(state: AgentState) -> str:
    """Đọc `state.next_node` (do Risk Manager đặt) → key edge.

    Trả về một trong các literal:

    * ``"memory_consolidator"`` — finalize (final_decision đã có
      hoặc đã hết iteration). Sau memory_consolidator sẽ là END.
    * ``"technical_agent"``    — vòng phản biện mới với Technical.
    * ``"macro_news_agent"``   — vòng phản biện mới với Macro/News.

    Có 3 hard guard để tránh infinite loop:

    1. Nếu `final_decision` đã được set → finalize.
    2. Nếu `iteration >= max_iterations` → finalize.
    3. Nếu `next_node` không hợp lệ → finalize.
    """
    if state.final_decision is not None:
        return "memory_consolidator"
    if state.iteration >= state.max_iterations:
        logger.warning(
            "Đã chạm max_iterations=%d, ép finalize qua memory_consolidator.",
            state.max_iterations,
        )
        return "memory_consolidator"
    nxt = state.next_node
    if nxt in ("technical_agent", "macro_news_agent"):
        return nxt
    return "memory_consolidator"


# ---------------------------------------------------------------------------
# Build & compile
# ---------------------------------------------------------------------------


def build_graph(checkpointer: Optional[object] = None):
    """Build và compile LangGraph StateGraph.

    :param checkpointer:  Optional checkpointer (ví dụ
                          ``MemorySaver``) để hỗ trợ pause/resume.
                          Mặc định None — workflow chạy stateless,
                          state nằm hoàn toàn trong process.
    :returns:             ``CompiledStateGraph`` sẵn sàng `invoke`.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Cần cài extra `agentic`: pip install -e \".[agentic]\""
        ) from exc

    graph = StateGraph(AgentState)

    # 1. Đăng ký node
    graph.add_node("data_gatherer", data_gatherer_node)
    graph.add_node("technical_agent", technical_agent_node)
    graph.add_node("macro_news_agent", macro_news_agent_node)
    graph.add_node("bull_case_agent", bull_case_agent_node)
    graph.add_node("bear_case_agent", bear_case_agent_node)
    graph.add_node("risk_manager", risk_manager_node)
    graph.add_node("memory_consolidator", memory_consolidator_node)

    # 2. Edge tĩnh
    graph.add_edge(START, "data_gatherer")
    graph.add_edge("data_gatherer", "technical_agent")
    graph.add_edge("technical_agent", "macro_news_agent")
    graph.add_conditional_edges(
        "macro_news_agent",
        debate_router,
        {
            "bull_case_agent": "bull_case_agent",
            "risk_manager": "risk_manager",
        },
    )
    graph.add_edge("bull_case_agent", "bear_case_agent")
    graph.add_edge("bear_case_agent", "risk_manager")
    graph.add_edge("memory_consolidator", END)

    # 3. Conditional edge sau Risk Manager
    graph.add_conditional_edges(
        "risk_manager",
        risk_router,
        {
            "memory_consolidator": "memory_consolidator",
            "technical_agent": "technical_agent",
            "macro_news_agent": "macro_news_agent",
        },
    )

    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


_COMPILED = None


def get_default_graph():
    """Lấy graph mặc định (cached). Dùng cho runner / test nhanh."""
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()
    return _COMPILED
