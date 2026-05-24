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
    graph.add_node("risk_manager", risk_manager_node)
    graph.add_node("memory_consolidator", memory_consolidator_node)

    # 2. Edge tĩnh
    graph.add_edge(START, "data_gatherer")
    graph.add_edge("data_gatherer", "technical_agent")
    graph.add_edge("technical_agent", "macro_news_agent")
    graph.add_edge("macro_news_agent", "risk_manager")
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
