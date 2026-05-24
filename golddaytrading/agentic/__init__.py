"""Kiến trúc Agentic Workflow cho GoldDayTrading.

Module này refactor pipeline tuyến tính cũ (``graph/pipeline.py``)
thành state machine dạng đồ thị có chu trình (cyclic graph) sử
dụng **LangGraph**, kèm bộ nhớ lịch sử dạng **Graph RAG** (NetworkX
+ ChromaDB).

Triết lý thiết kế
-----------------

* **API-only:** mọi LLM call và mọi vector embedding đều đi qua API
  (mặc định: OpenAI). Máy local chỉ làm orchestrator CPU, không tải
  bất kỳ model HuggingFace nào.
* **Cùng tồn tại:** module ``agentic/`` không thay thế pipeline cũ.
  Người dùng vẫn có thể chạy ``gdt analyze`` (linear pipeline) như
  trước. Workflow mới mở qua ``gdt agentic-run``.
* **Có thể kiểm toán:** state là Pydantic schema → mọi snapshot
  JSON-serializable, dễ replay và debug.
* **Đồ thị có chu trình:** Risk Manager có thể *quay ngược* về
  Technical hoặc Macro để hỏi lại khi phát hiện mâu thuẫn — điều
  pipeline tuyến tính cũ không làm được.
* **Dynamic Tool Calling:** mỗi agent được bind tool *cụ thể của
  nó* (Technical: market tools; Macro: news + Graph RAG); LLM tự
  quyết định khi nào gọi tool nào (pattern ToolNode + bound LLM).

Cấu trúc module
---------------

* ``state.py``      — Pydantic ``AgentState`` + sub-schema.
* ``graph_rag.py``  — Knowledge Graph (NetworkX) + Vector RAG (Chroma).
* ``llm.py``        — Builder ``ChatOpenAI`` API-only.
* ``tools/``        — Tool function ``@tool`` cho từng agent.
* ``nodes/``        — Các node của LangGraph.
* ``graph.py``      — Lắp ráp StateGraph + conditional edges.
* ``runner.py``     — Entry-point ``run_agentic_workflow``.

Cách dùng nhanh
---------------

::

    from golddaytrading.agentic import run_agentic_workflow

    state = run_agentic_workflow("XAUUSD=X", max_iterations=3)
    print(state.final_decision.bias, state.final_decision.confidence)
"""

from golddaytrading.agentic.state import (  # noqa: F401
    AgentState,
    AgentOutput,
    FinalDecision,
    DebateMessage,
    MarketDataSnapshot,
    MacroEvent,
)
from golddaytrading.agentic.graph_rag import (  # noqa: F401
    GraphRAG,
    HistoricalEpisode,
    CorrelationEdge,
    RAGQueryResult,
    build_episode_from_macro_pulse,
)

# Lazy re-export — chỉ ai cài extra `agentic` mới import được.
# Người dùng pipeline cũ import được state/graph_rag mà không phải
# cài langgraph.

def __getattr__(name: str):  # PEP 562
    if name in ("run_agentic_workflow", "render_final_report"):
        from golddaytrading.agentic.runner import (
            run_agentic_workflow, render_final_report,
        )
        return {
            "run_agentic_workflow": run_agentic_workflow,
            "render_final_report": render_final_report,
        }[name]
    if name in ("build_graph", "get_default_graph"):
        from golddaytrading.agentic.graph import (
            build_graph, get_default_graph,
        )
        return {
            "build_graph": build_graph,
            "get_default_graph": get_default_graph,
        }[name]
    raise AttributeError(f"module 'golddaytrading.agentic' has no attribute {name!r}")


__all__ = [
    "AgentState", "AgentOutput", "FinalDecision",
    "DebateMessage", "MarketDataSnapshot", "MacroEvent",
    "GraphRAG", "HistoricalEpisode", "CorrelationEdge",
    "RAGQueryResult", "build_episode_from_macro_pulse",
    "run_agentic_workflow", "render_final_report",
    "build_graph", "get_default_graph",
]
