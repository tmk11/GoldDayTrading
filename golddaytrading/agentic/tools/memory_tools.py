"""Graph RAG tool — bộ nhớ lịch sử cho Macro & News Agent.

Tool ``query_historical_context`` cho phép agent hỏi:

    "Gold phản ứng thế nào 3 lần gần nhất khi DXY và US10Y phân kỳ
     (DXY xuống, US10Y lên)?"

Bên dưới, tool gọi ``GraphRAG.query_historical_context`` (xem
``agentic/graph_rag.py``) → top-k episode tương tự về ngữ nghĩa từ
ChromaDB + cạnh KG liên quan từ NetworkX → render thành block
markdown sẵn cho prompt.

Singleton instance
------------------

GraphRAG là expensive (mở Chroma persistent client + load KG pickle
+ giữ embedder OpenAI). Ta dùng **module-level singleton** với lazy
init: lần đầu tool được gọi mới khởi tạo. Runner có thể inject một
instance custom qua :func:`set_graph_rag` (vd: cho test hoặc dùng
persist_dir riêng).
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from langchain_core.tools import tool

from golddaytrading.agentic.graph_rag import GraphRAG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------


_RAG_LOCK = threading.RLock()
_RAG_INSTANCE: Optional[GraphRAG] = None


def get_graph_rag() -> GraphRAG:
    """Trả về singleton :class:`GraphRAG`, lazy init lần gọi đầu."""
    global _RAG_INSTANCE
    with _RAG_LOCK:
        if _RAG_INSTANCE is None:
            _RAG_INSTANCE = GraphRAG.default()
            logger.info(
                "GraphRAG đã init: %s", _RAG_INSTANCE.stats()
            )
        return _RAG_INSTANCE


def set_graph_rag(instance: Optional[GraphRAG]) -> None:
    """Override singleton (cho test hoặc cấu hình runner).

    Truyền ``None`` để reset về lazy-init.
    """
    global _RAG_INSTANCE
    with _RAG_LOCK:
        _RAG_INSTANCE = instance


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool
def query_historical_context(
    query: str,
    k: int = 3,
    regime_filter: Optional[str] = None,
) -> str:
    """Truy vấn bộ nhớ lịch sử (Graph RAG) cho các tình huống tương tự.

    Trả về top-``k`` episode quá khứ giống nhất về ngữ nghĩa, kèm
    cạnh Knowledge Graph liên quan tới các thực thể trong những
    episode đó. Output là markdown sẵn cho prompt — đã tóm tắt
    timestamp, regime, narrative và quan hệ KG (DXY⊥GOLD…).

    Tham số:
        query:          câu hỏi tự nhiên. Ví dụ:
                        "Gold phản ứng ra sao khi CPI thấp hơn dự
                        báo và DXY giảm?"
        k:              số episode trả về (mặc định 3).
        regime_filter:  giới hạn trong một regime cụ thể, ví dụ
                        ``"USD_WEAKNESS"`` hoặc
                        ``"REAL_YIELD_DRIVE"``. Để None để duyệt
                        tất cả.

    Quan trọng: Graph RAG là **prior**, KHÔNG phải dự báo. Nếu setup
    hiện tại không khớp episode nào, hãy nói rõ trong rationale.
    """
    rag = get_graph_rag()
    result = rag.query_historical_context(
        query=query, k=max(1, int(k)), regime_filter=regime_filter
    )
    return result.rendered_block


MEMORY_TOOLS = [query_historical_context]
