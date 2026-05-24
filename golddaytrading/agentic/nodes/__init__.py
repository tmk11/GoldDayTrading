"""Các node của LangGraph workflow.

Mỗi module export một hàm ``<name>_node(state) -> dict`` mà
``StateGraph.add_node`` consume trực tiếp. Hàm trả về một **dict
partial** (theo convention của LangGraph) — chỉ chứa field thực sự
muốn cập nhật; các field còn lại giữ nguyên hoặc được merge qua
reducer định nghĩa trong ``state.py``.
"""

from golddaytrading.agentic.nodes.data_gatherer import data_gatherer_node
from golddaytrading.agentic.nodes.macro_news_agent import macro_news_agent_node
from golddaytrading.agentic.nodes.memory_consolidator import memory_consolidator_node
from golddaytrading.agentic.nodes.risk_manager import risk_manager_node
from golddaytrading.agentic.nodes.technical_agent import technical_agent_node

__all__ = [
    "data_gatherer_node",
    "technical_agent_node",
    "macro_news_agent_node",
    "risk_manager_node",
    "memory_consolidator_node",
]
