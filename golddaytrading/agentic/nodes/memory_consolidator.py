"""Memory Consolidation Node — biến mỗi run thành một "trading episode".

Vị trí trong workflow
---------------------

::

    risk_manager  --(finalize)-->  memory_consolidator  -->  END
                  --(route_back)-->  technical / macro_news

Tức là chỉ chạy SAU khi Risk Manager đã phát ``FinalDecision``.
Nếu Risk Manager route_back, conditional edge bypass node này.

Trách nhiệm
-----------

1. Đọc state đã hoàn chỉnh (asset, market_data, macro_regime,
   macro_scalars, agent_outputs, final_decision).
2. Build một :class:`HistoricalEpisode` với:
   * ``narrative`` ngắn gọn, đủ để retrieval ngữ nghĩa hoạt động.
   * ``entity_nodes`` = các thực thể KG liên quan ([GOLD, DXY,
     US10Y, VIX] + event_id của các sự kiện high-impact gần).
   * ``metadata`` = các scalar primitive (regime, bias, confidence,
     dxy_chg_1h, tnx_chg_1h, vix_chg_1h, …) — dùng cho filter.
3. Gọi ``GraphRAG.ingest_episode(...)``.
4. Ghi log vào ``debate_history``.

Robustness
----------

Toàn bộ logic bọc trong try/except: nếu Chroma timeout, OpenAI
embedding API lỗi, hoặc KG pickle write fail, node vẫn return state
update bình thường (chỉ thêm vào ``errors``). KHÔNG bao giờ kill
workflow ở giai đoạn này — mất ingest 1 episode còn nhẹ hơn mất cả
``FinalDecision`` đã sinh được.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from golddaytrading.agentic.state import AgentState, DebateMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Map tên macro key trong KG (xem `_seed_kg` trong graph_rag.py)
_KG_NODES_DEFAULT = ["GOLD", "DXY", "US10Y", "VIX"]


def _classify_event_kg_node(title: str) -> str:
    """Map title sự kiện high-impact sang node id KG đã seed.

    Sự kiện không khớp regex được bỏ qua (không tự sinh node mới ở
    đây — để tránh KG bloat khi ForexFactory đổi cách đặt tên).
    """
    t = title.upper()
    if "CPI" in t:
        return "CPI_release"
    if "NON-FARM" in t or "NONFARM" in t or "NFP" in t or "PAYROLLS" in t:
        return "NFP_release"
    if "FOMC" in t or "FED" in t and "RATE" in t:
        return "FOMC_release"
    return ""


def _build_narrative(state: AgentState) -> str:
    """Tạo narrative <= 60 từ, đủ để embedding bắt được ngữ cảnh."""
    fd = state.final_decision
    md = state.market_data
    s = state.macro_scalars or {}
    regime = state.macro_regime or "RANGE_BOUND"

    parts: List[str] = [f"[{regime}]"]

    # Macro deltas
    deltas: List[str] = []
    if "dxy_chg_1h" in s:
        deltas.append(f"DXY {s['dxy_chg_1h']:+.2f}%")
    if "tnx_chg_1h" in s:
        deltas.append(f"US10Y {s['tnx_chg_1h']:+.2f}%")
    if "vix_chg_1h" in s:
        deltas.append(f"VIX {s['vix_chg_1h']:+.2f}%")
    if deltas:
        parts.append("Driver vĩ mô 1h: " + ", ".join(deltas) + ".")

    # Snapshot kỹ thuật
    if md and md.last_price is not None:
        parts.append(
            f"Gold ~{md.last_price:.2f}, "
            f"session={md.active_session}, HTF={md.htf_trend}."
        )

    # Quyết định
    if fd is not None:
        if fd.bias == "NEUTRAL":
            parts.append(
                f"Quyết định: NEUTRAL (conf {fd.confidence:.2f})."
            )
        else:
            parts.append(
                f"Quyết định: {fd.bias} entry {fd.entry:.2f} "
                f"stop {fd.stop_loss:.2f}"
                + (f" tp {fd.take_profit_1:.2f}"
                   if fd.take_profit_1 is not None else "")
                + f" (conf {fd.confidence:.2f})."
            )

    # Mâu thuẫn đã giải quyết (tóm 1 câu nếu có)
    if fd is not None and fd.contradictions_resolved:
        parts.append(
            "Mâu thuẫn xử lý: "
            + "; ".join(fd.contradictions_resolved[:2])
        )

    return " ".join(parts).strip()


def _build_metadata(state: AgentState) -> Dict[str, Any]:
    """Metadata Chroma — chỉ primitive."""
    fd = state.final_decision
    md = state.market_data
    meta: Dict[str, Any] = {
        "regime": state.macro_regime or "RANGE_BOUND",
        "asset": state.asset,
    }

    # Macro scalars
    for k, v in (state.macro_scalars or {}).items():
        if isinstance(v, (int, float)):
            meta[k] = float(v)

    # Decision summary
    if fd is not None:
        meta["bias"] = fd.bias
        meta["confidence"] = float(fd.confidence)
        if fd.entry is not None:
            meta["entry"] = float(fd.entry)
        if fd.stop_loss is not None:
            meta["stop_loss"] = float(fd.stop_loss)
        if fd.rr_ratio is not None:
            meta["rr_ratio"] = float(fd.rr_ratio)

    # Market context
    if md is not None:
        if md.last_price is not None:
            meta["last_price"] = float(md.last_price)
        if md.htf_trend:
            meta["htf_trend"] = str(md.htf_trend)
        if md.active_session:
            meta["session"] = str(md.active_session)
        # Indicator scalar (RSI / ATR / VWAP) nếu có
        for ind_key in ("rsi14", "atr14", "vwap", "macd_hist"):
            v = md.indicators.get(ind_key)
            if isinstance(v, (int, float)):
                meta[ind_key] = float(v)

    # Iterations đã trải qua + có route_back hay không
    meta["iterations"] = int(state.iteration)
    meta["had_contradiction"] = bool(state.has_contradiction())
    return meta


def _build_entity_nodes(state: AgentState) -> List[str]:
    """KG node id liên quan tới episode này.

    Mặc định: [GOLD, DXY, US10Y, VIX]. Bổ sung event node cho các
    sự kiện high-impact trong < 4h.
    """
    nodes = list(_KG_NODES_DEFAULT)
    if state.macro_events:
        for ev in state.macro_events:
            try:
                m = ev.minutes_until()
            except Exception:
                continue
            if ev.impact == "High" and -60 <= m <= 240:
                kg_id = _classify_event_kg_node(ev.title)
                if kg_id and kg_id not in nodes:
                    nodes.append(kg_id)
    return nodes


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def memory_consolidator_node(state: AgentState) -> Dict[str, Any]:
    """Ghi episode hiện tại vào Graph RAG.

    Chạy ngay trước END. Không gọi LLM. Không có side effect ngoài
    việc upsert vào Chroma + write KG pickle.
    """
    if state.final_decision is None:
        # Không có quyết định để consolidate — giữ nguyên state.
        logger.debug("memory_consolidator: skip (no final_decision).")
        return {}

    try:
        # Lazy import GraphRAG để tránh ép cài extra agentic khi chỉ
        # chạy pipeline cũ (`gdt analyze`).
        from golddaytrading.agentic.tools.memory_tools import get_graph_rag
        from golddaytrading.agentic.graph_rag import HistoricalEpisode
    except ImportError as exc:
        logger.warning(
            "memory_consolidator: thiếu extra agentic (%s) — bỏ qua.",
            exc,
        )
        return {"errors": [f"memory_consolidator: import {exc}"]}

    try:
        narrative = _build_narrative(state)
        metadata = _build_metadata(state)
        entity_nodes = _build_entity_nodes(state)

        episode = HistoricalEpisode(
            episode_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            narrative=narrative,
            entity_nodes=entity_nodes,
            metadata=metadata,
        )

        rag = get_graph_rag()
        rag.ingest_episode(episode)

        log = DebateMessage(
            role="system",
            iteration=state.iteration,
            content=(
                f"[memory_consolidator] Đã ingest episode "
                f"`{episode.episode_id[:8]}` (regime={metadata.get('regime')}, "
                f"bias={metadata.get('bias')}, entities={entity_nodes})."
            ),
        )
        logger.info(
            "Memory consolidated: %s regime=%s bias=%s",
            episode.episode_id[:8],
            metadata.get("regime"),
            metadata.get("bias"),
        )
        return {"debate_history": [log]}

    except Exception as exc:  # PHẢI catch broad — không kill workflow
        logger.warning("memory_consolidator lỗi: %s", exc)
        log = DebateMessage(
            role="system",
            iteration=state.iteration,
            content=f"[memory_consolidator] Lỗi ingest: {exc}",
        )
        return {
            "debate_history": [log],
            "errors": [f"memory_consolidator: {exc}"],
        }
