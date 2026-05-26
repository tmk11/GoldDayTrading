"""Lightweight bull/bear debate nodes for the LangGraph workflow."""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal

from pydantic import ValidationError

from golddaytrading.agentic.llm import get_chat_model
from golddaytrading.agentic.state import AgentState, DebateCase, DebateMessage

logger = logging.getLogger(__name__)

_BULL_SYSTEM = """\
Bạn là Bull Case Agent. Nhiệm vụ của bạn KHÔNG phải quyết định trade cuối.
Bạn chỉ xây dựng kịch bản LONG tốt nhất có thể dựa trên dữ liệu đã có.
Không bỏ qua risk. Không bịa giá. Nếu cần level, dùng level đã có trong Technical output.
Trả về DebateCase side='bull' gồm: thesis, supporting_evidence,
required_confirmation, invalidation_level, risk_factors, confidence.
Viết ngắn gọn bằng tiếng Việt.
"""

_BEAR_SYSTEM = """\
Bạn là Bear Case Agent. Nhiệm vụ của bạn KHÔNG phải quyết định trade cuối.
Bạn chỉ xây dựng kịch bản SHORT tốt nhất có thể dựa trên dữ liệu đã có.
Không bỏ qua risk. Không bịa giá. Nếu cần level, dùng level đã có trong Technical output.
Trả về DebateCase side='bear' gồm: thesis, supporting_evidence,
required_confirmation, invalidation_level, risk_factors, confidence.
Viết ngắn gọn bằng tiếng Việt.
"""

def _build_prompt(state: AgentState, side: Literal["bull", "bear"]) -> str:
    tech = state.agent_outputs.get("technical")
    macro = state.agent_outputs.get("macro_news")
    md = state.market_data
    parts = [
        f"## Asset: {state.asset}",
        f"## Debate side: {side}",
        f"## Debate trigger: {state.debate_reason or 'unspecified'}",
    ]
    if md:
        parts.append(
            f"- Last price: `{md.last_price}` · Session: `{md.active_session}` · "
            f"HTF trend: `{md.htf_trend}`"
        )
        if md.indicator_block:
            parts.append("\n### Indicator snapshot\n" + md.indicator_block)
    parts.append("\n### Technical Agent")
    if tech:
        parts.append(f"- bias={tech.bias}, confidence={tech.confidence:.2f}")
        if tech.key_levels:
            parts.append(f"- key_levels={tech.key_levels}")
        parts.append(tech.summary)
    else:
        parts.append("_(missing)_")
    parts.append("\n### Macro/News Agent")
    if macro:
        parts.append(f"- bias={macro.bias}, confidence={macro.confidence:.2f}")
        parts.append(macro.summary)
    else:
        parts.append("_(missing)_")
    if state.macro_events:
        parts.append("\n### High-impact events")
        for event in state.macro_events[:6]:
            parts.append(
                f"- {event.title} ({event.country}, {event.impact}) "
                f"in {event.minutes_until():.0f} minutes"
            )
    parts.append(
        "\nReturn only the structured DebateCase. Keep thesis <= 120 words, "
        "supporting_evidence <= 5 bullets, risk_factors <= 5 bullets."
    )
    return "\n".join(parts)

def _fallback_case(state: AgentState, side: Literal["bull", "bear"], reason: str) -> DebateCase:
    tech = state.agent_outputs.get("technical")
    invalidation = None
    if tech and tech.key_levels:
        stop_keys = [key for key in tech.key_levels if "stop" in key.lower()]
        if stop_keys:
            invalidation = tech.key_levels[stop_keys[0]]
    return DebateCase(
        side=side,
        thesis=f"Không tạo được {side} case đầy đủ; giữ vai trò advisory only.",
        supporting_evidence=[f"Fallback reason: {reason}"],
        required_confirmation="Risk Manager must require fresh price confirmation before any trade.",
        invalidation_level=invalidation,
        risk_factors=["Debate agent output unavailable or invalid."],
        confidence=0.0,
    )

def _run_case_node(state: AgentState, side: Literal["bull", "bear"]) -> Dict[str, Any]:
    chat_model = get_chat_model(state.llm_model_deep, temperature=0.15)
    system = _BULL_SYSTEM if side == "bull" else _BEAR_SYSTEM
    prompt = _build_prompt(state, side)
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        structured = chat_model.with_structured_output(DebateCase)
        case = structured.invoke([
            SystemMessage(content=system),
            HumanMessage(content=prompt),
        ])
        if not isinstance(case, DebateCase):
            case = DebateCase(**dict(case))
        case.side = side
    except (Exception, ValidationError) as exc:
        logger.warning("[%s_case] structured output failed: %s", side, exc)
        case = _fallback_case(state, side, str(exc))
    content = (
        f"{case.thesis} Confirmation: {case.required_confirmation} "
        f"Invalidation: {case.invalidation_level}. "
        f"Risks: {'; '.join(case.risk_factors[:3])}"
    )
    return {
        f"{side}_case": case,
        "debate_history": [DebateMessage(
            role=f"{side}_case", iteration=state.iteration, content=content,
        )],
    }

def bull_case_agent_node(state: AgentState) -> Dict[str, Any]:
    return _run_case_node(state, "bull")

def bear_case_agent_node(state: AgentState) -> Dict[str, Any]:
    return _run_case_node(state, "bear")
