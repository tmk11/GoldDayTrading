"""Risk Manager Node — self-reflection + dynamic routing.

Khác hai agent worker, Risk Manager **không có tool**. Nó chỉ đọc
output của Technical + Macro/News, kiểm tra mâu thuẫn / blackout /
geometry, rồi:

* Hoặc **finalize**: phát ra `FinalDecision` (LONG/SHORT/NEUTRAL với
  entry/stop/tp/RR). Khi đó conditional edge route tới `END`.
* Hoặc **route_back**: chỉ định `next_node` tường minh (Technical
  hoặc Macro_News) cùng câu hỏi cụ thể cần làm rõ. Câu hỏi này được
  push vào ``debate_history`` để node target đọc ở vòng sau.

Cơ chế routing **A) Explicit Node Selection** (theo lựa chọn user):
LLM quyết định node nào quay lại, không phải heuristic Python.

Để đảm bảo output luôn parseable, ta dùng `with_structured_output`
với schema discriminated-union :class:`RiskManagerVerdict`.
"""

from __future__ import annotations

import logging
import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from golddaytrading.agentic.llm import get_chat_model
from golddaytrading.agentic.state import (
    AgentState,
    DebateMessage,
    FinalDecision,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verdict schema
# ---------------------------------------------------------------------------


class RiskManagerVerdict(BaseModel):
    """Output structured của Risk Manager.

    Hai trạng thái rời rạc:

    * ``action='finalize'`` → bắt buộc kèm `final_decision`.
    * ``action='route_back'`` → bắt buộc kèm `next_node` và
      `clarification_request` (câu hỏi cho node target).

    Validation cứng tại :meth:`_check_consistency` đảm bảo node sau
    không phải parse defensive.
    """

    action: Literal["finalize", "route_back"]

    # finalize branch
    final_decision: Optional[FinalDecision] = None

    # route_back branch
    next_node: Optional[Literal["technical_agent", "macro_news_agent"]] = None
    clarification_request: Optional[str] = Field(
        None,
        description=(
            "Câu hỏi cụ thể, dài <= 300 từ, viết bằng tiếng Việt, "
            "mô tả chính xác mâu thuẫn cần làm rõ."
        ),
    )

    # luôn có
    rationale: str = Field("", description="Tóm tắt suy luận <= 200 từ.")
    contradictions_detected: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> "RiskManagerVerdict":
        if self.action == "finalize":
            if self.final_decision is None:
                raise ValueError("action='finalize' phải có final_decision.")
            if self.next_node is not None:
                raise ValueError(
                    "action='finalize' không được kèm next_node."
                )
        else:  # route_back
            if self.next_node is None:
                raise ValueError("action='route_back' phải có next_node.")
            if not self.clarification_request:
                raise ValueError(
                    "action='route_back' phải có clarification_request."
                )
            if self.final_decision is not None:
                raise ValueError(
                    "action='route_back' không được kèm final_decision."
                )
        return self


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """\
Bạn là Risk Manager — vai trò self-reflection và *dynamic router*
trong workflow agentic cho XAU/USD intraday.

Bạn nhận output có cấu trúc từ:
- Technical Agent (price action, level pool)
- Macro & News Agent (DXY/yields, calendar, RSS, Graph RAG)

Hai lựa chọn output (không có lựa chọn thứ ba)
----------------------------------------------

A) `action = 'finalize'` — phát hành `FinalDecision`. Chỉ chọn khi:
   * không có mâu thuẫn nghiêm trọng giữa các agent, HOẶC
   * `state.iteration` đã chạm `max_iterations` (phải kết thúc).

B) `action = 'route_back'` — chỉ định `next_node` (technical_agent
   hoặc macro_news_agent) kèm `clarification_request` ngắn gọn.

Khi nào ROUTE BACK
------------------

- Technical bullish nhưng Macro bearish (hoặc ngược lại) → route về
  agent có confidence thấp hơn để xác minh.
- Có sự kiện high-impact trong < 15 phút mà chưa agent nào nhắc tới
  → route về macro_news_agent yêu cầu xác nhận blackout.
- Technical đề xuất entry không có `setup_id` từ level pool →
  route về technical_agent yêu cầu chọn từ level pool.
- Macro Agent chưa gọi `query_historical_context` ở regime hiếm →
  route về macro_news_agent yêu cầu gọi RAG.

Quy tắc cứng cho FINALIZE
-------------------------

- LONG/SHORT bắt buộc có `entry`, `stop_loss`, `take_profit_1`,
  `take_profit_2`, và `rr_ratio >= 1.5`. Geometry: với LONG thì
  `stop_loss < entry < take_profit_1 <= take_profit_2`.
- Lấy giá CHÍNH XÁC từ level pool đã có sẵn trong output của
  Technical Agent (`agent_outputs.technical.key_levels`). KHÔNG bịa
  giá. Nếu Technical không cung cấp key_levels mà bias không phải
  NEUTRAL → đó là contradiction → ROUTE BACK technical_agent.
- Nếu đang trong blackout window → bias buộc phải là NEUTRAL.
- Position size: dùng tham số risk-per-trade trong rationale.
- `time_in_force_minutes`: ước lượng tối đa thời gian giữ lệnh,
  thường <= 240 phút intraday.

`rationale` viết bằng TIẾNG VIỆT, <= 200 từ.
"""


def _build_user_prompt(state: AgentState) -> str:
    parts = [
        f"## Asset: {state.asset}",
        f"## Iteration: {state.iteration} / max {state.max_iterations}",
    ]

    if state.market_data and state.market_data.last_price is not None:
        parts.append(
            f"- **Last price:** `{state.market_data.last_price:.2f}`  ·  "
            f"**Session:** `{state.market_data.active_session}`"
        )

    # Output từ các agent
    tech = state.agent_outputs.get("technical")
    macro = state.agent_outputs.get("macro_news")

    parts.append("\n### Technical Agent output")
    if tech:
        parts.append(
            f"- bias=`{tech.bias}`, confidence=`{tech.confidence:.2f}`"
        )
        if tech.key_levels:
            parts.append(f"- key_levels: `{tech.key_levels}`")
        parts.append(f"- tools_called: {tech.tools_called}")
        parts.append(f"- summary: {tech.summary}")
    else:
        parts.append("_(chưa có)_")

    parts.append("\n### Macro & News Agent output")
    if macro:
        parts.append(
            f"- bias=`{macro.bias}`, confidence=`{macro.confidence:.2f}`"
        )
        parts.append(f"- tools_called: {macro.tools_called}")
        parts.append(f"- summary: {macro.summary}")
    else:
        parts.append("_(chưa có)_")

    # Cảnh báo
    contradiction = state.has_contradiction()
    consensus = state.consensus_bias()
    parts.append("\n### Helper deterministic")
    parts.append(f"- has_contradiction (helper): `{contradiction}`")
    parts.append(f"- consensus_bias (helper): `{consensus}`")

    # Sự kiện sắp tới
    soon = [
        e for e in state.macro_events
        if e.impact == "High" and 0 <= e.minutes_until() <= 30
    ]
    if soon:
        parts.append("\n### BLACKOUT có thể đang active")
        for e in soon:
            parts.append(
                f"- {e.title} ({e.country}) — trong {e.minutes_until():.0f} phút"
            )

    # Lịch sử iteration
    if state.iteration >= state.max_iterations - 1:
        parts.append(
            "\n**LƯU Ý:** đây là vòng cuối cùng có thể (iteration đã chạm "
            "max). Bạn BUỘC phải finalize, không được route_back."
        )

    parts.append(
        "\n**Yêu cầu:** trả về `RiskManagerVerdict` theo schema. "
        "Nếu finalize, lấy entry/stop/tp từ key_levels của Technical."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def risk_manager_node(state: AgentState) -> Dict[str, Any]:
    """Risk Manager node entry — phát hành verdict structured."""
    chat_model = get_chat_model(state.llm_model_deep, temperature=0.0)

    user_prompt = _build_user_prompt(state)

    # Hard rule: nếu chạm max_iterations thì cấm route_back ở Python level.
    forced_finalize = state.iteration >= (state.max_iterations - 1)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        structured = chat_model.with_structured_output(RiskManagerVerdict)
        raw_verdict = structured.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        verdict = _coerce_verdict(raw_verdict)
    except Exception as exc:
        logger.warning("Risk Manager structured output lỗi: %s", exc)
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            raw = chat_model.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
                HumanMessage(content=(
                    "Trả về DUY NHẤT một JSON object hợp lệ theo schema RiskManagerVerdict. "
                    "Nếu finalize NEUTRAL thì final_decision phải có bias, confidence, rationale; "
                    "time_in_force_minutes phải bỏ trống/null nếu không vào lệnh. Không thêm markdown."
                )),
            ])
            verdict = _coerce_verdict(_extract_json_object(getattr(raw, "content", raw)))
        except Exception as repair_exc:
            logger.warning("Risk Manager JSON repair lỗi: %s", repair_exc)
            return _fallback_finalize(state, reason=f"structured-output failed: {repair_exc}")

    # Override an toàn: nếu đã hết iteration mà LLM vẫn route_back →
    # ép về NEUTRAL finalize.
    if forced_finalize and verdict.action == "route_back":
        logger.warning(
            "Risk Manager cố route_back ở vòng cuối; ép NEUTRAL finalize."
        )
        return _fallback_finalize(
            state,
            reason="hết iteration nhưng LLM vẫn muốn route back",
            contradictions=verdict.contradictions_detected,
        )

    iteration_next = state.iteration + 1
    rm_log = DebateMessage(
        role="risk_manager",
        iteration=state.iteration,
        content=verdict.rationale,
    )

    if verdict.action == "finalize":
        logger.info(
            "[risk_manager] FINALIZE bias=%s conf=%.2f",
            verdict.final_decision.bias if verdict.final_decision else "?",
            verdict.final_decision.confidence if verdict.final_decision else 0.0,
        )
        return {
            "final_decision": verdict.final_decision,
            "next_node": "END",
            "iteration": iteration_next,
            "debate_history": [rm_log],
        }

    # route_back
    clar_msg = DebateMessage(
        role="risk_manager",
        iteration=iteration_next,  # gắn iteration KẾ TIẾP để target node thấy
        content=(
            f"[ROUTE BACK → {verdict.next_node}] "
            + (verdict.clarification_request or "")
        ),
    )
    logger.info(
        "[risk_manager] ROUTE BACK → %s  (iter %d → %d)",
        verdict.next_node, state.iteration, iteration_next,
    )
    return {
        "next_node": verdict.next_node,
        "iteration": iteration_next,
        "debate_history": [rm_log, clar_msg],
    }


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def _fallback_finalize(
    state: AgentState,
    *,
    reason: str,
    contradictions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Tạo FinalDecision NEUTRAL khi LLM lỗi hoặc hết iteration."""
    consensus = state.consensus_bias()
    if consensus == "NEUTRAL":
        decision = FinalDecision(
            bias="NEUTRAL",
            confidence=0.2,
            rationale=(
                "Risk Manager fallback NEUTRAL: " + reason +
                ". Consensus deterministic cũng là NEUTRAL."
            ),
            contradictions_resolved=contradictions or [],
        )
    else:
        # Có consensus nhưng RM lỗi → vẫn NEUTRAL cho an toàn
        decision = FinalDecision(
            bias="NEUTRAL",
            confidence=0.15,
            rationale=(
                "Risk Manager fallback NEUTRAL: " + reason +
                f". (Consensus deterministic = {consensus} nhưng giảm về "
                "NEUTRAL để an toàn vì không có verdict LLM hợp lệ.)"
            ),
            contradictions_resolved=contradictions or [],
        )

    log = DebateMessage(
        role="risk_manager",
        iteration=state.iteration,
        content=decision.rationale,
    )
    return {
        "final_decision": decision,
        "next_node": "END",
        "iteration": state.iteration + 1,
        "debate_history": [log],
        "errors": [f"risk_manager_fallback: {reason}"],
    }

def _extract_json_object(content: Any) -> Dict[str, Any]:
    text = content if isinstance(content, str) else str(content)
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("RiskManagerVerdict JSON must be an object")
    return data

def _coerce_verdict(raw: Any) -> RiskManagerVerdict:
    data = raw.model_dump() if isinstance(raw, BaseModel) else dict(raw)
    if not data.get("rationale"):
        final_decision = data.get("final_decision") or {}
        data["rationale"] = (
            final_decision.get("rationale")
            if isinstance(final_decision, dict) else None
        ) or "Risk Manager finalized after schema normalization."
    final_decision = data.get("final_decision")
    if isinstance(final_decision, dict):
        if final_decision.get("time_in_force_minutes") == 0:
            final_decision["time_in_force_minutes"] = None
        if not final_decision.get("rationale"):
            final_decision["rationale"] = data["rationale"]
    return RiskManagerVerdict(**data)
