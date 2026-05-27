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
import os
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from golddaytrading.agentic.llm import get_chat_model
from golddaytrading.agentic.state import (
    AgentState,
    DebateMessage,
    FinalDecision,
)

logger = logging.getLogger(__name__)

def _clip(text: object, limit: int = 900) -> str:
    value = str(text or "")
    return value if len(value) <= limit else value[:limit].rstrip() + "…"


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
  `take_profit_2`, và `rr_ratio >= 1.2`. Geometry: với LONG thì
  `stop_loss < entry < take_profit_1 <= take_profit_2`.
- `final_action = LONG` hoặc `SHORT` chỉ dùng khi setup đã trigger
  hoặc có thể vào ngay theo confirmation hiện tại.
- `final_action = LONG_SETUP` hoặc `SHORT_SETUP` dùng cho conditional
  setup: có entry/stop/take_profit/risk_reward hợp lệ từ level pool,
  nhưng giá chưa kích hoạt hoặc còn cần xác nhận. Đây là kế hoạch chờ,
  không phải lệnh market ngay.
- `final_action = NO_TRADE` chỉ dùng khi không có setup hợp lệ, R:R < 1.2,
  blackout, hoặc risk rules không đạt.
- Lấy giá CHÍNH XÁC từ level pool đã có sẵn trong output của
  Technical Agent (`agent_outputs.technical.key_levels`). KHÔNG bịa
  giá. Nếu Technical không cung cấp key_levels mà bias không phải
  NEUTRAL → đó là contradiction → ROUTE BACK technical_agent.
- Nếu đang trong blackout window → bias buộc phải là NEUTRAL.
- Position size: dùng tham số risk-per-trade trong rationale.
- `time_in_force_minutes`: ước lượng tối đa thời gian giữ lệnh,
  thường <= 240 phút intraday.
- Không dùng voting đơn giản. Bull/Bear case chỉ là bằng chứng đối
  lập; quyết định cuối phải dựa trên explicit risk rules: blackout,
  geometry, R:R, invalidation, confirmation, position sizing.
- Khi finalize, `FinalDecision` phải thể hiện các field downstream:
  `final_action` LONG/SHORT/LONG_SETUP/SHORT_SETUP/NO_TRADE,
  `confidence_score` 0-100,
  `entry`, `stop_loss`, `take_profit`, `risk_reward`,
  `position_size_recommendation`, `reasons`,
  `conditions_to_cancel_trade`.

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
    bull = state.bull_case
    bear = state.bear_case

    parts.append("\n### Technical Agent output")
    if tech:
        parts.append(
            f"- bias=`{tech.bias}`, confidence=`{tech.confidence:.2f}`"
        )
        if tech.key_levels:
            parts.append(f"- key_levels: `{tech.key_levels}`")
        parts.append(f"- tools_called: {tech.tools_called}")
        parts.append(f"- summary: {_clip(tech.summary)}")
    else:
        parts.append("_(chưa có)_")

    parts.append("\n### Macro & News Agent output")
    if macro:
        parts.append(
            f"- bias=`{macro.bias}`, confidence=`{macro.confidence:.2f}`"
        )
        parts.append(f"- tools_called: {macro.tools_called}")
        parts.append(f"- summary: {_clip(macro.summary)}")
    else:
        parts.append("_(chưa có)_")

    parts.append("\n### Bull Case Agent output")
    if bull:
        parts.append(f"- confidence=`{bull.confidence:.2f}`")
        parts.append(f"- bullish_thesis: {_clip(bull.thesis, 500)}")
        parts.append(f"- supporting_evidence: {bull.supporting_evidence}")
        parts.append(f"- required_confirmation: {_clip(bull.required_confirmation, 400)}")
        parts.append(f"- invalidation_level: {bull.invalidation_level}")
        parts.append(f"- risk_factors: {bull.risk_factors}")
    else:
        parts.append("_(debate skipped)_")

    parts.append("\n### Bear Case Agent output")
    if bear:
        parts.append(f"- confidence=`{bear.confidence:.2f}`")
        parts.append(f"- bearish_thesis: {_clip(bear.thesis, 500)}")
        parts.append(f"- supporting_evidence: {bear.supporting_evidence}")
        parts.append(f"- required_confirmation: {_clip(bear.required_confirmation, 400)}")
        parts.append(f"- invalidation_level: {bear.invalidation_level}")
        parts.append(f"- risk_factors: {bear.risk_factors}")
    else:
        parts.append("_(debate skipped)_")

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
        "Nếu finalize, lấy entry/stop/tp từ key_levels của Technical. "
        "Nếu setup hợp lệ nhưng chưa trigger, dùng LONG_SETUP/SHORT_SETUP thay vì NO_TRADE. "
        "Không dùng voting; hãy phân xử bull/bear bằng risk rules rõ ràng."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def risk_manager_node(state: AgentState) -> Dict[str, Any]:
    """Risk Manager node entry — phát hành verdict structured."""
    chat_model = get_chat_model(
        state.llm_model_deep,
        temperature=0.0,
        timeout=float(os.environ.get("GDT_RISK_MANAGER_TIMEOUT", os.environ.get("GDT_LLM_TIMEOUT", "120"))),
        max_retries=0,
    )

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
        if "timed out" in str(exc).lower() or "timeout" in str(exc).lower():
            return _fallback_finalize(state, reason=f"structured-output failed: {exc}")
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
    """Finalize defensively when Risk Manager LLM fails.

    Prefer a deterministic conditional setup from Technical level pool
    over blind NO_TRADE when entry/stop/tp/R:R are already available.
    """
    setup_decision = _deterministic_setup_decision(state, reason)
    if setup_decision is not None:
        log = DebateMessage(
            role="risk_manager",
            iteration=state.iteration,
            content=setup_decision.rationale,
        )
        return {
            "final_decision": setup_decision,
            "next_node": "END",
            "iteration": state.iteration + 1,
            "debate_history": [log],
            "errors": [f"risk_manager_fallback_setup: {reason}"],
        }

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

def _deterministic_setup_decision(state: AgentState, reason: str) -> Optional[FinalDecision]:
    min_rr = float(os.environ.get("GDT_AGENTIC_MIN_RR", "1.2"))
    candidates = _candidates_from_agent_key_levels(state, min_rr)
    if not candidates:
        candidates = _candidates_from_live_level_pool(state, min_rr)
    if not candidates:
        return None
    _, setup_id, fields, side = max(candidates, key=lambda item: item[0])
    entry = fields["entry"]
    stop = fields.get("stop") or fields.get("stop_loss")
    tp1 = fields.get("tp1") or fields.get("take_profit") or fields.get("take_profit_1")
    tp2 = fields.get("tp2") or fields.get("take_profit_2")
    rr = fields.get("rr1") or fields.get("rr") or fields.get("risk_reward") or fields.get("rr_ratio")
    last_price = state.market_data.last_price if state.market_data else None
    triggered = False
    if last_price is not None:
        triggered = (side == "LONG" and last_price >= entry) or (side == "SHORT" and last_price <= entry)
    final_action = side if triggered else f"{side}_SETUP"
    confidence = 0.58 if triggered else 0.54
    setup_kind = "active" if triggered else "conditional"
    rationale = (
        f"Risk Manager LLM fallback due to {reason}. Deterministic {setup_kind} "
        f"{setup_id} selected from Technical level pool because entry/stop/take-profit "
        f"geometry is valid and R:R {rr:.2f} >= {min_rr:.2f}."
    )
    confirmation = (
        f"Wait for price confirmation at/through {entry:.2f} before execution."
        if not triggered else "Setup trigger is already satisfied; still require fresh spread/liquidity check."
    )
    return FinalDecision(
        final_action=final_action, bias=side,
        confidence=confidence,
        rationale=rationale,
        entry=entry, stop_loss=stop, take_profit=tp1,
        take_profit_1=tp1, take_profit_2=tp2,
        rr_ratio=rr, risk_reward=rr,
        position_size_recommendation=(
            "0.5R until trigger confirms; do not increase size because RM LLM timed out."
            if not triggered else "Use configured risk-per-trade; no size increase because RM LLM timed out."
        ),
        reasons=[
            f"{setup_id} comes from deterministic Technical level pool.",
            f"R:R {rr:.2f} meets relaxed threshold {min_rr:.2f}.",
            confirmation,
            "Risk Manager LLM timed out, so decision is conservative and conditional.",
        ],
        conditions_to_cancel_trade=[
            f"Cancel if price does not confirm {entry:.2f}.",
            f"Invalidate if price accepts beyond stop {stop:.2f}.",
            "Cancel during high-impact blackout or abnormal spread/liquidity.",
        ],
    )

def _candidates_from_agent_key_levels(state: AgentState, min_rr: float):
    tech = state.agent_outputs.get("technical")
    if not tech or not tech.key_levels:
        return []
    setups: Dict[str, Dict[str, float]] = {}
    for raw_key, value in tech.key_levels.items():
        if not isinstance(value, (int, float)):
            continue
        if "." in raw_key:
            setup_id, field = raw_key.rsplit(".", 1)
        else:
            setup_id, field = "setup", raw_key
        setups.setdefault(setup_id, {})[field.lower()] = float(value)
    return _filter_setup_candidates(setups, min_rr)

def _candidates_from_live_level_pool(state: AgentState, min_rr: float):
    try:
        from golddaytrading.dataflows.indicators import compute_indicators
        from golddaytrading.dataflows.intraday_data import fetch_intraday_ohlcv
        from golddaytrading.signals.levels import build_level_pool
    except Exception as exc:  # pragma: no cover
        logger.warning("Risk Manager fallback level-pool import failed: %s", exc)
        return []
    try:
        df = fetch_intraday_ohlcv(state.asset, timeframe=state.primary_timeframe, bars=200)
        if df is None or df.empty:
            return []
        ind = compute_indicators(df)
    except Exception as exc:  # pragma: no cover
        logger.warning("Risk Manager fallback OHLCV failed: %s", exc)
        return []
    htf_trend = state.market_data.htf_trend if state.market_data else None
    pool = build_level_pool(df, ind, min_rr=min_rr, htf_trend=htf_trend)
    setups: Dict[str, Dict[str, float]] = {}
    for idea in pool.ideas:
        setups[idea.setup_id] = {
            "entry": float(idea.entry),
            "stop": float(idea.stop),
            "tp1": float(idea.tp1),
            "tp2": float(idea.tp2),
            "rr1": float(idea.rr1),
            "score": float(idea.score),
        }
    return _filter_setup_candidates(setups, min_rr)

def _filter_setup_candidates(setups: Dict[str, Dict[str, float]], min_rr: float):
    candidates = []
    for setup_id, fields in setups.items():
        entry = fields.get("entry")
        stop = fields.get("stop") or fields.get("stop_loss")
        tp1 = fields.get("tp1") or fields.get("take_profit") or fields.get("take_profit_1")
        rr = fields.get("rr1") or fields.get("rr") or fields.get("risk_reward") or fields.get("rr_ratio")
        if entry is None or stop is None or tp1 is None or rr is None or rr < min_rr:
            continue
        upper_id = setup_id.upper()
        if "SHORT" in upper_id or (stop > entry and tp1 < entry):
            side = "SHORT"
        elif "LONG" in upper_id or (stop < entry and tp1 > entry):
            side = "LONG"
        else:
            continue
        score = float(fields.get("score", 0.0)) + float(rr)
        candidates.append((score, setup_id, fields, side))
    return candidates

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
        if "final_action" in final_decision and "bias" not in final_decision:
            action = final_decision.get("final_action")
            if action == "NO_TRADE":
                final_decision["bias"] = "NEUTRAL"
            elif action == "LONG_SETUP":
                final_decision["bias"] = "LONG"
            elif action == "SHORT_SETUP":
                final_decision["bias"] = "SHORT"
            else:
                final_decision["bias"] = action
        if isinstance(final_decision.get("confidence"), (int, float)) and final_decision["confidence"] > 1:
            final_decision["confidence"] = float(final_decision["confidence"]) / 100.0
        if "confidence_score" in final_decision and "confidence" not in final_decision:
            final_decision["confidence"] = float(final_decision.get("confidence_score") or 0) / 100.0
        if "take_profit" in final_decision and "take_profit_1" not in final_decision:
            final_decision["take_profit_1"] = final_decision.get("take_profit")
        if "risk_reward" in final_decision and "rr_ratio" not in final_decision:
            final_decision["rr_ratio"] = final_decision.get("risk_reward")
        if final_decision.get("time_in_force_minutes") == 0:
            final_decision["time_in_force_minutes"] = None
        if not final_decision.get("rationale"):
            final_decision["rationale"] = data["rationale"]
    return RiskManagerVerdict(**data)
