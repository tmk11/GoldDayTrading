"""Runner cho LangGraph workflow agentic.

Cung cấp:

* :func:`run_agentic_workflow` — chạy 1 lần, trả về `AgentState` cuối.
* :func:`render_final_report`  — render markdown output cho CLI.

Đây là interface duy nhất mà CLI / test / external caller cần. Mọi
chi tiết về StateGraph / tool / node được đóng gói bên dưới.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from golddaytrading.agentic.graph import build_graph, get_default_graph
from golddaytrading.agentic.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_agentic_workflow(
    asset: str = "XAUUSD=X",
    *,
    deep_llm: Optional[str] = None,
    quick_llm: Optional[str] = None,
    max_iterations: int = 3,
    recursion_limit: int = 25,
    initial_state_overrides: Optional[Dict[str, Any]] = None,
) -> AgentState:
    """Chạy workflow LangGraph 1 lần.

    :param asset:          symbol (mặc định ``XAUUSD=X``).
    :param deep_llm:       override `state.llm_model_deep`. Nếu None,
                           dùng env ``GDT_DEEP_LLM`` hoặc ``gpt-4o``.
    :param quick_llm:      override `state.llm_model_quick`. Tương tự.
    :param max_iterations: số vòng lặp tối đa qua Risk Manager.
    :param recursion_limit: hard cap của LangGraph cho số lượt
                           transition. Để cao hơn `max_iterations *
                           4` để có biên an toàn.
    :param initial_state_overrides: dict ghi đè field bất kỳ của
                           AgentState ban đầu (advanced).
    """
    import os

    # Chuẩn bị state khởi tạo
    deep = deep_llm or os.environ.get("GDT_DEEP_LLM", "gpt-4o")
    quick = quick_llm or os.environ.get("GDT_QUICK_LLM", "gpt-4o-mini")

    init_kwargs: Dict[str, Any] = {
        "asset": asset,
        "timestamp": datetime.now(timezone.utc),
        "max_iterations": max_iterations,
        "iteration": 0,
        "llm_provider": "openai",
        "llm_model_deep": deep,
        "llm_model_quick": quick,
    }
    if initial_state_overrides:
        init_kwargs.update(initial_state_overrides)

    initial_state = AgentState(**init_kwargs)

    graph = get_default_graph()

    t0 = time.time()
    logger.info(
        "Khởi chạy agentic workflow: asset=%s deep_llm=%s max_iter=%d",
        asset, deep, max_iterations,
    )

    final_dict = graph.invoke(
        initial_state,
        config={"recursion_limit": recursion_limit},
    )

    # LangGraph có thể trả dict hoặc AgentState tuỳ version — chuẩn hoá.
    if isinstance(final_dict, AgentState):
        final_state = final_dict
    else:
        final_state = AgentState(**dict(final_dict))

    elapsed = time.time() - t0
    logger.info(
        "Workflow xong sau %.1fs, iterations=%d, final_bias=%s",
        elapsed,
        final_state.iteration,
        final_state.final_decision.bias if final_state.final_decision else "?",
    )
    return final_state


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render_final_report(state: AgentState) -> str:
    """Render markdown report cho CLI / journal."""
    lines = [
        "# Báo cáo Agentic Workflow",
        f"- **Asset:** `{state.asset}`",
        f"- **Timestamp:** {state.timestamp.isoformat()}",
        f"- **Iterations:** {state.iteration} / {state.max_iterations}",
        f"- **Model:** `{state.llm_model_deep}`",
    ]

    if state.market_data:
        md = state.market_data
        lines.append(
            f"- **Session / HTF trend:** `{md.active_session}` / `{md.htf_trend}`"
        )
        if md.last_price is not None:
            lines.append(f"- **Last price:** `{md.last_price:.2f}`")

    # Agent outputs
    lines.append("\n## Output từng Agent\n")
    for name in ("technical", "macro_news"):
        out = state.agent_outputs.get(name)
        if not out:
            continue
        lines.append(f"### {name}")
        lines.append(
            f"- bias=`{out.bias}`, confidence=`{out.confidence:.2f}`, "
            f"tools={out.tools_called}"
        )
        if out.key_levels:
            lines.append(f"- key_levels: `{out.key_levels}`")
        lines.append("")
        lines.append(out.summary)
        lines.append("")

    if state.bull_case or state.bear_case:
        lines.append("## Bull/Bear Debate (ngắn gọn)\n")
        for title, case in (("Bull Case", state.bull_case), ("Bear Case", state.bear_case)):
            if not case:
                continue
            lines.append(f"### {title}")
            lines.append(f"- thesis: {case.thesis}")
            if case.supporting_evidence:
                lines.append("- evidence: " + "; ".join(case.supporting_evidence[:4]))
            lines.append(f"- confirmation: {case.required_confirmation}")
            if case.invalidation_level is not None:
                lines.append(f"- invalidation: `{case.invalidation_level:.2f}`")
            if case.risk_factors:
                lines.append("- risks: " + "; ".join(case.risk_factors[:4]))
            lines.append("")

    # Final decision
    lines.append("## Final Decision\n")
    fd = state.final_decision
    if fd is None:
        lines.append("Status: `ERROR` — workflow kết thúc bất thường.")
    else:
        status_label = {
            "LONG": "READY_LONG",
            "SHORT": "READY_SHORT",
            "LONG_SETUP": "PENDING_LONG_SETUP",
            "SHORT_SETUP": "PENDING_SHORT_SETUP",
            "NO_TRADE": "NO_TRADE",
        }.get(fd.final_action, fd.final_action)
        lines.append(f"Action: `{fd.final_action}`  |  Status: `{status_label}`")
        lines.append(f"Confidence: `{fd.confidence_score}/100`")
        if fd.final_action != "NO_TRADE":
            lines.append(f"Entry: `{fd.entry:.2f}`")
            lines.append(f"Stop loss: `{fd.stop_loss:.2f}`")
            lines.append(f"Take profit: `{(fd.take_profit or fd.take_profit_1):.2f}`")
            if fd.take_profit_2 is not None:
                lines.append(f"Take profit 2: `{fd.take_profit_2:.2f}`")
            if fd.risk_reward is not None:
                lines.append(f"Risk/Reward: `{fd.risk_reward:.2f}`")
            if fd.position_size_recommendation:
                lines.append(f"Position size: {fd.position_size_recommendation}")
        else:
            lines.append("Entry: `—`")
            lines.append("Stop loss: `—`")
            lines.append("Take profit: `—`")
            lines.append("Risk/Reward: `—`")
        if fd.time_in_force_minutes:
            lines.append(f"Time in force: `{fd.time_in_force_minutes} phút`")
        if fd.blackout_warning:
            lines.append(f"Blackout: {fd.blackout_warning}")
        if fd.reasons:
            lines.append("Reasons:")
            for reason in fd.reasons[:4]:
                lines.append(f"- {reason}")
        else:
            lines.append(f"Reason: {fd.rationale}")
        if fd.conditions_to_cancel_trade:
            lines.append("Cancel / invalidate if:")
            for condition in fd.conditions_to_cancel_trade[:4]:
                lines.append(f"- {condition}")
        if fd.contradictions_resolved:
            lines.append("Resolved conflicts: " + "; ".join(fd.contradictions_resolved[:4]))
        lines.append("")

    # Debate log (rút gọn)
    if state.debate_history:
        lines.append("\n## Lịch sử debate (rút gọn)\n")
        for m in state.debate_history[-12:]:
            lines.append(
                f"- [iter {m.iteration}] **{m.role}**: {m.content[:240]}"
                + ("..." if len(m.content) > 240 else "")
            )

    if state.errors:
        lines.append("\n## Errors\n")
        for e in state.errors:
            lines.append(f"- {e}")

    return "\n".join(lines)
