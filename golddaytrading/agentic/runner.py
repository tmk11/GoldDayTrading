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

    # Final decision
    lines.append("## Quyết định cuối\n")
    fd = state.final_decision
    if fd is None:
        lines.append("_(không có — workflow kết thúc bất thường)_")
    else:
        lines.append(
            f"- **Bias:** `{fd.bias}`  ·  **Confidence:** `{fd.confidence:.2f}`"
        )
        if fd.bias != "NEUTRAL":
            lines.append(
                f"- Entry `{fd.entry:.2f}` · Stop `{fd.stop_loss:.2f}` · "
                f"TP1 `{fd.take_profit_1:.2f}`"
                + (f" · TP2 `{fd.take_profit_2:.2f}`"
                   if fd.take_profit_2 is not None else "")
            )
            if fd.rr_ratio is not None:
                lines.append(f"- R:R = `{fd.rr_ratio:.2f}`")
            if fd.position_size_units is not None:
                lines.append(f"- Size = `{fd.position_size_units}`")
            if fd.time_in_force_minutes:
                lines.append(f"- TIF = `{fd.time_in_force_minutes} phút`")
        if fd.blackout_warning:
            lines.append(f"- ⚠️ Blackout: {fd.blackout_warning}")
        if fd.contradictions_resolved:
            lines.append(
                "- Mâu thuẫn đã giải quyết: "
                + "; ".join(fd.contradictions_resolved)
            )
        lines.append("")
        lines.append("**Lý do:**")
        lines.append(fd.rationale)

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
