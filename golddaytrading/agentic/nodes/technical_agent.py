"""Technical Agent Node — phân tích price action + liquidity.

Tool được bind:
* :func:`get_live_price`
* :func:`get_intraday_ohlcv`
* :func:`get_higher_timeframe_trend`
* :func:`get_active_session`
* :func:`get_level_pool`

Output: :class:`AgentOutput` với ``agent_name='technical'``.

Lưu ý quan trọng:

* Data Gatherer đã bơm sẵn block OHLCV + indicator vào state.
  Agent có thể đọc thẳng các block đó để **tiết kiệm token**, chỉ
  gọi tool khi cần verify hoặc xem khung khác (vd: 1m/5m).
* ``get_level_pool`` là tool *bắt buộc nên dùng* nếu agent đề xuất
  entry cụ thể — ta không muốn LLM bịa giá. Prompt yêu cầu rõ điều
  đó.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from golddaytrading.agentic.llm import get_chat_model
from golddaytrading.agentic.nodes._react_loop import run_react_agent
from golddaytrading.agentic.state import AgentState, DebateMessage
from golddaytrading.agentic.tools import TECHNICAL_AGENT_TOOLS

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
Bạn là Technical Agent — chuyên gia phân tích price action và
liquidity intraday cho XAU/USD (gold).

Trách nhiệm
-----------

1. Đọc indicator snapshot, OHLCV, session, HTF trend đã có sẵn
   trong message.
2. Khi cần thêm thông tin (ví dụ: nến 5m gần nhất, hoặc level pool
   với min_rr khác), hãy GỌI TOOL phù hợp. Đừng tự đoán giá.
3. Tổng hợp thành một `AgentOutput` Pydantic.

Quy tắc bắt buộc
----------------

- KHÔNG được bịa giá entry/stop/tp. Nếu bạn đề xuất giá cụ thể
  trong `key_levels`, phải gọi `get_level_pool` trước và lấy giá từ
  bảng đó (theo `setup_id`).
- Phải nêu rõ trong `summary`:
    * trend khung primary và HTF;
    * vị trí giá so với VWAP, EMA stack, Bollinger;
    * liquidity zones gần nhất (swing high/low, opening range,
      pivot D/W/M);
    * setup nào trong level pool (nếu có) đáng cân nhắc;
    * điều kiện invalidate setup.
- Nếu không có setup R:R đủ → `bias = NEUTRAL`.
- `confidence` đặt theo confluence:
    * 0.7+: nhiều tín hiệu đồng thuận, HTF cùng chiều, ATR vừa.
    * 0.4-0.7: tín hiệu mixed.
    * < 0.4: chỉ có 1 tín hiệu hoặc đang chop.
- `cited_sources`: liệt kê tên tool đã dùng (vd:
  "tool:get_level_pool", "block:indicator_snapshot").

Tóm tắt phải bằng TIẾNG VIỆT, ngắn gọn (<= 300 từ), markdown được.
"""


def _build_user_prompt(state: AgentState) -> str:
    md = state.market_data
    parts = [
        f"## Asset: {state.asset}",
        f"## Iteration: {state.iteration} (max {state.max_iterations})",
    ]
    if md:
        parts.append(
            f"- **Khung primary:** {md.primary_timeframe}  "
            f"·  **Khung HTF:** {md.higher_timeframe}  "
            f"·  **HTF trend tag:** `{md.htf_trend}`  "
            f"·  **Session:** `{md.active_session}`"
        )
        if md.last_price is not None:
            parts.append(f"- **Last price:** `{md.last_price:.2f}`  "
                         f"(close `{md.last_close:.2f}`)")
        if md.ohlcv_block:
            parts.append("\n" + md.ohlcv_block)
        if md.indicator_block:
            parts.append("\n" + md.indicator_block)
        if md.higher_indicator_block:
            parts.append(
                "\n### Indicator khung HTF (" + md.higher_timeframe + ")\n"
                + md.higher_indicator_block
            )

    # Nếu Risk Manager đã có feedback từ vòng trước, đính kèm
    feedback_msgs = [
        m for m in state.debate_history
        if m.role == "risk_manager" and m.iteration >= state.iteration
    ]
    if feedback_msgs:
        parts.append("\n### Phản hồi Risk Manager (cần làm rõ)\n")
        for m in feedback_msgs:
            parts.append(f"- {m.content}")

    parts.append(
        "\n**Yêu cầu:** phân tích kỹ thuật, đề xuất bias + confidence "
        "+ (tuỳ chọn) key_levels lấy từ `get_level_pool`. "
        "Không bịa giá. Trả lời theo schema `AgentOutput`."
    )
    return "\n".join(parts)


def technical_agent_node(state: AgentState) -> Dict[str, Any]:
    """Technical Agent node entry."""
    chat_model = get_chat_model(state.llm_model_deep, temperature=0.2)

    user_prompt = _build_user_prompt(state)

    output, tools_called = run_react_agent(
        chat_model=chat_model,
        tools=TECHNICAL_AGENT_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        agent_name="technical",
        max_tool_steps=4,
    )

    debate = DebateMessage(
        role="technical",
        iteration=state.iteration,
        content=output.summary,
    )

    logger.info(
        "[technical] bias=%s conf=%.2f tools=%s",
        output.bias, output.confidence, tools_called,
    )

    return {
        "agent_outputs": {"technical": output},
        "debate_history": [debate],
    }
