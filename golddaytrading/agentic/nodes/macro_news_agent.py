"""Macro & News Agent Node — driver vĩ mô + catalyst tin tức + RAG.

Tool được bind:
* :func:`get_macro_pulse`            — DXY/yields/VIX/realyield…
* :func:`get_econ_calendar`          — sự kiện 24h tới
* :func:`get_gold_news`               — RSS gold-relevant
* :func:`query_historical_context`    — Graph RAG (KG + Vector)

Output: :class:`AgentOutput` với ``agent_name='macro_news'``.

Node này là **người dùng chính của Graph RAG**. Prompt khuyến khích
agent gọi ``query_historical_context`` ít nhất 1 lần khi:

* phát hiện regime hiếm (REAL_YIELD_DRIVE / RISK_OFF_HAVEN_BID),
* hoặc có sự kiện high-impact sắp xảy ra (FOMC/CPI/NFP),

để retrieve các tình huống quá khứ tương tự và tránh "phản ứng theo
quán tính".
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from golddaytrading.agentic.llm import get_chat_model
from golddaytrading.agentic.nodes._react_loop import run_react_agent
from golddaytrading.agentic.state import AgentState, DebateMessage
from golddaytrading.agentic.tools import MACRO_NEWS_AGENT_TOOLS

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
Bạn là Macro & News Agent — chuyên gia về driver vĩ mô và catalyst
tin tức cho XAU/USD (gold).

Trách nhiệm
-----------

1. Đọc các block đã được Data Gatherer bơm sẵn vào prompt:
   `macro_pulse_block`, `calendar_block`, `news_block`.
2. Khi cần ngữ cảnh lịch sử để cân nhắc một regime đặc biệt hoặc
   một sự kiện sắp xảy ra, GỌI tool `query_historical_context`
   với câu hỏi tự nhiên cụ thể. Đây là **bộ nhớ chiến lược** —
   coi đó là Bayesian prior, không phải dự báo.
3. Tổng hợp thành `AgentOutput` Pydantic.

Khi nào nên gọi `query_historical_context`
------------------------------------------

- Khi regime tag là `REAL_YIELD_DRIVE` hoặc `RISK_OFF_HAVEN_BID`
  hoặc `GROWTH_SCARE` (các regime ít gặp, prior LLM kém).
- Khi có sự kiện High impact trong < 4h (CPI / FOMC / NFP / ECB).
- Khi DXY và US10Y phân kỳ rõ rệt — đó là tình huống mà LLM dễ trả
  lời sai nhất.
- Khi có headline đặc biệt mạnh (chiến tranh, phá sản ngân hàng,
  central-bank surprise).

Quy tắc bắt buộc
----------------

- Phải phân biệt rõ trong `summary`:
    * driver vĩ mô (DXY / yields / VIX / EURUSD) đang đẩy gold theo
      hướng nào trong 1-4 giờ tới;
    * catalyst tin tức nào nổi bật và độ tin cậy của nguồn;
    * có **blackout window** từ lịch kinh tế không (sự kiện High
      impact trong < 30 phút)? Nếu có, NEUTRAL là mặc định;
    * tóm tắt insight rút ra từ Graph RAG (nếu đã gọi).
- `confidence` cao chỉ khi nhiều driver đồng thuận VÀ Graph RAG có
  ít nhất 1 episode lịch sử khớp.
- `cited_sources` liệt kê tool đã gọi và (nếu có) headline RSS.

Tóm tắt phải bằng TIẾNG VIỆT, ngắn gọn (<= 300 từ), markdown được.
"""


def _build_user_prompt(state: AgentState) -> str:
    parts = [
        f"## Asset: {state.asset}",
        f"## Iteration: {state.iteration} (max {state.max_iterations})",
        f"## Timestamp: {state.timestamp.isoformat()}",
    ]

    if state.market_data:
        md = state.market_data
        parts.append(
            f"- **Session:** `{md.active_session}`  ·  "
            f"**HTF trend tag:** `{md.htf_trend}`"
        )
        if md.last_price is not None:
            parts.append(f"- **Last price:** `{md.last_price:.2f}`")

    if state.macro_pulse_block:
        parts.append("\n" + state.macro_pulse_block)
    if state.calendar_block:
        parts.append("\n" + state.calendar_block)
    if state.news_block:
        parts.append("\n" + state.news_block)

    if state.macro_events:
        # Highlight các sự kiện gần nhất < 60 phút
        soon = [e for e in state.macro_events if 0 < e.minutes_until() <= 60]
        if soon:
            parts.append("\n### CẢNH BÁO: sự kiện high-impact trong 60' tới")
            for e in soon:
                parts.append(
                    f"- {e.title} ({e.country}, {e.impact}) — "
                    f"trong {e.minutes_until():.0f} phút"
                )

    # Phản hồi Risk Manager (nếu route lại)
    feedback_msgs = [
        m for m in state.debate_history
        if m.role == "risk_manager" and m.iteration >= state.iteration
    ]
    if feedback_msgs:
        parts.append("\n### Phản hồi Risk Manager (cần làm rõ)")
        for m in feedback_msgs:
            parts.append(f"- {m.content}")

    # Output từ Technical Agent (để Macro Agent biết Tech đang nghĩ gì)
    tech_out = state.agent_outputs.get("technical")
    if tech_out:
        parts.append("\n### Output mới nhất của Technical Agent")
        parts.append(
            f"- bias=`{tech_out.bias}`, confidence=`{tech_out.confidence:.2f}`"
        )
        parts.append(f"- summary: {tech_out.summary[:500]}")

    parts.append(
        "\n**Yêu cầu:** phân tích driver vĩ mô + catalyst tin tức, "
        "gọi `query_historical_context` khi cần prior lịch sử, và "
        "trả về một `AgentOutput`."
    )
    return "\n".join(parts)


def macro_news_agent_node(state: AgentState) -> Dict[str, Any]:
    """Macro & News Agent node entry."""
    chat_model = get_chat_model(state.llm_model_deep, temperature=0.2)

    user_prompt = _build_user_prompt(state)

    output, tools_called = run_react_agent(
        chat_model=chat_model,
        tools=MACRO_NEWS_AGENT_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        agent_name="macro_news",
        max_tool_steps=4,
    )

    debate = DebateMessage(
        role="macro_news",
        iteration=state.iteration,
        content=output.summary,
    )

    logger.info(
        "[macro_news] bias=%s conf=%.2f tools=%s",
        output.bias, output.confidence, tools_called,
    )

    return {
        "agent_outputs": {"macro_news": output},
        "debate_history": [debate],
    }
