"""Vòng lặp ReAct dùng chung cho Technical và Macro agent.

Mỗi agent node thực hiện 2 giai đoạn:

1. **Tool-calling loop** — LLM được ``bind_tools(...)``. Trong tối
   đa ``max_tool_steps`` lượt:
   * gọi LLM với message hiện tại;
   * nếu LLM trả về ``tool_calls`` → thực thi tool và append
     ``ToolMessage`` vào lịch sử;
   * nếu không → thoát vòng lặp.

2. **Structured output** — gọi LLM lần cuối với
   ``.with_structured_output(AgentOutput)`` để buộc trả về Pydantic
   ``AgentOutput``.

Tách riêng helper này để Technical và Macro chỉ khác nhau ở:
* danh sách tool được bind,
* system prompt,
* tên agent (để gắn vào ``AgentOutput.agent_name``).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Tuple

from golddaytrading.agentic.state import AgentOutput

logger = logging.getLogger(__name__)


def _tool_registry(tools: Iterable[Any]) -> Dict[str, Any]:
    """Map ``tool.name`` → tool callable. Sửa nếu LangChain đổi attr."""
    reg: Dict[str, Any] = {}
    for t in tools:
        name = getattr(t, "name", None) or getattr(t, "__name__", None)
        if name:
            reg[name] = t
    return reg


def run_react_agent(
    *,
    chat_model: Any,
    tools: List[Any],
    system_prompt: str,
    user_prompt: str,
    agent_name: str,
    max_tool_steps: int = 4,
) -> Tuple[AgentOutput, List[str]]:
    """Chạy ReAct loop và buộc structured output.

    Trả về ``(AgentOutput, tool_names_called)``. Caller chịu trách
    nhiệm push output vào ``state.agent_outputs``.

    Robustness:
    * Nếu tool throw exception → trả ToolMessage chứa ``error: …``
      để LLM tự thấy và recover, không kill node.
    * Nếu structured-output call fail → trả về AgentOutput NEUTRAL
      kèm summary cảnh báo, để Risk Manager vẫn route được.
    """
    from langchain_core.messages import (
        AIMessage, HumanMessage, SystemMessage, ToolMessage,
    )

    registry = _tool_registry(tools)
    bound = chat_model.bind_tools(tools) if tools else chat_model

    messages: List[Any] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    tools_called: List[str] = []

    for step in range(max_tool_steps):
        try:
            ai_msg = bound.invoke(messages)
        except Exception as exc:
            logger.warning("[%s] LLM call lỗi step=%d: %s",
                           agent_name, step, exc)
            return _fallback_output(agent_name, str(exc)), tools_called

        messages.append(ai_msg)
        tool_calls = getattr(ai_msg, "tool_calls", None) or []

        if not tool_calls:
            break  # LLM đã đưa ra answer, sang structured output

        for tc in tool_calls:
            name = tc.get("name")
            args = tc.get("args") or {}
            tool_call_id = tc.get("id") or ""
            tools_called.append(str(name))
            tool_fn = registry.get(name)
            if tool_fn is None:
                content = f"Error: unknown tool `{name}`."
            else:
                try:
                    content = tool_fn.invoke(args)
                except Exception as exc:  # tool-side fail, không kill node
                    logger.warning("[%s] tool %s lỗi: %s", agent_name, name, exc)
                    content = f"Error: tool `{name}` failed: {exc}"
            if not isinstance(content, str):
                content = str(content)
            messages.append(ToolMessage(
                content=content,
                tool_call_id=tool_call_id,
                name=name,
            ))
    else:
        # Hết max_tool_steps mà vẫn còn tool_calls → ghi note rồi force final.
        messages.append(HumanMessage(content=(
            "Đã hết ngân sách gọi tool. Giờ hãy tổng hợp tất cả thông tin "
            "đã có và trả lời theo schema AgentOutput."
        )))

    # ----- 2. Structured final output -----
    try:
        structured = chat_model.with_structured_output(AgentOutput)
        final_msg = HumanMessage(content=(
            "Dựa trên TOÀN BỘ thảo luận và tool output ở trên, "
            f"hãy trả về một `AgentOutput` (agent_name = '{agent_name}'). "
            "Trường `summary` phải <= 300 từ, viết bằng tiếng Việt. "
            "Trường `key_levels` chỉ điền khi bạn có đề xuất giá cụ thể. "
            "Trường `tools_called` để TRỐNG (caller sẽ điền)."
        ))
        result = structured.invoke(messages + [final_msg])
        if not isinstance(result, AgentOutput):
            result = AgentOutput(**_normalize_agent_output(dict(result)))  # type: ignore[arg-type]
        result.agent_name = agent_name
        result.tools_called = tools_called
        return result, tools_called
    except Exception as exc:
        logger.warning("[%s] structured output lỗi: %s", agent_name, exc)

    try:
        raw = chat_model.invoke(messages + [HumanMessage(content=(
            "Trả về DUY NHẤT một JSON object hợp lệ theo schema AgentOutput: "
            "agent_name:string, bias:LONG|SHORT|NEUTRAL, confidence:number 0..1, "
            "summary:string, key_levels:object, cited_sources:array, tools_called:array. "
            "Nếu có nhiều setup/level, hãy đặt vào key_levels dưới dạng object, "
            "không dùng array. Không thêm markdown ngoài JSON."
        ))])
        payload = _extract_json_object(getattr(raw, "content", raw))
        result = AgentOutput(**_normalize_agent_output(payload))
        result.agent_name = agent_name
        result.tools_called = tools_called
        return result, tools_called
    except Exception as exc:
        logger.warning("[%s] JSON repair output lỗi: %s", agent_name, exc)
        return _fallback_output(agent_name, f"structured-output failed: {exc}"), tools_called

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
        raise ValueError("AgentOutput JSON must be an object")
    return data

def _normalize_agent_output(data: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(data)
    key_levels = data.get("key_levels")
    if isinstance(key_levels, list):
        normalized: Dict[str, Any] = {}
        for idx, item in enumerate(key_levels, start=1):
            if isinstance(item, dict):
                prefix = str(item.get("setup_id") or item.get("setup") or f"setup_{idx}")
                for key, value in item.items():
                    if isinstance(value, (int, float)):
                        normalized[f"{prefix}.{key}"] = float(value)
            elif isinstance(item, (int, float)):
                normalized[f"level_{idx}"] = float(item)
        data["key_levels"] = normalized
    elif not isinstance(key_levels, dict):
        data["key_levels"] = {}
    else:
        data["key_levels"] = {
            str(key): float(value)
            for key, value in key_levels.items()
            if isinstance(value, (int, float))
        }
    return data


def _fallback_output(agent_name: str, reason: str) -> AgentOutput:
    return AgentOutput(
        agent_name=agent_name,
        bias="NEUTRAL",
        confidence=0.0,
        summary=(
            f"_Agent `{agent_name}` không hoàn thành được phân tích "
            f"({reason}). Risk Manager nên giảm size hoặc giữ NEUTRAL._"
        ),
        key_levels={},
        cited_sources=[],
        tools_called=[],
    )
