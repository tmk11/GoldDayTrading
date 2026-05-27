"""Helper khoi tao Chat model cho LangGraph workflow.

Chi ho tro **API-only** (mac dinh OpenAI). Khac voi
``llm/client.py`` (cu, dung cho pipeline tuyen tinh), module nay
sinh ra `ChatModel` cua LangChain — kieu co san cac tien ich:

* ``.bind_tools([...])``               — gan tool cho LLM (hint cho
  function calling).
* ``.with_structured_output(Schema)``  — buoc model tra ve Pydantic
  object thay vi free text.

Hai tien ich nay la dieu kien tien quyet de pattern (B) — ToolNode
+ bound LLM + structured Risk Manager — chay duoc.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def get_chat_model(
    model: str,
    *,
    temperature: float = 0.2,
    timeout: float = 60.0,
    max_retries: int = 2,
):
    """Tra ve mot `BaseChatModel` cua LangChain.

    Hien chi ho tro OpenAI-compatible API — dung yeu cau "API-only".
    De dung provider khac (Anthropic, Gemini) trong tuong lai, mo
    rong switch tai day ma khong phai sua node nao.

    :param model: ten model do `cfg.deep_llm` / `cfg.quick_llm` /
                  env ``GDT_DEEP_LLM`` / ``GDT_QUICK_LLM`` truyen vao.
                  Vi du ``gpt-4o``, ``gpt-4o-mini``, hoac bat cu ten
                  nao provider ho tro — module **khong** hardcode.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Workflow agentic can `langchain-openai`. "
            "Cai: pip install -e \".[agentic]\""
        ) from exc

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "Workflow agentic chi hoat dong voi API key that "
            "(OPENAI_API_KEY chua duoc set). "
            "Pipeline tuyen tinh cu (`gdt analyze`) van chay duoc "
            "o che do offline heuristic."
        )

    base_url: Optional[str] = os.environ.get("OPENAI_BASE_URL") or None
    if timeout == 60.0:
        timeout = float(os.environ.get("GDT_LLM_TIMEOUT", timeout))
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        base_url=base_url,
    )
