"""LLM client abstraction.

The pipeline only needs ``complete(system_prompt, user_prompt) -> str``.
We wrap OpenAI/Anthropic/Gemini behind that single method so swapping
providers is a config change. An ``offline`` provider returns a
deterministic heuristic answer so the smoke pipeline runs without
any API key.
"""

from golddaytrading.llm.client import LLMClient, build_client  # noqa: F401
