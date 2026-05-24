"""LLM client wrapper.

Supported providers:

* ``openai``    — uses the official ``openai>=1.0`` SDK.
* ``anthropic`` — optional, requires the ``anthropic`` extra.
* ``gemini``    — optional, requires the ``gemini`` extra.
* ``offline``   — no network call; returns a deterministic heuristic
  string. Useful for smoke tests, CI, and demos without API keys.

The interface is a single ``LLMClient.complete(system, user, model)``
method returning plain text. Agents do not assume any provider-
specific feature (function calling, tools, structured output) so the
abstraction stays minimal.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMClient:
    provider: str
    model: str
    _impl: object = None      # provider-specific SDK client (or None)

    def complete(
        self,
        system: str,
        user: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> str:
        """Synchronous chat completion. Returns assistant text.

        On any provider-side error we fall back to the offline
        heuristic so the pipeline never crashes mid-run because of a
        transient LLM outage. The error is logged.
        """
        target = model or self.model
        try:
            if self.provider == "openai":
                return _complete_openai(self._impl, system, user, target,
                                        temperature, max_tokens)
            if self.provider == "anthropic":
                return _complete_anthropic(self._impl, system, user, target,
                                           temperature, max_tokens)
            if self.provider == "gemini":
                return _complete_gemini(self._impl, system, user, target,
                                        temperature, max_tokens)
        except Exception as exc:
            logger.warning("LLM provider %s call failed: %s — using offline "
                           "heuristic for this turn.", self.provider, exc)
        return _offline_heuristic(system, user)


# ---------- builder ----------------------------------------------------------


def build_client(provider: str, model: str) -> LLMClient:
    """Construct a client for the requested provider.

    If the provider's SDK isn't installed or the API key is missing,
    we silently downgrade to ``offline`` so the pipeline still runs.
    """
    provider = (provider or "openai").lower()

    if provider == "offline":
        return LLMClient(provider="offline", model=model, _impl=None)

    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return LLMClient(provider="offline", model=model, _impl=None)
        try:
            from openai import OpenAI
            timeout = float(os.environ.get("OPENAI_TIMEOUT", "30"))
            return LLMClient(provider="openai", model=model,
                             _impl=OpenAI(timeout=timeout, max_retries=0))
        except ImportError:
            logger.warning("openai SDK not installed — offline mode")
            return LLMClient(provider="offline", model=model, _impl=None)

    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return LLMClient(provider="offline", model=model, _impl=None)
        try:
            import anthropic
            return LLMClient(provider="anthropic", model=model,
                             _impl=anthropic.Anthropic())
        except ImportError:
            logger.warning("anthropic SDK not installed — offline mode")
            return LLMClient(provider="offline", model=model, _impl=None)

    if provider == "gemini":
        if not os.environ.get("GOOGLE_API_KEY"):
            return LLMClient(provider="offline", model=model, _impl=None)
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
            return LLMClient(provider="gemini", model=model, _impl=genai)
        except ImportError:
            logger.warning("google-generativeai not installed — offline mode")
            return LLMClient(provider="offline", model=model, _impl=None)

    logger.warning("Unknown LLM provider %r — using offline mode", provider)
    return LLMClient(provider="offline", model=model, _impl=None)


# ---------- provider implementations ----------------------------------------


def _complete_openai(client, system: str, user: str, model: str,
                     temperature: float, max_tokens: int) -> str:
    rsp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (rsp.choices[0].message.content or "").strip()


def _complete_anthropic(client, system: str, user: str, model: str,
                        temperature: float, max_tokens: int) -> str:
    msg = client.messages.create(
        model=model,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": user}],
    )
    parts = []
    for block in getattr(msg, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return ("".join(parts)).strip()


def _complete_gemini(genai, system: str, user: str, model: str,
                     temperature: float, max_tokens: int) -> str:
    gm = genai.GenerativeModel(
        model_name=model,
        system_instruction=system,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        },
    )
    rsp = gm.generate_content(user)
    return (getattr(rsp, "text", "") or "").strip()


# ---------- offline heuristic -----------------------------------------------


def _offline_heuristic(system: str, user: str) -> str:
    """Deterministic placeholder reply for offline / no-key runs.

    The output is *useful* — it acknowledges the role from the system
    prompt and echoes back the most recent indicator/macro lines from
    the user prompt, which is enough to drive the downstream pipeline
    end-to-end (smoke tests, demos) without a real LLM.
    """
    role_line = ""
    for line in system.splitlines():
        s = line.strip()
        if s.startswith("You are") or s.startswith("Role:"):
            role_line = s
            break

    # Pull a few indicator-y lines from the user prompt to make the
    # response look grounded.
    snippets = []
    for line in user.splitlines():
        s = line.strip()
        if s.startswith(("- ", "* ")) and any(
            tag in s for tag in
            ("RSI", "MACD", "ATR", "EMA", "VWAP", "Pivot",
             "DXY", "TNX", "VIX", "session", "Last close", "Trend",
             "Latest", "in 1h", "in 4h")
        ):
            snippets.append(s)
        if len(snippets) >= 6:
            break

    body = (
        "Offline heuristic mode (no LLM key configured). "
        "Reasoning is rule-based and deterministic.\n\n"
    )
    if role_line:
        body += f"_{role_line}_\n\n"
    if snippets:
        body += "Key inputs noted:\n" + "\n".join(snippets) + "\n\n"
    body += (
        "Heuristic call: lean on confluence of EMA stack, VWAP "
        "relation, and RSI extremes. With no LLM available, the "
        "downstream Day Trader will compose a NEUTRAL plan and the "
        "Risk Manager will reduce size accordingly."
    )
    return body
