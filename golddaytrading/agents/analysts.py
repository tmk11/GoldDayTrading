"""Analyst agents: Technical, Session, Macro Pulse, News, Sentiment.

Each analyst is a thin wrapper that:

1. Pulls the relevant context block(s) from ``ctx``.
2. Builds a user prompt by concatenating those blocks.
3. Calls the LLM with the role's system prompt.
4. Returns the assistant's report (markdown).

The pipeline orchestrator stores each return value back on ``ctx``
under a stable key (e.g. ``ctx["technical_report"]``) so downstream
agents can read it.
"""

from __future__ import annotations

from golddaytrading.agents import prompts
from golddaytrading.config import GDTConfig
from golddaytrading.llm.client import LLMClient


def _user_prompt(blocks: list[str]) -> str:
    """Stitch context blocks into a single user prompt."""
    return "\n\n".join(b for b in blocks if b).strip()


def technical_analyst(ctx: dict, llm: LLMClient, cfg: GDTConfig) -> str:
    user = _user_prompt([
        f"## Ticker: `{ctx['ticker']}` — primary tf `{cfg.primary_timeframe}`, "
        f"higher tf `{cfg.higher_timeframe}`",
        ctx.get("price_block", ""),
        ctx.get("ohlcv_block", ""),
        ctx.get("indicator_block", ""),
        ctx.get("higher_indicator_block", ""),
        ctx.get("quant_signal_block", ""),
        "Produce your structured technical read now.",
    ])
    return llm.complete(
        prompts.localise(prompts.TECHNICAL_ANALYST, cfg.output_language),
        user,
        model=cfg.quick_llm,
    )


def session_strategist(ctx: dict, llm: LLMClient, cfg: GDTConfig) -> str:
    user = _user_prompt([
        ctx.get("session_block", ""),
        "## Technical context",
        ctx.get("technical_report", "_(technical analyst skipped)_"),
        f"Preferred sessions for new entries: "
        f"{', '.join(cfg.preferred_sessions)}.",
        "Produce your session read now.",
    ])
    return llm.complete(
        prompts.localise(prompts.SESSION_STRATEGIST, cfg.output_language),
        user,
        model=cfg.quick_llm,
    )


def macro_pulse_analyst(ctx: dict, llm: LLMClient, cfg: GDTConfig) -> str:
    user = _user_prompt([
        ctx.get("macro_pulse_block", ""),
        "Produce your intraday macro read now.",
    ])
    return llm.complete(
        prompts.localise(prompts.MACRO_PULSE_ANALYST, cfg.output_language),
        user,
        model=cfg.quick_llm,
    )


def news_catalyst_agent(ctx: dict, llm: LLMClient, cfg: GDTConfig) -> str:
    user = _user_prompt([
        ctx.get("news_block", ""),
        ctx.get("calendar_block", ""),
        "Produce your news/catalyst read now.",
    ])
    return llm.complete(
        prompts.localise(prompts.NEWS_CATALYST_AGENT, cfg.output_language),
        user,
        model=cfg.quick_llm,
    )


def sentiment_agent(ctx: dict, llm: LLMClient, cfg: GDTConfig) -> str:
    user = _user_prompt([
        ctx.get("price_block", ""),
        ctx.get("indicator_block", ""),
        "## Technical context",
        ctx.get("technical_report", "_(technical analyst skipped)_"),
        "Produce your sentiment read now.",
    ])
    return llm.complete(
        prompts.localise(prompts.SENTIMENT_AGENT, cfg.output_language),
        user,
        model=cfg.quick_llm,
    )
