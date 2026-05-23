"""Day Trader — composes the final, copy-pastable trade plan."""

from __future__ import annotations

from datetime import datetime, timezone

from golddaytrading.agents import prompts
from golddaytrading.config import GDTConfig
from golddaytrading.llm.client import LLMClient


def day_trader(ctx: dict, llm: LLMClient, cfg: GDTConfig) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    guard = ctx.get("guardrail")
    guard_block = "_(no guardrail computed)_"
    if guard is not None:
        guard_block = (
            f"- approved: {guard.approved}\n"
            f"- position units: {guard.position_size_units}\n"
            f"- risk $: {guard.risk_dollars}\n"
            f"- R:R: {guard.rr_ratio}\n"
            f"- hard_block: {guard.hard_block}\n"
        )

    user = (
        f"Ticker: `{ctx['ticker']}`  |  Now: {ts}\n\n"
        "## Research-manager plan\n"
        + ctx.get("research_plan", "_skipped_")
        + "\n\n## Risk-manager report\n"
        + ctx.get("risk_report", "_skipped_")
        + "\n\n## Deterministic guardrail bundle\n"
        + guard_block
        + "\n\nNow output the final plan in the exact markdown skeleton."
    )

    return llm.complete(
        prompts.localise(prompts.DAY_TRADER, cfg.output_language),
        user,
        model=cfg.deep_llm,
        max_tokens=700,
    )
