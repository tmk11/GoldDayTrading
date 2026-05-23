"""Bull / Bear short debate + Research Manager synthesis.

The upstream framework runs multi-round debates that can stretch
the wall-clock to several minutes. Day-trading needs decisions in
seconds, so we cap debate at ``cfg.debate_rounds`` (default 1) and
keep each turn under ~200 words via the prompt.
"""

from __future__ import annotations

from golddaytrading.agents import prompts
from golddaytrading.config import GDTConfig
from golddaytrading.llm.client import LLMClient


def _analyst_pack(ctx: dict) -> str:
    """Stitch all analyst reports into a single context block."""
    parts = [
        "## Technical analyst", ctx.get("technical_report", "_skipped_"),
        "## Session strategist", ctx.get("session_report", "_skipped_"),
        "## Macro pulse", ctx.get("macro_report", "_skipped_"),
        "## News & catalysts", ctx.get("news_report", "_skipped_"),
        "## Sentiment", ctx.get("sentiment_report", "_skipped_"),
    ]
    return "\n\n".join(parts)


def bull_bear_debate(ctx: dict, llm: LLMClient, cfg: GDTConfig) -> dict:
    """Run a short bull/bear debate. Returns a dict of turn outputs."""
    pack = _analyst_pack(ctx)
    history: list[str] = []

    bull_user = pack + "\n\nProduce your bull case now."
    bull = llm.complete(
        prompts.localise(prompts.BULL_RESEARCHER, cfg.output_language),
        bull_user,
        model=cfg.quick_llm,
        max_tokens=400,
    )
    history.append("### Bull (round 1)\n" + bull)

    bear_user = (
        pack + "\n\n## Bull just argued:\n" + bull
        + "\n\nProduce your bear rebuttal now."
    )
    bear = llm.complete(
        prompts.localise(prompts.BEAR_RESEARCHER, cfg.output_language),
        bear_user,
        model=cfg.quick_llm,
        max_tokens=400,
    )
    history.append("### Bear (round 1)\n" + bear)

    # Optional second round (rarely needed intraday).
    for r in range(2, cfg.debate_rounds + 1):
        bull = llm.complete(
            prompts.localise(prompts.BULL_RESEARCHER, cfg.output_language),
            pack + "\n\n## Latest bear argument:\n" + bear
            + "\n\nProduce your follow-up bull case now.",
            model=cfg.quick_llm,
            max_tokens=300,
        )
        history.append(f"### Bull (round {r})\n" + bull)
        bear = llm.complete(
            prompts.localise(prompts.BEAR_RESEARCHER, cfg.output_language),
            pack + "\n\n## Latest bull argument:\n" + bull
            + "\n\nProduce your follow-up bear case now.",
            model=cfg.quick_llm,
            max_tokens=300,
        )
        history.append(f"### Bear (round {r})\n" + bear)

    return {"bull": bull, "bear": bear, "history": "\n\n".join(history)}


def research_manager(ctx: dict, llm: LLMClient, cfg: GDTConfig) -> str:
    """Synthesise the debate into a single recommendation."""
    debate = ctx.get("debate", {})
    user = (
        _analyst_pack(ctx)
        + "\n\n## Debate transcript\n"
        + debate.get("history", "_skipped_")
        + "\n\nProduce your synthesis now."
    )
    return llm.complete(
        prompts.localise(prompts.RESEARCH_MANAGER, cfg.output_language),
        user,
        model=cfg.deep_llm,
        max_tokens=900,
    )
