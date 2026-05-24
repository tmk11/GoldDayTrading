"""Bull / Bear short debate + Research Manager synthesis.

The upstream framework runs multi-round debates that can stretch
the wall-clock to several minutes. Day-trading needs decisions in
seconds, so we cap debate at ``cfg.debate_rounds`` (default 1) and
keep each turn under ~200 words via the prompt.

The Research Manager additionally receives the deterministic level
pool and quantitative baseline signal so its output is a *selection*
across pre-computed structures, not free-form price invention.
"""

from __future__ import annotations

from typing import Optional

from golddaytrading.agents import prompts
from golddaytrading.config import GDTConfig
from golddaytrading.llm.client import LLMClient
from golddaytrading.signals.envelope import (
    ResearchEnvelope,
    parse_envelope,
    render_envelope_block,
)
from golddaytrading.signals.levels import LevelPool


def _analyst_pack(ctx: dict) -> str:
    """Stitch all analyst reports into a single context block."""
    parts = [
        "## Technical analyst", ctx.get("technical_report", "_skipped_"),
        "## Session strategist", ctx.get("session_report", "_skipped_"),
        "## Macro pulse", ctx.get("macro_report", "_skipped_"),
        "## News & catalysts", ctx.get("news_report", "_skipped_"),
        "## Sentiment", ctx.get("sentiment_report", "_skipped_"),
    ]
    quant_block = ctx.get("quant_signal_block")
    if quant_block:
        parts.extend(["## Quant baseline", quant_block])
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
    """Synthesise the debate into a single recommendation.

    The RM is constrained to **select** a setup from the
    deterministic level pool (or return FLAT) by emitting a JSON
    envelope at the start of its reply. The pipeline parses that
    envelope, resolves it against the pool, and writes the
    structured decision back onto ``ctx`` for the Risk Manager and
    Day Trader to consume — bypassing the legacy free-form text
    parsing entirely on the happy path.

    When no LLM key is configured (``offline`` provider) the LLM
    only returns a heuristic stub, so we synthesise the envelope
    deterministically from the level-pool's top-ranked idea instead.
    """
    debate = ctx.get("debate", {})
    pool: Optional[LevelPool] = ctx.get("level_pool")
    pool_block = ctx.get("level_pool_block", "_(no level pool computed)_")

    user = (
        _analyst_pack(ctx)
        + "\n\n## Debate transcript\n"
        + debate.get("history", "_skipped_")
        + "\n\n## Deterministic level pool (pick one setup_id or FLAT)\n"
        + pool_block
        + "\n\nProduce your synthesis now. **Begin with the JSON "
          "envelope** as specified in the system prompt."
    )

    text = llm.complete(
        prompts.localise(prompts.RESEARCH_MANAGER, cfg.output_language),
        user,
        model=cfg.deep_llm,
        max_tokens=900,
    )

    env = parse_envelope(text, pool) if pool is not None else None

    # Offline / hallucinated reply: synthesise an envelope from the
    # top-ranked pool idea so the rest of the pipeline keeps full
    # determinism.
    if env is None and pool is not None and pool.best is not None:
        best = pool.best
        env = ResearchEnvelope(
            bias=best.bias,
            conviction="low",
            selected_setup_id=best.setup_id,
            rationale=(
                "LLM did not return a parseable JSON envelope; "
                "falling back to the highest-ranked pool idea."
            ),
            raw_json={"fallback": True},
            chosen_idea=best,
        )
        text = (
            "_(envelope auto-synthesised from level pool — see "
            "structured-decision block below)_\n\n"
            + text
        )

    ctx["research_envelope"] = env
    ctx["research_chosen_idea"] = env.chosen_idea if env is not None else None
    ctx["research_envelope_block"] = render_envelope_block(env)
    return text
