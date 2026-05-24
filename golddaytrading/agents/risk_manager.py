"""Intraday risk manager — combines hard rules + LLM judgement.

Day-trading risk control is *partly* deterministic (max risk per
trade, daily loss cap, blackout windows) and *partly* judgement-
based (does the setup actually meet R:R? is the stop logical given
volatility?). We compute the deterministic guardrails in code and
hand the LLM the chosen plan + the guardrail readings to produce a
final APPROVE/REDUCE/SKIP verdict with reasoning.

This belt-and-braces design means even if the LLM disagrees with a
hard rule (e.g. tries to APPROVE a trade inside a CPI blackout) the
caller can still inspect the deterministic ``hard_block`` field and
override.

Two extraction paths are supported:

1. **Structured (preferred):** when the pipeline parses the Research
   Manager's JSON envelope and resolves it against the level pool,
   it passes the resulting :class:`TradeIdea` directly to
   :func:`compute_guardrails`. Numbers are then guaranteed to match
   the deterministic pool — no LLM hallucination can sneak in.
2. **Legacy text (fallback):** if the JSON envelope is missing (or
   the model returned a setup_id we cannot resolve), we fall back to
   regex-extracting Entry / Stop / TP1 / TP2 / bias from the
   markdown body. This keeps the pipeline running in offline mode
   and against weaker / older LLMs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from golddaytrading.agents import prompts
from golddaytrading.config import GDTConfig
from golddaytrading.dataflows.econ_calendar import EconEvent, is_in_blackout
from golddaytrading.llm.client import LLMClient
from golddaytrading.signals.levels import TradeIdea


@dataclass
class RiskGuardrail:
    position_size_units: Optional[float]   # None when SKIP'd
    risk_dollars: Optional[float]
    rr_ratio: Optional[float]
    hard_block: Optional[str]              # set when a deterministic rule says no
    approved: bool


# Pattern fragments to pull entry/stop/TP1/TP2 out of free-form text
# (legacy fallback only — the structured envelope path is preferred).
#
# Accepts: "trigger 2345", "trigger: $2,345.60", "Stop **2345**",
#          "TP1 = 2,360.00", "entry around 2345.6", and so on.
# Numbers can have an optional sign and arbitrary thousands-separator
# commas — both are very common in LLM markdown output.
_NUM_RE = (
    r"[-+]?\$?[\s]*"
    r"([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)"
)


def _extract_level(text: str, label: str) -> Optional[float]:
    """Find the first number that follows ``label`` (case-insensitive).

    Handles patterns like ``Entry trigger: 2345.6``,
    ``- **Stop:** $2,340.00``, and ``TP1 = 2,360``.
    """
    pattern = rf"{label}[^0-9]*{_NUM_RE}"
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def compute_guardrails(
    manager_report: str,
    cfg: GDTConfig,
    upcoming_events: list[EconEvent],
    realised_daily_loss_usd: float = 0.0,
    chosen_idea: Optional[TradeIdea] = None,
) -> RiskGuardrail:
    """Run the deterministic risk checks and return a guardrail bundle.

    When ``chosen_idea`` is provided (the structured / preferred path)
    the levels come straight from the deterministic level pool. When
    it is ``None`` we fall back to regex-parsing ``manager_report``.
    """
    if chosen_idea is not None:
        side = chosen_idea.bias.upper()
        entry = chosen_idea.entry
        stop = chosen_idea.stop
        tp1 = chosen_idea.tp1
        tp2 = chosen_idea.tp2
    else:
        entry = _extract_level(manager_report, "trigger")
        if entry is None:
            entry = _extract_level(manager_report, "entry")
        stop = _extract_level(manager_report, "stop")
        tp1 = _extract_level(manager_report, "tp1")
        tp2 = _extract_level(manager_report, "tp2")

        side_match = re.search(r"\b(LONG|SHORT|FLAT)\b", manager_report,
                               flags=re.IGNORECASE)
        side = side_match.group(1).upper() if side_match else "FLAT"

    if side == "FLAT" or entry is None or stop is None:
        return RiskGuardrail(
            None, None, None,
            "Manager recommended FLAT or did not produce a parseable "
            "entry/stop.",
            approved=False,
        )

    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return RiskGuardrail(None, None, None,
                             "Entry equals stop — invalid setup.",
                             approved=False)

    risk_budget_usd = cfg.account_usd * (cfg.risk_per_trade_pct / 100.0)
    units = risk_budget_usd / risk_per_unit

    rr = None
    if tp1 is not None:
        reward = abs(tp1 - entry)
        rr = reward / risk_per_unit if risk_per_unit > 0 else None

    if rr is not None and rr < cfg.min_rr:
        return RiskGuardrail(units, risk_budget_usd, rr,
                             f"R:R {rr:.2f} below configured minimum "
                             f"{cfg.min_rr:.2f}.",
                             approved=False)

    blackout = is_in_blackout(
        upcoming_events,
        now=datetime.now(timezone.utc),
        pre_min=cfg.blackout_minutes_before_news,
        post_min=cfg.blackout_minutes_after_news,
    )
    if blackout is not None:
        return RiskGuardrail(units, risk_budget_usd, rr,
                             f"Inside blackout for "
                             f"{blackout.title} ({blackout.country}) at "
                             f"{blackout.when_utc:%Y-%m-%d %H:%M UTC}.",
                             approved=False)

    daily_cap = cfg.account_usd * (cfg.daily_loss_limit_pct / 100.0)
    if realised_daily_loss_usd + risk_budget_usd > daily_cap:
        return RiskGuardrail(units, risk_budget_usd, rr,
                             f"Worst-case daily loss "
                             f"(${realised_daily_loss_usd + risk_budget_usd:.0f}) "
                             f"exceeds cap ${daily_cap:.0f}.",
                             approved=False)

    return RiskGuardrail(units, risk_budget_usd, rr, None, approved=True)


def risk_manager(ctx: dict, llm: LLMClient, cfg: GDTConfig) -> str:
    """Compose the risk-management report combining guardrails + LLM."""
    manager_report = ctx.get("research_plan", "")
    events: list[EconEvent] = ctx.get("upcoming_events", []) or []
    chosen_idea: Optional[TradeIdea] = ctx.get("research_chosen_idea")

    guard = compute_guardrails(
        manager_report, cfg, events,
        realised_daily_loss_usd=ctx.get("realised_daily_loss_usd", 0.0),
        chosen_idea=chosen_idea,
    )
    ctx["guardrail"] = guard  # so the Day Trader can inspect it

    # Build the prompt as a list of lines so the conditional "N/A"
    # branches cannot accidentally swallow the surrounding context via
    # operator-precedence (a real bug we had — the trailing `if/else`
    # applied to the entire concatenated f-string sequence, not just
    # the one line, dropping account/risk/blackout context whenever
    # a guardrail field was None).
    units_line = (
        f"- Position size (units): {guard.position_size_units:.2f}"
        if guard.position_size_units is not None
        else "- Position size (units): N/A"
    )
    risk_line = (
        f"- Risk on the trade ($): {guard.risk_dollars:.2f}"
        if guard.risk_dollars is not None
        else "- Risk on the trade ($): N/A"
    )
    rr_line = (
        f"- R:R (vs TP1): {guard.rr_ratio:.2f}"
        if guard.rr_ratio is not None
        else "- R:R (vs TP1): N/A"
    )

    chosen_block = ctx.get("research_envelope_block", "").strip()

    parts: list[str] = [
        "## Research-manager plan",
        manager_report,
    ]
    if chosen_block:
        parts.extend(["", chosen_block])
    parts.extend([
        "",
        "## Account & risk parameters",
        f"- Account size: ${cfg.account_usd:,.0f}",
        (
            f"- Max risk per trade: {cfg.risk_per_trade_pct:.2f}% "
            f"(${cfg.account_usd * cfg.risk_per_trade_pct / 100:.0f})"
        ),
        (
            f"- Daily loss cap: {cfg.daily_loss_limit_pct:.2f}% "
            f"(${cfg.account_usd * cfg.daily_loss_limit_pct / 100:.0f})"
        ),
        f"- Min R:R: {cfg.min_rr:.2f}",
        (
            f"- Blackout: {cfg.blackout_minutes_before_news}m before, "
            f"{cfg.blackout_minutes_after_news}m after High events"
        ),
        "",
        "## Deterministic guardrail outputs",
        units_line,
        risk_line,
        rr_line,
        f"- Hard block: {guard.hard_block or 'none — eligible for APPROVE'}",
        "",
        "## Upcoming events (next 24h)",
        ctx.get("calendar_block", "_(no calendar)_"),
        "",
        "Produce the risk-manager verdict now.",
    ])
    user = "\n".join(parts)

    return llm.complete(
        prompts.localise(prompts.RISK_MANAGER, cfg.output_language),
        user,
        model=cfg.deep_llm,
        max_tokens=900,
    )
