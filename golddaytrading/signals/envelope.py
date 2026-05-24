"""Parse the structured JSON envelope emitted by the Research Manager.

The Research Manager prompt instructs the model to begin its reply
with a fenced JSON block listing bias / conviction / selected
``setup_id`` / rationale. We extract that block and *cross-reference*
the chosen ``setup_id`` against the deterministic level pool, so the
prices that flow downstream to the Risk Manager are guaranteed to
match the pool — not numbers the LLM happened to type.

Why the JSON envelope rather than free-form parsing?

LLM JSON output is now mature enough across providers (OpenAI,
Anthropic, Gemini) to rely on at the prompt level. Even when a model
slips into prose around the JSON, regex extraction of the *first*
balanced ``{...}`` block inside ```json fences works reliably. This
removes the regex zoo from the legacy text parser and gives us an
auditable contract between the RM and the rest of the pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from golddaytrading.signals.levels import LevelPool, TradeIdea


# Match a fenced ```json ... ``` block (most common) or the first
# top-level {...} that looks like an envelope.
_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)
_BARE_JSON_RE = re.compile(r"(\{[^{}]*\"bias\"[^{}]*\})", re.DOTALL)


@dataclass
class ResearchEnvelope:
    """Structured slice of the Research Manager output."""

    bias: str                         # LONG | SHORT | FLAT
    conviction: str                   # low | medium | high
    selected_setup_id: Optional[str]  # match against level pool
    rationale: str
    raw_json: dict
    chosen_idea: Optional[TradeIdea] = None

    @property
    def is_flat(self) -> bool:
        return self.bias.upper() == "FLAT" or self.selected_setup_id in (
            None, "", "FLAT", "NONE", "null",
        )


def _extract_json_block(text: str) -> Optional[dict]:
    """Pull the first JSON object out of an LLM reply, robust to prose."""
    if not text:
        return None
    for rx in (_FENCED_JSON_RE, _BARE_JSON_RE):
        m = rx.search(text)
        if not m:
            continue
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
    # Last resort: try to load the entire text in case the model just
    # returned bare JSON.
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None


def _normalise_bias(value: object) -> str:
    s = str(value or "").strip().upper()
    if s in ("BUY", "LONG"):
        return "LONG"
    if s in ("SELL", "SHORT"):
        return "SHORT"
    return "FLAT"


def _normalise_conviction(value: object) -> str:
    s = str(value or "").strip().lower()
    if s in ("low", "medium", "high"):
        return s
    return "low"


def parse_envelope(text: str, pool: LevelPool) -> Optional[ResearchEnvelope]:
    """Parse the RM reply and resolve ``selected_setup_id`` against ``pool``.

    Returns ``None`` when the JSON block is missing entirely — the
    caller can decide whether to fall back to the legacy regex
    extraction or to a deterministic FLAT.
    """
    payload = _extract_json_block(text)
    if not isinstance(payload, dict):
        return None

    bias = _normalise_bias(payload.get("bias"))
    conviction = _normalise_conviction(payload.get("conviction"))
    rationale = str(payload.get("rationale") or "").strip()
    setup_id = payload.get("selected_setup_id") or payload.get("setup_id")
    if isinstance(setup_id, str):
        setup_id = setup_id.strip().upper()
    else:
        setup_id = None

    chosen: Optional[TradeIdea] = None
    if setup_id and pool:
        for idea in pool.ideas:
            if idea.setup_id.upper() == setup_id:
                chosen = idea
                break

    # Sanity: a non-FLAT envelope without a matching pool idea is
    # likely a hallucination — downgrade to FLAT but preserve the
    # raw fields for audit.
    if bias != "FLAT" and chosen is None:
        bias = "FLAT"

    return ResearchEnvelope(
        bias=bias,
        conviction=conviction,
        selected_setup_id=setup_id if chosen is not None else None,
        rationale=rationale,
        raw_json=payload,
        chosen_idea=chosen,
    )


def render_envelope_block(env: Optional[ResearchEnvelope]) -> str:
    """Render the parsed envelope back into a compact markdown block.

    This is what the Risk Manager and Day Trader actually consume —
    they no longer parse the RM's free-form prose for prices.
    """
    if env is None:
        return (
            "### Research-manager structured decision\n"
            "_No JSON envelope detected in RM reply — pipeline will "
            "fall back to text extraction._\n"
        )
    if env.is_flat or env.chosen_idea is None:
        return (
            "### Research-manager structured decision\n"
            f"- **Bias:** FLAT\n"
            f"- **Conviction:** {env.conviction}\n"
            f"- **Rationale:** {env.rationale or '_(none)_'}\n"
        )
    idea = env.chosen_idea
    return (
        "### Research-manager structured decision\n"
        f"- **Bias:** {env.bias}  |  **Conviction:** {env.conviction}\n"
        f"- **Setup chosen:** `{idea.setup_id}` — {idea.setup_name}\n"
        f"- **Entry:** `{idea.entry:.2f}`  |  **Stop:** `{idea.stop:.2f}`\n"
        f"- **TP1:** `{idea.tp1:.2f}` (R:R `{idea.rr1:.2f}`)  |  "
        f"**TP2:** `{idea.tp2:.2f}` (R:R `{idea.rr2:.2f}`)\n"
        f"- **Rationale:** {env.rationale or idea.rationale}\n"
    )
