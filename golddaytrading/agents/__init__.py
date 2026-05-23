"""Agents (analyst / debate / risk / trader) for GoldDayTrading.

Each agent is a small callable with the signature::

    def run(context: dict, llm: LLMClient, cfg: GDTConfig) -> str

It receives the shared ``context`` dict (price data, indicators,
macro pulse, news, prior agent reports), uses the LLM client to
generate its report, and *returns* the report as a markdown string.
The pipeline orchestrator stores the result back on the context.
"""
