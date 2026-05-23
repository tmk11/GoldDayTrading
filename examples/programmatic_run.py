"""Programmatic GoldDayTrading run.

Shows how to construct a config, run the pipeline once, and inspect
each artefact (analyst reports, debate, risk guardrails, final plan)
without going through the CLI.

Run with::

    python examples/programmatic_run.py

If no LLM API key is configured, the pipeline falls back to the
deterministic offline heuristic so the example still produces output.
"""

from __future__ import annotations

import json
from pprint import pformat

from golddaytrading import DayTradingPipeline, load_config


def main() -> None:
    cfg = load_config(
        ticker="GLD",
        primary_timeframe="15m",
        higher_timeframe="1h",
        account_usd=10_000,
        risk_per_trade_pct=0.5,
        debate_rounds=1,
        # Skip the network-heavy news/calendar paths for a quick demo.
        enable_news=False,
        enable_econ_calendar=False,
        debug=True,
    )

    print("=" * 70)
    print("Config:")
    print(pformat(cfg.to_dict()))
    print("=" * 70)

    pipeline = DayTradingPipeline(cfg=cfg)
    ctx = pipeline.run()

    print("\n" + "=" * 70)
    print("Indicator block (raw, fed to the Technical Analyst):")
    print("=" * 70)
    print(ctx["indicator_block"])

    print("\n" + "=" * 70)
    print("Macro-pulse block (raw):")
    print("=" * 70)
    print(ctx["macro_pulse_block"])

    print("\n" + "=" * 70)
    print("Deterministic risk guardrail bundle:")
    print("=" * 70)
    guard = ctx.get("guardrail")
    if guard is not None:
        print(f"  approved:           {guard.approved}")
        print(f"  position size:      {guard.position_size_units}")
        print(f"  $ at risk:          {guard.risk_dollars}")
        print(f"  R:R vs TP1:         {guard.rr_ratio}")
        print(f"  hard block reason:  {guard.hard_block}")

    print("\n" + "=" * 70)
    print("Final plan:")
    print("=" * 70)
    print(ctx["final_plan"])

    print(f"\nWall clock: {ctx['wall_clock_sec']}s | "
          f"provider: {ctx['llm_provider']} | "
          f"deep model: {ctx['llm_model_deep']}")


if __name__ == "__main__":
    main()
