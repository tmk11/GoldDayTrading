"""Quick-start script for GoldDayTrading.

Reads configuration from environment variables (see ``.env.example``)
or the optional overrides passed on the command line. Runs one pass
of the day-trading pipeline against the default ticker and prints
the final plan.

For a richer, interactive UX prefer the ``golddaytrading`` CLI
entry-point (`python -m golddaytrading analyze ...`).
"""

from __future__ import annotations

from golddaytrading.config import load_config
from golddaytrading.graph.pipeline import DayTradingPipeline


def main() -> None:
    cfg = load_config()
    pipeline = DayTradingPipeline(cfg=cfg)
    ctx = pipeline.run()
    print(ctx.get("final_plan", "_(no plan produced)_"))


if __name__ == "__main__":
    main()
