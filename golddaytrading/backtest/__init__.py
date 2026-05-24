"""Backtest harness and trade journal for GoldDayTrading.

Sub-modules
-----------

* :mod:`golddaytrading.backtest.outcomes` — pure-function trade
  outcome simulator (entry trigger + stop / TP resolution).
* :mod:`golddaytrading.backtest.replay`   — walk-forward replay
  engine that drives the deterministic pipeline (indicators → quant
  signal → level pool) over historical bars and records outcomes.
* :mod:`golddaytrading.backtest.stats`    — aggregate stats
  (win-rate, expectancy, Sharpe-R, max-drawdown) with group-bys.
* :mod:`golddaytrading.backtest.journal`  — SQLite trade journal
  for **live** runs: log every approved plan, manually record
  outcomes, render rolling stats back into prompts.

The backtest layer deliberately runs only the *deterministic* parts
of the pipeline (no LLM calls, no debate). What it measures is the
quality of the level-pool + ranking foundation. If that foundation
has positive expectancy, the LLM debate adds value on top — and the
journal lets you measure whether it actually does, in production.
"""

from golddaytrading.backtest.outcomes import (  # noqa: F401
    TradeOutcome,
    simulate_outcome,
)
from golddaytrading.backtest.replay import (  # noqa: F401
    BacktestReport,
    run_backtest,
)
from golddaytrading.backtest.stats import (  # noqa: F401
    aggregate_stats,
    group_by,
    render_stats_table,
)
from golddaytrading.backtest.journal import (  # noqa: F401
    TradeJournal,
)
