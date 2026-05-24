"""Deterministic quantitative signals for the gold day-trading pipeline.

These modules produce *numeric* outputs (probability of up-move,
expected move in ATR units, ranked level pool) that anchor the
LLM's reasoning. The motivation is simple: large language models
hallucinate exact numbers, but they do paraphrase / synthesise well.
By computing the numbers in code and asking the LLM to *reason about*
them — not invent them — we cut a major class of accuracy errors.

Modules
-------

* :mod:`golddaytrading.signals.quant_baseline` — calibrated logistic
  baseline producing ``P(up)`` and expected move from indicator and
  macro features.
* :mod:`golddaytrading.signals.levels` — deterministic trade-level
  pool generator (entries, stops, targets) that the Research Manager
  selects from rather than inventing.
"""

from golddaytrading.signals.quant_baseline import (  # noqa: F401
    QuantSignal,
    compute_quant_signal,
    quant_signal_block,
)
from golddaytrading.signals.levels import (  # noqa: F401
    LevelPool,
    TradeIdea,
    build_level_pool,
    level_pool_block,
)
from golddaytrading.signals.envelope import (  # noqa: F401
    ResearchEnvelope,
    parse_envelope,
    render_envelope_block,
)
