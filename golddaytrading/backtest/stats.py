"""Aggregate stats for backtest / journal trade outcomes.

The metrics here intentionally avoid annualisation tricks — every
number is in **R-multiples** so it stays comparable across
timeframes and tickers.

Metrics
-------

* ``n``                   trades counted
* ``trigger_rate``        n_triggered / n_planned
* ``win_rate``            wins / n_triggered (a "win" is any
                          ``realised_r > 0``, so partial-expired
                          wins count too)
* ``avg_r``               mean realised R
* ``avg_win_r``           mean R of winners only
* ``avg_loss_r``          mean R of losers only (negative)
* ``expectancy_r``        win_rate * avg_win_r + (1 - win_rate) * avg_loss_r
* ``sharpe_r``            mean(R) / stdev(R) — *unannualised*; a
                          unitless edge ratio that is comparable
                          across setups regardless of bar size.
* ``max_dd_r``            biggest peak-to-trough drop on the
                          cumulative-R equity curve
* ``profit_factor``       sum_wins / |sum_losses|

Group-bys
---------

:func:`group_by` returns a dict ``{key: stats_dict}`` over any
attribute of :class:`TradeOutcome` (``setup_id``, ``session``,
``htf_trend``, ``resolution``). :func:`render_stats_table` renders
either a flat or a grouped block as markdown — used by both the
CLI ``backtest`` command and the journal's prompt-injection block.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from golddaytrading.backtest.outcomes import TradeOutcome


# ---------------------------------------------------------------------------
# Core stats
# ---------------------------------------------------------------------------


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def _max_drawdown(rs: Sequence[float]) -> float:
    """Worst peak-to-trough on the cumulative-R curve (returned as a
    *negative* number; ``0.0`` for an empty / monotonic curve)."""
    if not rs:
        return 0.0
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for r in rs:
        equity += r
        if equity > peak:
            peak = equity
        worst = min(worst, equity - peak)
    return worst


def aggregate_stats(outcomes: Iterable[TradeOutcome]) -> Dict[str, float]:
    """Compute the core metric set for a list of outcomes."""
    outs = list(outcomes)
    n_total = len(outs)
    triggered = [o for o in outs if o.triggered]
    n = len(triggered)

    if n_total == 0:
        return {
            "n": 0, "n_planned": 0, "trigger_rate": 0.0,
            "win_rate": 0.0, "avg_r": 0.0,
            "avg_win_r": 0.0, "avg_loss_r": 0.0,
            "expectancy_r": 0.0, "sharpe_r": 0.0,
            "max_dd_r": 0.0, "profit_factor": 0.0, "total_r": 0.0,
        }

    rs = [o.realised_r for o in triggered]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]

    win_rate = (len(wins) / n) if n else 0.0
    avg_r = _mean(rs)
    avg_win_r = _mean(wins)
    avg_loss_r = _mean(losses)
    expectancy_r = win_rate * avg_win_r + (1.0 - win_rate) * avg_loss_r
    std_r = _stdev(rs)
    sharpe_r = (avg_r / std_r) if std_r > 0 else 0.0
    sum_wins = sum(wins)
    sum_losses_abs = abs(sum(losses))
    profit_factor = (sum_wins / sum_losses_abs) if sum_losses_abs > 0 else (
        float("inf") if sum_wins > 0 else 0.0
    )

    return {
        "n": n,
        "n_planned": n_total,
        "trigger_rate": (n / n_total) if n_total else 0.0,
        "win_rate": win_rate,
        "avg_r": avg_r,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "expectancy_r": expectancy_r,
        "sharpe_r": sharpe_r,
        "max_dd_r": _max_drawdown(rs),
        "profit_factor": profit_factor,
        "total_r": sum(rs),
    }


# ---------------------------------------------------------------------------
# Group-by
# ---------------------------------------------------------------------------


KeyFn = Callable[[TradeOutcome], Optional[str]]


def _key_for(field: str) -> KeyFn:
    def _k(o: TradeOutcome) -> Optional[str]:
        v = getattr(o, field, None)
        return str(v) if v is not None else None
    return _k


def group_by(
    outcomes: Iterable[TradeOutcome],
    field: str,
    *,
    min_n: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Return ``{key: stats}`` over a TradeOutcome attribute.

    Buckets with fewer than ``min_n`` *triggered* trades are dropped
    so the table doesn't get cluttered with single-sample noise.
    """
    keyfn = _key_for(field)
    buckets: Dict[str, List[TradeOutcome]] = {}
    for o in outcomes:
        k = keyfn(o)
        if k is None:
            continue
        buckets.setdefault(k, []).append(o)

    out: Dict[str, Dict[str, float]] = {}
    for k, group in buckets.items():
        stats = aggregate_stats(group)
        if stats["n"] >= min_n:
            out[k] = stats
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt_inf(v: float) -> str:
    if v == float("inf"):
        return "  ∞"
    if v == float("-inf"):
        return " -∞"
    if math.isnan(v):
        return "  -"
    return f"{v:+.2f}"


def _stats_row(label: str, s: Dict[str, float]) -> str:
    return (
        f"| {label} | {int(s['n']):>3d} / {int(s['n_planned']):>3d} "
        f"| {s['trigger_rate']*100:5.1f}% | {s['win_rate']*100:5.1f}% "
        f"| {_fmt_inf(s['avg_r']):>6} | {_fmt_inf(s['avg_win_r']):>6} "
        f"| {_fmt_inf(s['avg_loss_r']):>6} | {_fmt_inf(s['expectancy_r']):>6} "
        f"| {_fmt_inf(s['sharpe_r']):>6} | {_fmt_inf(s['max_dd_r']):>6} "
        f"| {_fmt_inf(s['profit_factor']):>5} |"
    )


_HEADER = (
    "| Group | n trig / planned | trig% | win% | avg R | avg win R | "
    "avg loss R | EV (R) | Sharpe-R | Max DD R | PF |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def render_stats_table(
    summary: Dict[str, float] | Dict[str, Dict[str, float]],
    *,
    title: str = "Backtest stats",
    overall_label: str = "ALL",
) -> str:
    """Render either a single-bucket or a grouped stats dict as
    markdown.

    Auto-detects layout: a flat dict (``aggregate_stats`` output)
    is rendered as one row labelled ``ALL``; a grouped dict
    (``group_by`` output) renders one row per key.
    """
    if not summary:
        return f"### {title}\n_(no data)_\n"

    is_grouped = bool(summary) and isinstance(next(iter(summary.values())), dict)
    rows: list[str] = []

    if is_grouped:
        ranked = sorted(
            summary.items(),  # type: ignore[arg-type]
            key=lambda kv: kv[1].get("expectancy_r", 0.0),
            reverse=True,
        )
        for k, s in ranked:
            rows.append(_stats_row(k, s))
    else:
        rows.append(_stats_row(overall_label, summary))  # type: ignore[arg-type]

    return f"### {title}\n" + _HEADER + "\n".join(rows) + "\n"


def render_full_report(report) -> str:
    """Render the full breakdown of a :class:`BacktestReport`.

    Sections: overall, by setup, by session, by HTF trend, by
    resolution, plus a header line with the run params.
    """
    overall = aggregate_stats(report.outcomes)
    by_setup = group_by(report.outcomes, "setup_id", min_n=3)
    by_session = group_by(report.outcomes, "session", min_n=3)
    by_htf = group_by(report.outcomes, "htf_trend", min_n=3)
    by_resolution = group_by(report.outcomes, "resolution", min_n=1)

    head = (
        f"## Backtest report\n"
        f"- **Ticker:** `{report.ticker or '?'}`  |  "
        f"**TF:** `{report.timeframe or '?'}`  |  "
        f"**Strategy:** `{report.strategy}`\n"
        f"- **Bars:** {report.primary_bars}  |  "
        f"**Warmup:** {report.warmup_bars}  |  "
        f"**Horizon:** {report.horizon_bars}\n"
        f"- **Decisions:** {report.n_decisions}  |  "
        f"**With pool:** {report.n_with_pool}  |  "
        f"**Outcomes recorded:** {len(report.outcomes)}\n"
    )

    sections = [
        head,
        render_stats_table(overall, title="Overall (all outcomes)"),
        render_stats_table(by_setup, title="By setup_id (≥3 trades)"),
        render_stats_table(by_session, title="By session (≥3 trades)"),
        render_stats_table(by_htf, title="By higher-timeframe trend (≥3 trades)"),
        render_stats_table(by_resolution, title="By resolution"),
    ]
    return "\n".join(sections)
