"""SQLite trade journal for live runs.

Persists every approved plan emitted by the live pipeline so the
trader (and the prompts) can later look at *empirical* edge per
setup, per session, per regime — not just synthetic backtest stats.

Schema
------

Two tables, intentionally minimal:

* ``plans``     — one row per plan emitted by the live pipeline.
                  Captures the structured fields needed for stats
                  group-bys plus a ``raw_json`` blob for audit.
* ``outcomes``  — one row per resolution event. Linked to a plan by
                  ``plan_id``. ``resolution`` is one of ``tp1``,
                  ``tp2``, ``stop``, ``expired``, ``never_triggered``,
                  or ``manual`` (free-form, when the trader records
                  an outcome that doesn't fit the canonical buckets).

Why not pandas-on-disk or jsonl?

SQLite gives us cheap WHERE / GROUP BY / time-window queries with
no extra dependency, and it's the right shape for "rolling N-day
expectancy per setup" — the prompt-injection block. The DB is a
single file; the user can ship it, back it up, or open it in any
SQLite browser.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from golddaytrading.backtest.outcomes import TradeOutcome
from golddaytrading.backtest.stats import aggregate_stats, group_by


_SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    timeframe     TEXT,
    setup_id      TEXT,
    bias          TEXT,
    entry         REAL,
    stop          REAL,
    tp1           REAL,
    tp2           REAL,
    rr1           REAL,
    rr2           REAL,
    p_up          REAL,
    quant_logit   REAL,
    quant_label   TEXT,
    macro_regime  TEXT,
    session       TEXT,
    htf_trend     TEXT,
    approved      INTEGER,
    hard_block    TEXT,
    risk_dollars  REAL,
    position_units REAL,
    raw_json      TEXT
);

CREATE TABLE IF NOT EXISTS outcomes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id       INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    resolution    TEXT,
    realised_r    REAL,
    exit_ts       TEXT,
    exit_price    REAL,
    notes         TEXT,
    FOREIGN KEY (plan_id) REFERENCES plans(id)
);

CREATE INDEX IF NOT EXISTS idx_plans_setup    ON plans(setup_id);
CREATE INDEX IF NOT EXISTS idx_plans_created  ON plans(created_at);
CREATE INDEX IF NOT EXISTS idx_outcomes_plan  ON outcomes(plan_id);
"""


# Resolutions accepted on the outcomes table. ``manual`` is for
# notes-only entries (e.g. "scratched flat after Powell speech").
ACCEPTED_RESOLUTIONS = {
    "tp1", "tp2", "stop", "expired", "never_triggered", "manual",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


class TradeJournal:
    """SQLite-backed trade journal.

    Thread-safety: a fresh ``sqlite3.Connection`` is opened for each
    operation (``check_same_thread=False`` is unnecessary). Concurrent
    readers and a single writer is the assumed access pattern, which
    SQLite handles natively.
    """

    def __init__(self, db_path: str | os.PathLike[str]):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # -- schema -------------------------------------------------------

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    # -- writes -------------------------------------------------------

    def log_plan(self, ctx: dict) -> int:
        """Insert one row into ``plans`` from a pipeline ``ctx`` dict.

        Designed to be called from the pipeline's `_save` step. We
        pull only structured fields; the entire dict is *not* JSON
        because pandas DataFrames are not JSON-serialisable. We
        capture the chosen idea, guardrail, quant signal and regime.
        """
        idea = ctx.get("research_chosen_idea")
        guard = ctx.get("guardrail")
        sig = ctx.get("quant_signal")
        macro_pulse = ctx.get("macro_pulse") or {}
        active_session = ctx.get("active_session")

        # Compose the raw_json for audit — only JSON-safe fields.
        raw_payload = {
            "ticker": ctx.get("ticker"),
            "now_utc": _serialise_dt(ctx.get("now_utc")),
            "research_envelope": _envelope_to_json(ctx.get("research_envelope")),
            "guardrail": _guard_to_json(guard),
            "quant_signal": _quant_to_json(sig),
            "macro_regime": macro_pulse.get("__regime__")
                            if isinstance(macro_pulse, dict) else None,
            "macro_regime_bias": macro_pulse.get("__regime_bias__")
                                 if isinstance(macro_pulse, dict) else None,
            "htf_trend": ctx.get("htf_trend"),
        }

        row = (
            _now_iso(),
            str(ctx.get("ticker") or ""),
            ctx.get("primary_timeframe"),
            getattr(idea, "setup_id", None) if idea else None,
            getattr(idea, "bias", None) if idea else None,
            _safe_float(getattr(idea, "entry", None) if idea else None),
            _safe_float(getattr(idea, "stop", None) if idea else None),
            _safe_float(getattr(idea, "tp1", None) if idea else None),
            _safe_float(getattr(idea, "tp2", None) if idea else None),
            _safe_float(getattr(idea, "rr1", None) if idea else None),
            _safe_float(getattr(idea, "rr2", None) if idea else None),
            _safe_float(getattr(sig, "p_up", None)) if sig else None,
            _safe_float(getattr(sig, "direction_logit", None)) if sig else None,
            getattr(sig, "direction_label", None) if sig else None,
            macro_pulse.get("__regime__")
                if isinstance(macro_pulse, dict) else None,
            getattr(active_session, "name", None) if active_session else None,
            ctx.get("htf_trend"),
            int(bool(getattr(guard, "approved", False))) if guard else 0,
            getattr(guard, "hard_block", None) if guard else None,
            _safe_float(getattr(guard, "risk_dollars", None)) if guard else None,
            _safe_float(getattr(guard, "position_size_units", None))
                if guard else None,
            json.dumps(raw_payload, default=str),
        )

        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO plans (
                    created_at, ticker, timeframe, setup_id, bias,
                    entry, stop, tp1, tp2, rr1, rr2,
                    p_up, quant_logit, quant_label, macro_regime,
                    session, htf_trend, approved, hard_block,
                    risk_dollars, position_units, raw_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                row,
            )
            conn.commit()
            assert cur.lastrowid is not None
            return int(cur.lastrowid)

    def record_outcome(
        self,
        plan_id: int,
        *,
        resolution: str,
        realised_r: Optional[float] = None,
        exit_ts: Optional[str] = None,
        exit_price: Optional[float] = None,
        notes: str = "",
    ) -> int:
        """Insert one row into ``outcomes`` linked to ``plan_id``."""
        if resolution not in ACCEPTED_RESOLUTIONS:
            raise ValueError(
                f"resolution {resolution!r} not in {sorted(ACCEPTED_RESOLUTIONS)}"
            )
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO outcomes (
                    plan_id, created_at, resolution, realised_r,
                    exit_ts, exit_price, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    int(plan_id),
                    _now_iso(),
                    resolution,
                    _safe_float(realised_r) or 0.0,
                    exit_ts,
                    _safe_float(exit_price),
                    notes or "",
                ),
            )
            conn.commit()
            assert cur.lastrowid is not None
            return int(cur.lastrowid)

    # -- reads --------------------------------------------------------

    def list_plans(
        self,
        *,
        days_back: Optional[int] = None,
        limit: int = 100,
        ticker: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM plans"
        params: List[Any] = []
        clauses: List[str] = []
        if days_back is not None:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=int(days_back))
            ).isoformat(timespec="seconds")
            clauses.append("created_at >= ?")
            params.append(cutoff)
        if ticker:
            clauses.append("ticker = ?")
            params.append(ticker)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def list_outcomes_for_plan(self, plan_id: int) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM outcomes WHERE plan_id = ? ORDER BY id ASC",
                (int(plan_id),),
            ).fetchall()
        return [dict(r) for r in rows]

    def fetch_outcomes(
        self,
        *,
        days_back: Optional[int] = None,
        ticker: Optional[str] = None,
        only_resolved: bool = True,
    ) -> List[TradeOutcome]:
        """Return :class:`TradeOutcome` instances reconstructed from
        the latest outcome of each plan.

        ``only_resolved`` skips ``manual``-only logs (which carry
        free-form notes and may not have a clean ``realised_r``).
        """
        sql = """
            SELECT p.*, o.resolution AS o_resolution,
                   o.realised_r AS o_realised_r,
                   o.exit_ts AS o_exit_ts
              FROM plans AS p
              JOIN (
                SELECT plan_id, MAX(id) AS max_oid FROM outcomes
                 GROUP BY plan_id
              ) latest ON latest.plan_id = p.id
              JOIN outcomes AS o ON o.id = latest.max_oid
             WHERE 1=1
        """
        params: List[Any] = []
        if days_back is not None:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=int(days_back))
            ).isoformat(timespec="seconds")
            sql += " AND p.created_at >= ?"
            params.append(cutoff)
        if ticker:
            sql += " AND p.ticker = ?"
            params.append(ticker)
        if only_resolved:
            sql += " AND o.resolution != 'manual'"
        sql += " ORDER BY p.id DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        outs: List[TradeOutcome] = []
        for r in rows:
            res = r["o_resolution"]
            triggered = res not in ("never_triggered",)
            outs.append(TradeOutcome(
                setup_id=r["setup_id"] or "",
                bias=r["bias"] or "",
                entry=_safe_float(r["entry"]) or 0.0,
                stop=_safe_float(r["stop"]) or 0.0,
                tp1=_safe_float(r["tp1"]) or 0.0,
                tp2=_safe_float(r["tp2"]) or 0.0,
                rr1=_safe_float(r["rr1"]) or 0.0,
                rr2=_safe_float(r["rr2"]) or 0.0,
                triggered=triggered,
                resolution=res or "never_triggered",
                realised_r=_safe_float(r["o_realised_r"]) or 0.0,
                decision_ts=_parse_iso(r["created_at"]),
                exit_ts=_parse_iso(r["o_exit_ts"]),
                session=r["session"],
                htf_trend=r["htf_trend"],
                quant_p_up=_safe_float(r["p_up"]),
                macro_regime=r["macro_regime"],
            ))
        return outs

    # -- aggregation / rendering --------------------------------------

    def stats_summary(
        self,
        *,
        days_back: Optional[int] = 30,
        ticker: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Return per-setup stats over the rolling window."""
        outs = self.fetch_outcomes(days_back=days_back, ticker=ticker)
        return group_by(outs, "setup_id", min_n=1)

    def stats_block(
        self,
        *,
        days_back: int = 30,
        ticker: Optional[str] = None,
    ) -> str:
        """Render rolling per-setup stats as a markdown prompt block.

        Designed to be included in the Research Manager's user prompt
        so the LLM can apply Bayesian reasoning ("VWAP_RECLAIM_LONG
        had EV=+0.3R over the last 30 days, lean toward it when
        narrative is mixed").
        """
        from golddaytrading.backtest.stats import render_stats_table

        summary = self.stats_summary(days_back=days_back, ticker=ticker)
        if not summary:
            return (
                "### Trade journal (rolling stats)\n"
                f"_No resolved trades in the last {days_back} days "
                f"{'for ' + ticker if ticker else ''} — journal "
                "stats unavailable._\n"
            )
        return render_stats_table(
            summary,
            title=(
                f"Trade journal — last {days_back}d, by setup_id"
                + (f", {ticker}" if ticker else "")
            ),
        )

    def reset(self) -> None:
        """Drop all rows. Schema is preserved. Used by tests."""
        with self._connect() as conn:
            conn.execute("DELETE FROM outcomes")
            conn.execute("DELETE FROM plans")
            conn.commit()


# ---------------------------------------------------------------------------
# Internal helpers (JSON / dt)
# ---------------------------------------------------------------------------


def _serialise_dt(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat(timespec="seconds")
    return str(v)


def _parse_iso(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    try:
        return datetime.fromisoformat(str(v))
    except ValueError:
        return None


def _envelope_to_json(env: Any) -> Optional[Dict[str, Any]]:
    if env is None:
        return None
    return {
        "bias": getattr(env, "bias", None),
        "conviction": getattr(env, "conviction", None),
        "selected_setup_id": getattr(env, "selected_setup_id", None),
        "rationale": getattr(env, "rationale", None),
    }


def _guard_to_json(g: Any) -> Optional[Dict[str, Any]]:
    if g is None:
        return None
    return {
        "approved": bool(getattr(g, "approved", False)),
        "position_size_units": _safe_float(
            getattr(g, "position_size_units", None)
        ),
        "risk_dollars": _safe_float(getattr(g, "risk_dollars", None)),
        "rr_ratio": _safe_float(getattr(g, "rr_ratio", None)),
        "hard_block": getattr(g, "hard_block", None),
    }


def _quant_to_json(s: Any) -> Optional[Dict[str, Any]]:
    if s is None:
        return None
    return {
        "p_up": _safe_float(getattr(s, "p_up", None)),
        "expected_move_atr": _safe_float(getattr(s, "expected_move_atr", None)),
        "direction_logit": _safe_float(getattr(s, "direction_logit", None)),
        "direction_label": getattr(s, "direction_label", None),
        "confidence": _safe_float(getattr(s, "confidence", None)),
    }
