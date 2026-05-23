"""FX / gold trading-session classification.

Gold trades 23 hours a day across overlapping FX sessions. Day-traders
care about session because:

* **Tokyo (Asia)** — typically thin, range-bound; fade extremes.
* **London open** — first big liquidity injection; trend or breakout.
* **London / NY overlap (13:00-17:00 UTC)** — the meatiest session,
  most volume, most directional moves; preferred by gold day-traders.
* **NY** — US data prints (NFP, CPI, FOMC) hit here; news-driven.
* **Late NY / pre-Tokyo** — illiquid, avoid.

All times are UTC. The classifier is deliberately simple — fixed
windows that ignore DST shifts, since gold is FX-style 23h and the
windows are wide enough that a 1h DST drift does not change the
trading regime materially.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import List


@dataclass(frozen=True)
class SessionWindow:
    name: str
    start_hour_utc: int           # inclusive
    end_hour_utc: int             # exclusive
    description: str


# Order matters: we return the *first* matching session, and overlap
# windows are listed before single-session windows so they take priority.
SESSIONS: List[SessionWindow] = [
    SessionWindow(
        name="LONDON_NY_OVERLAP",
        start_hour_utc=13,
        end_hour_utc=17,
        description=(
            "London/NY overlap: highest liquidity and largest "
            "directional moves of the day. Preferred window for gold "
            "breakouts and trend continuation."
        ),
    ),
    SessionWindow(
        name="TOKYO",
        start_hour_utc=0,
        end_hour_utc=8,
        description=(
            "Tokyo (Asia) session: thin liquidity, range-bound bias. "
            "Mean-reversion strategies tend to outperform breakouts."
        ),
    ),
    SessionWindow(
        name="LONDON",
        start_hour_utc=8,
        end_hour_utc=13,
        description=(
            "London session: first major liquidity injection. London "
            "open breakouts and London-fix flows (10:30 / 15:00 UTC) "
            "frequently set the day's direction for gold."
        ),
    ),
    SessionWindow(
        name="NEW_YORK",
        start_hour_utc=17,
        end_hour_utc=21,
        description=(
            "New York session post-overlap: US data prints (NFP, CPI, "
            "FOMC, retail sales) hit during this window and produce "
            "the largest single-event gold moves."
        ),
    ),
    SessionWindow(
        name="LATE_NY",
        start_hour_utc=21,
        end_hour_utc=24,
        description=(
            "Late NY / pre-Tokyo: thin liquidity, slippage risk, "
            "stop-hunt zone. Day-traders typically flat by NY close."
        ),
    ),
]


def classify_session(now_utc: datetime | None = None) -> SessionWindow:
    """Return the trading session that contains ``now_utc`` (UTC).

    Defaults to the current UTC moment when no datetime is supplied.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    h = now_utc.hour
    for s in SESSIONS:
        if s.start_hour_utc <= h < s.end_hour_utc:
            return s
    # Fallback (should be unreachable since the windows tile [0,24))
    return SESSIONS[-1]


def session_summary_block(now_utc: datetime | None = None) -> str:
    """Render a markdown summary of the current trading session.

    Used by the Session Strategist agent to ground its prompt.
    """
    s = classify_session(now_utc)
    ts = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    next_session = _next_session(s, ts)
    return (
        f"# Trading Session Snapshot\n"
        f"- **Now (UTC):** {ts.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- **Active session:** `{s.name}` "
        f"({s.start_hour_utc:02d}:00–{s.end_hour_utc:02d}:00 UTC)\n"
        f"- **Regime hint:** {s.description}\n"
        f"- **Next session:** `{next_session.name}` "
        f"in {_hours_until(next_session, ts):.1f}h\n"
    )


def _next_session(current: SessionWindow, now: datetime) -> SessionWindow:
    """Return the next single-session window after ``current``."""
    # Skip overlaps when looking for "next session" so the user gets a
    # cleaner forward outlook (otherwise LONDON_NY_OVERLAP overlaps
    # with both LONDON end and NY start and never appears as 'next').
    candidates = [s for s in SESSIONS if s.name != "LONDON_NY_OVERLAP"]
    h = now.hour
    for s in candidates:
        if s.start_hour_utc > h:
            return s
    return candidates[0]  # wrap to next day


def _hours_until(target: SessionWindow, now: datetime) -> float:
    """Hours until ``target.start_hour_utc`` from ``now`` (UTC)."""
    h = now.hour + now.minute / 60.0
    delta = target.start_hour_utc - h
    if delta <= 0:
        delta += 24
    return delta


def is_preferred(session_name: str, preferred: List[str]) -> bool:
    """Return True if ``session_name`` is in the preferred list."""
    return session_name in preferred
