"""Lightweight economic-calendar fetcher for high-impact USD events.

Day-trading gold without an economic calendar is gambling — NFP, CPI,
FOMC, retail sales, and PPI prints regularly produce 30-100+ point
moves in XAU/USD within minutes. This module pulls upcoming high-
impact USD events from a free public source and surfaces them so the
Risk Manager can enforce a blackout window around each release.

Implementation notes
--------------------

We avoid the major paid calendar APIs (Trading Economics, Investing
"licensed feed") and the JS-only Investing.com web calendar. Instead
we use ForexFactory's plain JSON endpoint, which has been stable for
years and requires no auth. If the request fails (network, schema
change, rate-limit) the helper degrades to ``[]`` and the pipeline
continues — the Risk Manager just won't have a blackout list.

The output is a list of :class:`EconEvent` records pre-filtered to
high-impact USD (and EUR if relevant for DXY) events in the next
24 hours.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# ForexFactory's public JSON. No auth, browser-style UA.
_FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
    "golddaytrading/0.1"
)
_TIMEOUT = 10.0

# Events that move gold the most. We keep all High-impact USD prints,
# plus a curated whitelist of medium-impact ones that historically
# move XAU/USD (e.g. ADP, Jobless Claims).
_INTERESTING_TITLES = {
    "Non-Farm Employment Change",
    "ADP Non-Farm Employment Change",
    "Unemployment Rate",
    "CPI m/m", "CPI y/y", "Core CPI m/m",
    "PPI m/m", "Core PPI m/m",
    "Retail Sales m/m", "Core Retail Sales m/m",
    "FOMC Statement", "FOMC Press Conference",
    "FOMC Economic Projections", "FOMC Meeting Minutes",
    "Federal Funds Rate",
    "Unemployment Claims",
    "GDP q/q", "Advance GDP q/q",
    "ISM Manufacturing PMI", "ISM Services PMI",
}


@dataclass
class EconEvent:
    title: str
    country: str
    impact: str           # "High" | "Medium" | "Low"
    when_utc: datetime
    forecast: Optional[str]
    previous: Optional[str]

    def minutes_until(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (self.when_utc - now).total_seconds() / 60.0


def _parse_ff_event(raw: dict) -> Optional[EconEvent]:
    """Parse one ForexFactory JSON entry into an EconEvent."""
    try:
        when_str = raw.get("date") or ""
        # FF format: "2026-05-24T13:30:00-04:00"
        when = datetime.fromisoformat(when_str)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        else:
            when = when.astimezone(timezone.utc)
        return EconEvent(
            title=str(raw.get("title", "")).strip(),
            country=str(raw.get("country", "")).strip().upper(),
            impact=str(raw.get("impact", "")).strip().capitalize(),
            when_utc=when,
            forecast=raw.get("forecast") or None,
            previous=raw.get("previous") or None,
        )
    except (ValueError, TypeError):
        return None


def fetch_upcoming_events(hours_ahead: int = 24) -> List[EconEvent]:
    """Return high-impact gold-relevant events in the next ``hours_ahead``.

    Always returns a list (possibly empty). Network and parse errors
    are swallowed and logged so the pipeline keeps running with no
    blackout calendar rather than crashing.
    """
    try:
        req = Request(_FF_URL, headers={"User-Agent": _UA})
        with urlopen(req, timeout=_TIMEOUT) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
        data = json.loads(payload)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("ForexFactory fetch failed: %s", exc)
        return []
    except Exception as exc:
        logger.warning("ForexFactory unexpected error: %s", exc)
        return []

    if not isinstance(data, list):
        return []

    now = datetime.now(timezone.utc)
    horizon = now + timedelta(hours=hours_ahead)
    events: List[EconEvent] = []
    for raw in data:
        ev = _parse_ff_event(raw)
        if ev is None:
            continue
        if ev.when_utc < now or ev.when_utc > horizon:
            continue
        # Filter: USD high-impact, or curated medium for USD/EUR
        if ev.country in ("USD", "EUR") and (
            ev.impact == "High"
            or (ev.impact == "Medium" and ev.title in _INTERESTING_TITLES)
        ):
            events.append(ev)

    events.sort(key=lambda e: e.when_utc)
    return events


def calendar_block(events: List[EconEvent]) -> str:
    """Render the upcoming-event list as a prompt-ready block."""
    if not events:
        return (
            "### Economic calendar (next 24h, USD/EUR high-impact)\n"
            "_No high-impact USD/EUR events scheduled in the next 24h "
            "(or calendar feed unavailable)._\n"
        )
    rows = [
        "| When (UTC) | In | Country | Impact | Event | Fcst | Prev |",
        "|---|---|---|---|---|---|---|",
    ]
    for ev in events:
        mins = ev.minutes_until()
        if mins < 60:
            in_str = f"{mins:.0f} min"
        else:
            in_str = f"{mins / 60:.1f} h"
        rows.append(
            f"| {ev.when_utc.strftime('%m-%d %H:%M')} | {in_str} | "
            f"{ev.country} | {ev.impact} | {ev.title} | "
            f"{ev.forecast or '-'} | {ev.previous or '-'} |"
        )
    return (
        "### Economic calendar (next 24h, USD/EUR high-impact)\n"
        + "\n".join(rows)
        + "\n\nRule: the Risk Manager forbids new entries in a "
        "5-minute window before and 15-minute window after every "
        "**High** impact print listed above.\n"
    )


def is_in_blackout(events: List[EconEvent],
                   now: Optional[datetime] = None,
                   pre_min: int = 5,
                   post_min: int = 15) -> Optional[EconEvent]:
    """Return the offending event if ``now`` is inside any blackout window."""
    now = now or datetime.now(timezone.utc)
    for ev in events:
        if ev.impact != "High":
            continue
        start = ev.when_utc - timedelta(minutes=pre_min)
        end = ev.when_utc + timedelta(minutes=post_min)
        if start <= now <= end:
            return ev
    return None
