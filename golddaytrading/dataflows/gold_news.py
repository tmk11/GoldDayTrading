"""Latest gold-relevant news headlines (RSS-based, no API key).

Adapted from the upstream Gold-Edition `gold_news.py` but trimmed for
day-trading: we only want the *most recent* 6-12h of headlines (older
news is already priced in for an intraday horizon) and we render a
compact attributed block rather than a long-form report.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
    "golddaytrading/0.1"
)
_TIMEOUT = 8.0


@dataclass(frozen=True)
class NewsFeed:
    label: str
    url: str


# Day-trading-relevant feeds. Each one is RSS so we can parse it with
# the stdlib (no scraping). Order is preserved in the rendered block.
GOLD_NEWS_FEEDS: List[NewsFeed] = [
    NewsFeed(
        "Investing.com Commodities & Futures",
        "https://www.investing.com/rss/news_11.rss",
    ),
    NewsFeed(
        "Investing.com Economy",
        "https://www.investing.com/rss/news_14.rss",
    ),
    NewsFeed(
        "Mining.com (gold tag)",
        "https://www.mining.com/tag/gold/feed/",
    ),
    NewsFeed(
        "Bloomberg Markets",
        "https://feeds.bloomberg.com/markets/news.rss",
    ),
]


def _parse_pubdate(raw: str) -> Optional[datetime]:
    """Parse RFC-822 / ISO-8601 publish dates into UTC datetime."""
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError:
        return None


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _fetch_one_feed(feed: NewsFeed,
                    cutoff: datetime,
                    limit: int) -> Tuple[NewsFeed, List[dict]]:
    """Pull and parse one RSS feed; degrade to empty list on failure."""
    try:
        req = Request(feed.url, headers={"User-Agent": _UA})
        with urlopen(req, timeout=_TIMEOUT) as resp:
            payload = resp.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        logger.warning("News feed fetch failed [%s]: %s", feed.label, exc)
        return feed, []
    except Exception as exc:
        logger.warning("News feed unexpected error [%s]: %s", feed.label, exc)
        return feed, []

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        logger.warning("News feed parse error [%s]: %s", feed.label, exc)
        return feed, []

    items = []
    # RSS items live under <channel><item>...
    for item in root.iter("item"):
        title = _strip_html((item.findtext("title") or "").strip())
        if not title:
            continue
        pubdate = _parse_pubdate(item.findtext("pubDate") or "")
        if pubdate and pubdate < cutoff:
            continue
        items.append({
            "title": title,
            "link": (item.findtext("link") or "").strip(),
            "pubdate": pubdate,
        })
        if len(items) >= limit:
            break
    return feed, items


def fetch_recent_gold_news(
    hours_back: int = 12,
    per_feed_limit: int = 5,
) -> List[Tuple[NewsFeed, List[dict]]]:
    """Pull recent items from each gold-news feed (sequential, fast enough)."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_back)
    return [
        _fetch_one_feed(feed, cutoff, per_feed_limit)
        for feed in GOLD_NEWS_FEEDS
    ]


def gold_news_block(news: List[Tuple[NewsFeed, List[dict]]],
                    hours_back: int = 12) -> str:
    """Render fetched news as a markdown block."""
    sections = [
        f"### Gold-relevant headlines (last {hours_back}h)\n"
    ]
    any_items = False
    for feed, items in news:
        if not items:
            sections.append(f"**{feed.label}** — _(no items / feed unreachable)_\n")
            continue
        any_items = True
        sections.append(f"**{feed.label}**")
        for it in items:
            ts = it["pubdate"].strftime("%m-%d %H:%M") if it["pubdate"] else "?"
            sections.append(f"- [{ts}] {it['title']}")
        sections.append("")
    if not any_items:
        sections.append(
            "_(All news feeds unreachable; analyst will rely on macro "
            "pulse and price action only.)_\n"
        )
    return "\n".join(sections) + "\n"
