# Changelog

All notable changes to GoldDayTrading.

## [0.1.0] - 2026-05-23

Initial release. Adapted from
[tmk11/TradingAgents](https://github.com/tmk11/TradingAgents) (which
itself forks `TauricResearch/TradingAgents`) with the entire stack
re-targeted from daily equity analysis to **intraday gold day
trading**.

### Added
- Intraday OHLCV fetcher (`dataflows/intraday_data.py`) that maps
  every requested timeframe (1m / 5m / 15m / 1h / 4h) to the
  appropriate yfinance `period`, plus a 4h resampler since yfinance
  doesn't expose 4h directly.
- Day-trading indicator battery (`dataflows/indicators.py`):
  EMA20/50/200 stack, RSI(14), MACD, ATR(14), session-anchored VWAP,
  60-minute opening range, classic floor-trader pivots from the
  previous day. Pure pandas/numpy — no ta-lib dependency.
- Trading-session classifier (`sessions.py`) for Tokyo / London /
  London-NY overlap / NY / Late-NY with regime hints.
- Intraday macro-pulse fetcher (`dataflows/macro_pulse.py`) for DXY,
  10Y/30Y Treasury yields, VIX, TIPS, and S&P futures on 1h cadence,
  with a "gold bias" tag so the LLM knows which direction each
  driver pushes gold.
- Economic-calendar fetcher (`dataflows/econ_calendar.py`) using the
  free ForexFactory JSON feed, filtered to high-impact USD/EUR
  prints, with `is_in_blackout` for pre/post-news trade blocking.
- Latest-headlines fetcher (`dataflows/gold_news.py`) for last-12h
  RSS items from Investing.com / Mining.com / Bloomberg.

### Agents
- **Technical Analyst**, **Session Strategist**, **Macro Pulse
  Analyst**, **News & Catalyst Agent**, **Sentiment Agent**.
- Compressed 1-round **Bull / Bear** debate.
- **Research Manager** synthesis (LONG / SHORT / FLAT + levels).
- **Risk Manager** — deterministic guardrails (position sizing,
  R:R check, daily-loss cap, news-blackout) + LLM verdict.
- **Day Trader** — composes the final copy-pastable trade plan
  (bias, entry, stop, TP1/TP2, units, time-in-force, blackouts).

### Tooling
- Provider-agnostic LLM client with OpenAI / Anthropic / Gemini and
  a deterministic `offline` heuristic so the smoke pipeline never
  fails for missing API keys.
- Typer + Rich CLI with `analyze` and `info` subcommands.
- Smoke tests (no network, no LLM) covering the indicator battery,
  session classifier, risk guardrails, and offline LLM fallback.

### Removed (vs upstream)
- LangGraph / LangChain dependency (replaced by a linear pipeline).
- Fundamentals analyst (gold has no earnings or balance sheet).
- Multi-round debate (capped at 1 round; configurable).
- Daily-cadence FRED macro tool (replaced by intraday yfinance pulse).
- Web UI / FastAPI server (out of scope for v0.1).
