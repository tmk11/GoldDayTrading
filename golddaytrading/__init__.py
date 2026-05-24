"""GoldDayTrading: a multi-agent LLM framework for intraday gold trading.

This package adapts the multi-agent debate pattern from
`TradingAgents` (Tauric Research) but specialises every layer for
**day-trading the gold complex (XAU/USD, GC=F, MGC=F, GLD)**:

* Intraday OHLCV at 1m/5m/15m/1h (yfinance) instead of daily candles.
* Trading-session classification (Tokyo / London / NY / overlap).
* Day-trading indicators: VWAP, ATR, opening-range, 20/50/200 EMA,
  RSI, MACD, daily/weekly pivot points.
* Macro pulse (DXY, ^TNX real-yield proxy, ^VIX) on hourly cadence.
* Economic-calendar awareness for high-impact USD events.
* Compressed bull/bear debate optimised for fast decision cycles.
* Intraday Risk Manager that enforces R-multiple stops, daily loss
  limits, and prohibits trades through high-impact news windows.
* Final output is a structured trade plan: bias, entry, stop, TP1,
  TP2, time-in-force, session validity, R:R.

The framework is provider-agnostic for LLMs (OpenAI / Anthropic /
Gemini) and degrades gracefully to an `offline` heuristic mode when
no API key is configured, so the smoke pipeline always runs.
"""

from golddaytrading.config import GDTConfig, load_config  # noqa: F401
from golddaytrading.graph.pipeline import DayTradingPipeline  # noqa: F401

__version__ = "0.2.0"
