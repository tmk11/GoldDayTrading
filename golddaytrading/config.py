"""Runtime configuration for the GoldDayTrading pipeline.

All fields are populated from environment variables (see ``.env.example``)
with sensible defaults for gold day trading. Construct the config once
via :func:`load_config` and pass it through the pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# Gold-complex tickers GoldDayTrading knows how to route. These are
# the only tickers the framework is calibrated for; passing anything
# else will still work but the prompts and risk rules are gold-specific.
GOLD_TICKERS: tuple[str, ...] = (
    "XAUUSD=X",   # spot gold (preferred for intraday FX-style trading)
    "XAU=X",      # alt spot symbol
    "GC=F",       # COMEX front-month futures
    "MGC=F",      # COMEX micro-gold futures
    "GLD",        # SPDR Gold Shares ETF
    "IAU",        # iShares Gold Trust
    "GDX",        # gold miners ETF (correlated, higher beta)
    "GDXJ",       # junior gold miners ETF
)

# Macro tickers polled by the Macro Pulse Analyst on intraday cadence.
# All available on yfinance; no API key required.
MACRO_PULSE_TICKERS: tuple[str, ...] = (
    "DX-Y.NYB",   # DXY (US dollar index) - direct gold inverse
    "^TNX",       # 10Y nominal Treasury yield - opportunity-cost driver
    "^VIX",       # volatility index - risk-off proxy
    "TIP",        # TIPS ETF - inverse real-yield proxy
    "^TYX",       # 30Y Treasury yield - long-duration check
    "ES=F",       # S&P 500 futures - risk-on/off context
)


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass
class GDTConfig:
    """Configuration bundle for one pipeline run."""

    # ---- LLM ----
    llm_provider: str = "openai"   # openai | anthropic | gemini | offline
    deep_llm: str = "gpt-4o"
    quick_llm: str = "gpt-4o-mini"

    # ---- Ticker / timeframe ----
    ticker: str = "XAUUSD=X"
    primary_timeframe: str = "15m"           # 1m | 5m | 15m | 1h
    higher_timeframe: str = "1h"             # used for trend confirmation
    lookback_bars_primary: int = 200         # bars of primary tf to fetch
    lookback_bars_higher: int = 120          # bars of higher tf to fetch

    # ---- Risk management ----
    account_usd: float = 10_000.0
    risk_per_trade_pct: float = 0.5          # % of account per trade
    daily_loss_limit_pct: float = 2.0        # % of account; flat-out trigger
    min_rr: float = 1.5                      # minimum risk:reward ratio
    atr_stop_multiplier: float = 1.5         # stop = ATR * multiplier
    blackout_minutes_before_news: int = 5    # no entries N min before red event
    blackout_minutes_after_news: int = 15

    # ---- Sessions ----
    # Names match :mod:`golddaytrading.sessions`.
    preferred_sessions: List[str] = field(
        default_factory=lambda: ["LONDON", "LONDON_NY_OVERLAP", "NEW_YORK"]
    )

    # ---- Debate / agents ----
    debate_rounds: int = 1                    # bull/bear rounds (1 is enough intraday)
    enable_econ_calendar: bool = True
    enable_news: bool = True
    enable_sentiment: bool = True

    # ---- Output / runtime ----
    output_language: str = "English"          # "English" | "Vietnamese"
    results_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.expanduser("~"), ".golddaytrading", "runs"
        )
    )
    debug: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def has_api_key(self) -> bool:
        if self.llm_provider == "offline":
            return False
        return bool(os.environ.get(_PROVIDER_ENV[self.llm_provider]))


_PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


def load_config(**overrides) -> GDTConfig:
    """Build a :class:`GDTConfig` from environment variables, then
    apply explicit overrides on top.
    """
    cfg = GDTConfig(
        llm_provider=os.environ.get("GDT_LLM_PROVIDER", "openai").lower(),
        deep_llm=os.environ.get("GDT_DEEP_LLM", "gpt-4o"),
        quick_llm=os.environ.get("GDT_QUICK_LLM", "gpt-4o-mini"),
        ticker=os.environ.get("GDT_DEFAULT_TICKER", "XAUUSD=X"),
        primary_timeframe=os.environ.get("GDT_PRIMARY_TIMEFRAME", "15m"),
        higher_timeframe=os.environ.get("GDT_HIGHER_TIMEFRAME", "1h"),
        account_usd=_env_float("GDT_ACCOUNT_USD", 10_000.0),
        risk_per_trade_pct=_env_float("GDT_RISK_PER_TRADE_PCT", 0.5),
        daily_loss_limit_pct=_env_float("GDT_DAILY_LOSS_LIMIT_PCT", 2.0),
        min_rr=_env_float("GDT_MIN_RR", 1.5),
        atr_stop_multiplier=_env_float("GDT_ATR_STOP_MULT", 1.5),
        debate_rounds=_env_int("GDT_DEBATE_ROUNDS", 1),
        enable_econ_calendar=_env_bool("GDT_ENABLE_ECON_CALENDAR", True),
        enable_news=_env_bool("GDT_ENABLE_NEWS", True),
        enable_sentiment=_env_bool("GDT_ENABLE_SENTIMENT", True),
        output_language=os.environ.get("GDT_OUTPUT_LANGUAGE", "English"),
        debug=_env_bool("GDT_DEBUG", False),
    )

    # When no API key is found and the user did not explicitly pick
    # ``offline``, downgrade to ``offline`` so the pipeline still runs
    # in heuristic mode rather than failing on the first LLM call.
    if cfg.llm_provider != "offline":
        env_var = _PROVIDER_ENV.get(cfg.llm_provider)
        if env_var and not os.environ.get(env_var):
            cfg.llm_provider = "offline"

    for k, v in overrides.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
