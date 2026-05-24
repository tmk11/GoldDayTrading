"""Intraday macro-pulse fetcher with regime classification.

Day-trading gold means watching a tight set of macro inverses on the
same intraday cadence as the gold tape. The previous implementation
tracked only DXY / ^TNX / ^TYX / ^VIX / TIP / ES=F. That set misses
the *single most important driver of the modern gold cycle*: real
yields. It also misses the gold-silver ratio (a cleaner regime
indicator than absolute price), 2Y yields (Fed-policy expectations),
EURUSD (57 % of DXY), and BTC (alt-store-of-value crossover).

We now pull a richer macro panel and classify the **macro regime**
deterministically into one of:

* ``REAL_YIELD_DRIVE``      — real yields up, USD up, gold under pressure
* ``USD_WEAKNESS``          — DXY down, gold tailwind regardless of yields
* ``RISK_OFF_HAVEN_BID``    — VIX up + ES down + yields down → safe-haven
* ``GROWTH_SCARE``          — yields down + USD down + ES down (ambiguous-bullish)
* ``RANGE_BOUND``           — no clear driver leading
* ``RISK_ON``               — VIX down, ES up, USD steady (mild gold headwind)

The regime tag is fed into the gold-bias mapping so the analyst
agents and the Research Manager get a *contextual* sign on each
driver instead of the simplistic "DXY up = bearish" of v0.1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from golddaytrading.config import MACRO_PULSE_TICKERS
from golddaytrading.dataflows.intraday_data import fetch_intraday_ohlcv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Driver metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroDriver:
    ticker: str
    label: str
    base_bias: str       # gold bias when *this* series is up
    note: str            # one-line interpretation for prompts


# Order is the order rendered in the prompt block.
DRIVERS: tuple[MacroDriver, ...] = (
    MacroDriver("DX-Y.NYB", "DXY (USD index)", "bearish",
                "DXY up = stronger USD = denominator effect, gold weaker."),
    MacroDriver("EURUSD=X", "EURUSD", "bullish",
                "EURUSD ~57% of DXY weight; up implies USD weakness."),
    MacroDriver("^TNX", "10Y nominal yield", "bearish",
                "Higher 10Y yield = higher opportunity cost of holding gold."),
    MacroDriver("^FVX", "5Y nominal yield", "bearish",
                "5Y is the policy-expectation belly — moves with rate path."),
    MacroDriver("^TYX", "30Y nominal yield", "bearish",
                "Long-end yields = duration / inflation premia."),
    MacroDriver("TIP", "TIPS ETF (TIP)", "bullish",
                "TIPS up = real yields down = gold up (real yields are the "
                "dominant gold driver since 2018)."),
    MacroDriver("^VIX", "VIX (equity vol)", "bullish",
                "VIX spike = risk-off = safe-haven gold bid."),
    MacroDriver("ES=F", "S&P 500 futures", "neutral",
                "Risk-on/off context; mild headwind for gold when ES rallies hard."),
    MacroDriver("CL=F", "WTI crude (CL=F)", "bullish",
                "Crude up = inflation impulse = gold tailwind in mid-cycle."),
    MacroDriver("BTC-USD", "Bitcoin", "neutral",
                "Alt store-of-value; sometimes a competitor for risk-haven flows."),
    MacroDriver("SI=F", "Silver futures (SI=F)", "bullish",
                "Silver leads gold in late-stage rallies; divergence = warning."),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _change_pct(series: pd.Series, bars_back: int) -> Optional[float]:
    if len(series) <= bars_back:
        return None
    prev = series.iloc[-(bars_back + 1)]
    last = series.iloc[-1]
    if prev == 0 or pd.isna(prev) or pd.isna(last):
        return None
    return float((last - prev) / prev * 100.0)


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "  -- "
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


def _safe_last(d: dict, key: str) -> Optional[float]:
    info = d.get(key) or {}
    v = info.get("last")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _safe_chg(d: dict, key: str, field: str = "chg_1h") -> Optional[float]:
    info = d.get(key) or {}
    v = info.get(field)
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


def fetch_macro_pulse() -> Dict[str, dict]:
    """Pull recent 1h candles for each macro driver and summarise.

    Returns a dict keyed by ticker; each value is
    ``{"last", "ts", "chg_1h", "chg_4h", "chg_1d", "bias"}``.
    Failed tickers map to a sentinel dict with ``"last": None``.
    """
    bias_lookup = {d.ticker: d.base_bias for d in DRIVERS}
    out: Dict[str, dict] = {}

    # Iterate the union of legacy MACRO_PULSE_TICKERS (for back-compat
    # with anything that still introspects them) and the richer
    # DRIVERS list.
    target_tickers: list[str] = []
    seen: set[str] = set()
    for tk in (*MACRO_PULSE_TICKERS, *(d.ticker for d in DRIVERS)):
        if tk not in seen:
            seen.add(tk)
            target_tickers.append(tk)

    for tk in target_tickers:
        df = fetch_intraday_ohlcv(tk, timeframe="60m", bars=72)
        if df is None or df.empty:
            out[tk] = {
                "last": None, "ts": None,
                "chg_1h": None, "chg_4h": None, "chg_1d": None,
                "bias": bias_lookup.get(tk, "neutral"),
            }
            continue
        close = df["Close"]
        out[tk] = {
            "last": float(close.iloc[-1]),
            "ts": df.index[-1].strftime("%Y-%m-%d %H:%M UTC"),
            "chg_1h": _change_pct(close, 1),
            "chg_4h": _change_pct(close, 4),
            "chg_1d": _change_pct(close, 24),
            "bias": bias_lookup.get(tk, "neutral"),
        }

    # Derived series — gold-silver ratio + real-yield proxy.
    out["__derived__"] = _derive_extras(out)
    # Regime classification — attached so the quant baseline and any
    # downstream consumer can read the same tag the rendering uses.
    regime = classify_regime(out)
    out["__regime__"] = regime.name
    out["__regime_bias__"] = regime.gold_bias
    out["__regime_description__"] = regime.description
    return out


def _derive_extras(pulse: Dict[str, dict]) -> dict:
    """Compute derived intraday series the LLM cannot infer cheaply."""
    derived: dict = {}

    # Gold-silver ratio is a leading regime indicator. A rising ratio
    # (gold outperforming silver) historically correlates with stress
    # / late-stage cycle; falling ratio = "risk-on" reflation.
    # We can't compute it from yfinance silver alone — we'd need spot
    # gold too. For now we just expose silver moves and let the LLM
    # cross-reference against the ticker's price block.
    si_last = _safe_last(pulse, "SI=F")
    derived["silver_last"] = si_last

    # Real-yield proxy: TIP ETF inverse. TIPS up => real yield down.
    # We expose the 1h change with the sign flipped so the analyst
    # can read it directly.
    tip_chg = _safe_chg(pulse, "TIP")
    derived["real_yield_proxy_chg_1h"] = -tip_chg if tip_chg is not None else None

    # 5Y - 2Y curve slope (we approximate with FVX vs TYX changes —
    # TYX is 30Y but we lack a free 2Y intraday series; FVX-TNX
    # gives the belly-vs-10Y slope).
    fvx_chg = _safe_chg(pulse, "^FVX")
    tnx_chg = _safe_chg(pulse, "^TNX")
    if fvx_chg is not None and tnx_chg is not None:
        derived["belly_vs_10y_chg_1h"] = fvx_chg - tnx_chg
    else:
        derived["belly_vs_10y_chg_1h"] = None

    return derived


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------


REGIMES = (
    "REAL_YIELD_DRIVE",
    "USD_WEAKNESS",
    "RISK_OFF_HAVEN_BID",
    "GROWTH_SCARE",
    "RISK_ON",
    "RANGE_BOUND",
)


@dataclass(frozen=True)
class MacroRegime:
    name: str
    gold_bias: str           # "bullish" | "bearish" | "neutral"
    description: str


def classify_regime(pulse: Dict[str, dict]) -> MacroRegime:
    """Heuristic regime tag from the 1h % changes.

    The thresholds are deliberately wide — we want a *coarse* regime
    label that holds for several hours, not a high-frequency flicker.
    """
    dxy = _safe_chg(pulse, "DX-Y.NYB") or 0.0
    tnx = _safe_chg(pulse, "^TNX") or 0.0
    tip = _safe_chg(pulse, "TIP") or 0.0
    vix_chg = _safe_chg(pulse, "^VIX") or 0.0
    es = _safe_chg(pulse, "ES=F") or 0.0
    vix_level = _safe_last(pulse, "^VIX")

    # Real-yield proxy: rising 10Y nominal AND falling TIPS = real yield up.
    real_yield_up = tnx > 0.10 and tip < -0.05

    if real_yield_up and dxy > 0.05:
        return MacroRegime(
            "REAL_YIELD_DRIVE", "bearish",
            "Real yields and USD both bid — classic gold-bearish regime. "
            "Counter-trend longs need a high R:R or a fresh news catalyst."
        )

    # Pure USD weakness regardless of yields: a fading DXY almost
    # always supports gold intraday.
    if dxy < -0.10 and abs(tnx) < 0.20:
        return MacroRegime(
            "USD_WEAKNESS", "bullish",
            "USD weakness without a yield spike — clean gold tailwind. "
            "Trend-following longs are the higher-EV play."
        )

    # Risk-off panic: VIX up, ES down, yields down (flight to quality).
    if vix_chg > 5.0 and es < -0.30 and tnx < -0.10:
        return MacroRegime(
            "RISK_OFF_HAVEN_BID", "bullish",
            "Equities falling, vol spiking, yields collapsing — safe-haven "
            "gold bid is on. Watch for one-way liquidity moves."
        )

    # Growth scare: ES + USD + yields all softening together.
    if es < -0.30 and tnx < -0.10 and dxy < -0.05:
        return MacroRegime(
            "GROWTH_SCARE", "bullish",
            "Synchronised softness in equities, yields and USD — "
            "early-cycle 'flight to gold' regime."
        )

    # Risk-on: ES bid, VIX low, no yield breakdown.
    if es > 0.20 and vix_chg < -3.0 and (vix_level is None or vix_level < 17):
        return MacroRegime(
            "RISK_ON", "neutral",
            "Risk-on equity tape with vol compressing — mild headwind "
            "for gold; expect chop unless a USD or rates story breaks out."
        )

    return MacroRegime(
        "RANGE_BOUND", "neutral",
        "No single macro driver dominating this hour — favour mean-"
        "reversion setups inside the day's existing range."
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def macro_pulse_block(pulse: Dict[str, dict]) -> str:
    """Render the macro pulse + regime as a markdown block."""
    if not pulse:
        return "_(macro pulse unavailable)_\n"

    rows = ["| Driver | Last | 1h | 4h | 1d | Gold bias if this is up |",
            "|---|---|---|---|---|---|"]
    for d in DRIVERS:
        info = pulse.get(d.ticker)
        if not info:
            continue
        last = f"{info['last']:.2f}" if info.get("last") is not None else "  -- "
        rows.append(
            f"| {d.label} | {last} "
            f"| {_fmt_pct(info.get('chg_1h'))} | {_fmt_pct(info.get('chg_4h'))} "
            f"| {_fmt_pct(info.get('chg_1d'))} | {d.base_bias} |"
        )

    derived = pulse.get("__derived__") or {}
    derived_lines = []
    if derived.get("real_yield_proxy_chg_1h") is not None:
        v = derived["real_yield_proxy_chg_1h"]
        sign = "+" if v >= 0 else ""
        derived_lines.append(
            f"- **Real-yield proxy 1h Δ:** `{sign}{v:.2f}%` "
            f"(inverse of TIP — positive => real yields up => gold headwind)"
        )
    if derived.get("belly_vs_10y_chg_1h") is not None:
        v = derived["belly_vs_10y_chg_1h"]
        sign = "+" if v >= 0 else ""
        derived_lines.append(
            f"- **5Y vs 10Y belly slope (1h Δ):** `{sign}{v:.2f}%` "
            f"(positive => front-end repricing higher faster than long end)"
        )
    derived_block = (
        "\n**Derived:**\n" + "\n".join(derived_lines) + "\n"
    ) if derived_lines else ""

    # Use the regime classification already attached by
    # fetch_macro_pulse so the rendered block agrees with whatever
    # the quant baseline saw.
    regime_name = pulse.get("__regime__")
    if regime_name:
        regime = MacroRegime(
            regime_name,
            pulse.get("__regime_bias__") or "neutral",
            pulse.get("__regime_description__") or "",
        )
    else:
        regime = classify_regime(pulse)
    regime_block = (
        f"\n**Macro regime:** `{regime.name}`  —  "
        f"gold bias **{regime.gold_bias}**.  \n"
        f"_{regime.description}_\n"
    )

    return (
        "### Intraday macro pulse (1h cadence)\n"
        + "\n".join(rows)
        + "\n"
        + derived_block
        + regime_block
        + "\n_Reading rule:_ each row's 'gold bias if this is up' "
        "applies in isolation. The `Macro regime` tag above is the "
        "*combined* read across drivers; **use it to weight the rows**, "
        "do not just sum the individual biases.\n"
    )
