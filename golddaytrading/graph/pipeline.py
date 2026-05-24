"""Day-trading pipeline orchestrator.

Why not LangGraph? The upstream framework uses LangGraph for its
multi-tool, multi-round equity workflow. For day-trading the graph
is essentially linear (analysts → debate → manager → risk → trader)
with a single conditional (skip-if-blackout). A plain Python
function is faster to read, debug, and port — and avoids pulling
LangGraph + LangChain into the dependency closure.

Stages:

    1. Fetch:    OHLCV (primary + higher TF), macro pulse, news, calendar.
    2. Analyse:  Technical, Session, Macro, News, Sentiment.
    3. Debate:   Bull vs Bear (1 round by default).
    4. Synthesis: Research Manager picks a side + levels.
    5. Risk:     Deterministic guardrails + LLM verdict.
    6. Plan:     Day Trader composes the final trade plan.

Each stage updates a shared ``ctx`` dict so the pipeline is easy to
unit-test (mock the LLM, hand it a synthetic ctx) and to replay
(every input is JSON-serialisable).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

from golddaytrading.agents import analysts, day_trader, debate, risk_manager
from golddaytrading.config import GDTConfig, load_config
from golddaytrading.dataflows.econ_calendar import (
    calendar_block, fetch_upcoming_events,
)
from golddaytrading.dataflows.gold_news import (
    fetch_recent_gold_news, gold_news_block,
)
from golddaytrading.dataflows.indicators import (
    compute_indicators, indicator_summary_block,
)
from golddaytrading.dataflows.intraday_data import (
    fetch_intraday_ohlcv, latest_price_block, ohlcv_summary_block,
)
from golddaytrading.dataflows.macro_pulse import (
    fetch_macro_pulse, macro_pulse_block,
)
from golddaytrading.llm.client import LLMClient, build_client
from golddaytrading.sessions import classify_session, session_summary_block
from golddaytrading.signals.levels import build_level_pool, level_pool_block
from golddaytrading.signals.quant_baseline import (
    compute_quant_signal, quant_signal_block,
)


def _now_utc_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class DayTradingPipeline:
    """End-to-end gold day-trading analysis pipeline."""

    def __init__(self, cfg: Optional[GDTConfig] = None,
                 llm: Optional[LLMClient] = None,
                 logger=print):
        self.cfg = cfg or load_config()
        self.llm = llm or build_client(self.cfg.llm_provider, self.cfg.deep_llm)
        self.log = logger if self.cfg.debug else (lambda *_: None)

    # ------------------------------------------------------------------
    # data-gathering stage
    # ------------------------------------------------------------------

    def _gather_data(self, ticker: str) -> dict:
        cfg = self.cfg
        self.log(f"[fetch] OHLCV {ticker} @ {cfg.primary_timeframe} & "
                 f"{cfg.higher_timeframe}")
        primary_df = fetch_intraday_ohlcv(
            ticker, cfg.primary_timeframe, cfg.lookback_bars_primary,
        )
        higher_df = fetch_intraday_ohlcv(
            ticker, cfg.higher_timeframe, cfg.lookback_bars_higher,
        )

        primary_ind = (compute_indicators(primary_df, vwap_anchor_hour_utc=cfg.vwap_anchor_hour_utc)
                       if primary_df is not None else {})
        higher_ind = (compute_indicators(higher_df, vwap_anchor_hour_utc=cfg.vwap_anchor_hour_utc)
                      if higher_df is not None else {})

        self.log("[fetch] macro pulse (DXY / yields / VIX / TIP / ES)")
        macro = fetch_macro_pulse()

        if cfg.enable_news:
            self.log("[fetch] gold news headlines")
            news = fetch_recent_gold_news(hours_back=12, per_feed_limit=5)
        else:
            news = []

        if cfg.enable_econ_calendar:
            self.log("[fetch] economic calendar (24h)")
            events = fetch_upcoming_events(hours_ahead=24)
        else:
            events = []

        # Determine higher-timeframe trend label for the level-pool
        # ranker. We use the EMA stack on the higher TF as a simple
        # deterministic regime tag.
        htf_trend: Optional[str] = None
        if higher_ind:
            ema20_h = float(higher_ind["ema20"].iloc[-1])
            ema50_h = float(higher_ind["ema50"].iloc[-1])
            ema200_h = float(higher_ind["ema200"].iloc[-1])
            if ema20_h > ema50_h > ema200_h:
                htf_trend = "up"
            elif ema20_h < ema50_h < ema200_h:
                htf_trend = "down"
            else:
                htf_trend = "chop"

        active_session = classify_session()
        quant_signal = compute_quant_signal(
            primary_df, primary_ind, macro,
            session_name=active_session.name,
        )
        level_pool = build_level_pool(
            primary_df, primary_ind,
            min_rr=cfg.min_rr,
            htf_trend=htf_trend,
            quant_p_up=quant_signal.p_up if quant_signal else None,
        )

        return {
            "ticker": ticker,
            "now_utc": datetime.now(timezone.utc),
            "primary_df": primary_df,
            "higher_df": higher_df,
            "primary_indicators": primary_ind,
            "higher_indicators": higher_ind,
            "macro_pulse": macro,
            "news_raw": news,
            "upcoming_events": events,
            "htf_trend": htf_trend,
            "active_session": active_session,
            "quant_signal": quant_signal,
            "level_pool": level_pool,
            # Pre-rendered prompt blocks (cheaper to pass to multiple agents)
            "session_block": session_summary_block(),
            "price_block": latest_price_block(ticker, primary_df),
            "ohlcv_block": ohlcv_summary_block(
                ticker, cfg.primary_timeframe, primary_df, tail_rows=10,
            ),
            "indicator_block": indicator_summary_block(primary_df, primary_ind),
            "higher_indicator_block":
                "### Higher-timeframe context (" + cfg.higher_timeframe + ")\n"
                + indicator_summary_block(higher_df, higher_ind),
            "macro_pulse_block": macro_pulse_block(macro),
            "news_block": gold_news_block(news, hours_back=12),
            "calendar_block": calendar_block(events),
            "quant_signal_block": quant_signal_block(quant_signal),
            "level_pool_block": level_pool_block(level_pool, ticker=ticker),
        }

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    def run(self, ticker: Optional[str] = None) -> dict:
        cfg = self.cfg
        ticker = ticker or cfg.ticker
        t0 = time.time()

        ctx = self._gather_data(ticker)

        self.log("[agent] technical analyst")
        ctx["technical_report"] = analysts.technical_analyst(ctx, self.llm, cfg)

        self.log("[agent] session strategist")
        ctx["session_report"] = analysts.session_strategist(ctx, self.llm, cfg)

        self.log("[agent] macro pulse analyst")
        ctx["macro_report"] = analysts.macro_pulse_analyst(ctx, self.llm, cfg)

        if cfg.enable_news:
            self.log("[agent] news/catalyst")
            ctx["news_report"] = analysts.news_catalyst_agent(ctx, self.llm, cfg)
        else:
            ctx["news_report"] = "_news/catalyst agent disabled by config_"

        if cfg.enable_sentiment:
            self.log("[agent] sentiment")
            ctx["sentiment_report"] = analysts.sentiment_agent(ctx, self.llm, cfg)
        else:
            ctx["sentiment_report"] = "_sentiment agent disabled by config_"

        self.log("[agent] bull/bear debate")
        ctx["debate"] = debate.bull_bear_debate(ctx, self.llm, cfg)

        self.log("[agent] research manager (synthesis)")
        ctx["research_plan"] = debate.research_manager(ctx, self.llm, cfg)

        self.log("[agent] risk manager (guardrails + LLM verdict)")
        ctx["risk_report"] = risk_manager.risk_manager(ctx, self.llm, cfg)

        self.log("[agent] day trader (final plan)")
        ctx["final_plan"] = day_trader.day_trader(ctx, self.llm, cfg)

        ctx["wall_clock_sec"] = round(time.time() - t0, 2)
        ctx["llm_provider"] = cfg.llm_provider
        ctx["llm_model_deep"] = cfg.deep_llm
        ctx["llm_model_quick"] = cfg.quick_llm

        # Persist the run.
        self._save(ctx)
        return ctx

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def _save(self, ctx: dict) -> str:
        """Write the run as a single markdown file under results_dir."""
        os.makedirs(self.cfg.results_dir, exist_ok=True)
        path = os.path.join(
            self.cfg.results_dir,
            f"{ctx['ticker'].replace('=', '_').replace('^', '')}_{_now_utc_str()}.md",
        )
        body = _render_run_md(ctx, self.cfg)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(body)
        except OSError:
            return ""
        return path


def _render_run_md(ctx: dict, cfg: GDTConfig) -> str:
    """Render the full run as a single markdown document."""
    sections = [
        f"# GoldDayTrading run — {ctx['ticker']}",
        f"_Generated: {ctx['now_utc'].strftime('%Y-%m-%d %H:%M UTC')}_  |  "
        f"_Provider: {ctx.get('llm_provider')}_  |  "
        f"_Wall clock: {ctx.get('wall_clock_sec')}s_\n",
        "## Final plan", ctx.get("final_plan", "_skipped_"),
        "## Risk-manager report", ctx.get("risk_report", "_skipped_"),
        "## Research-manager structured decision",
        ctx.get("research_envelope_block", "_skipped_"),
        "## Research-manager synthesis", ctx.get("research_plan", "_skipped_"),
        "## Bull / Bear debate", ctx.get("debate", {}).get("history", "_skipped_"),
        "## Quant baseline signal", ctx.get("quant_signal_block", "_skipped_"),
        "## Deterministic level pool", ctx.get("level_pool_block", "_skipped_"),
        "## Technical analyst", ctx.get("technical_report", "_skipped_"),
        "## Session strategist", ctx.get("session_report", "_skipped_"),
        "## Macro pulse analyst", ctx.get("macro_report", "_skipped_"),
        "## News / catalyst", ctx.get("news_report", "_skipped_"),
        "## Sentiment", ctx.get("sentiment_report", "_skipped_"),
        "## Raw context blocks",
        ctx.get("session_block", ""),
        ctx.get("price_block", ""),
        ctx.get("indicator_block", ""),
        ctx.get("higher_indicator_block", ""),
        ctx.get("macro_pulse_block", ""),
        ctx.get("news_block", ""),
        ctx.get("calendar_block", ""),
    ]
    return "\n\n".join(sections)
