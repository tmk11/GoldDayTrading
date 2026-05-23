"""Typer + Rich CLI for GoldDayTrading.

Two subcommands:

* ``analyze``  — run the day-trading pipeline once and print the plan.
* ``info``     — print the current trading session, macro pulse, and
                 calendar without invoking the LLM.

The CLI mirrors the upstream ``tradingagents`` UX (banner, progress
hints) but targets a single decision rather than a multi-page report.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from golddaytrading.config import GOLD_TICKERS, load_config
from golddaytrading.dataflows.econ_calendar import (
    calendar_block, fetch_upcoming_events,
)
from golddaytrading.dataflows.macro_pulse import (
    fetch_macro_pulse, macro_pulse_block,
)
from golddaytrading.graph.pipeline import DayTradingPipeline
from golddaytrading.sessions import classify_session, session_summary_block

app = typer.Typer(
    name="golddaytrading",
    help="Multi-agent LLM framework for intraday gold (XAU/USD) day trading.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _banner() -> None:
    console.print(Panel.fit(
        "[bold yellow]GoldDayTrading[/bold yellow]\n"
        "[dim]Multi-agent intraday system for the gold complex "
        "(XAU/USD · GC=F · GLD · GDX)[/dim]",
        border_style="yellow",
    ))


@app.command()
def analyze(
    ticker: Optional[str] = typer.Argument(
        None, help="Gold-complex ticker (defaults to GDT_DEFAULT_TICKER / XAUUSD=X).",
    ),
    timeframe: str = typer.Option(
        None, "--tf", "-t", help="Primary timeframe (1m / 5m / 15m / 1h).",
    ),
    provider: str = typer.Option(
        None, "--provider", "-p",
        help="LLM provider (openai / anthropic / gemini / offline).",
    ),
    deep_llm: str = typer.Option(
        None, "--deep-llm", help="Deep-thinking model id (default: gpt-4o).",
    ),
    quick_llm: str = typer.Option(
        None, "--quick-llm", help="Quick-thinking model id (default: gpt-4o-mini).",
    ),
    account_usd: float = typer.Option(
        None, "--account", help="Account size in USD for risk sizing.",
    ),
    risk_pct: float = typer.Option(
        None, "--risk-pct", help="Max risk per trade as %% of account.",
    ),
    language: str = typer.Option(
        None, "--lang", "-l",
        help="Output language (English / Vietnamese / ...).",
    ),
    debate_rounds: int = typer.Option(
        None, "--debate-rounds", help="Bull/Bear debate rounds (1 is enough).",
    ),
    no_news: bool = typer.Option(
        False, "--no-news", help="Skip the news-catalyst stage.",
    ),
    no_sentiment: bool = typer.Option(
        False, "--no-sentiment", help="Skip the sentiment stage.",
    ),
    no_calendar: bool = typer.Option(
        False, "--no-calendar", help="Skip the economic-calendar stage.",
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Print stage timings.",
    ),
) -> None:
    """Run the gold day-trading pipeline once and print the plan."""
    _banner()

    overrides = {}
    if timeframe:     overrides["primary_timeframe"] = timeframe
    if provider:      overrides["llm_provider"] = provider
    if deep_llm:      overrides["deep_llm"] = deep_llm
    if quick_llm:     overrides["quick_llm"] = quick_llm
    if account_usd:   overrides["account_usd"] = account_usd
    if risk_pct:      overrides["risk_per_trade_pct"] = risk_pct
    if language:      overrides["output_language"] = language
    if debate_rounds: overrides["debate_rounds"] = debate_rounds
    if no_news:       overrides["enable_news"] = False
    if no_sentiment:  overrides["enable_sentiment"] = False
    if no_calendar:   overrides["enable_econ_calendar"] = False
    if debug:         overrides["debug"] = True

    cfg = load_config(**overrides)
    if ticker:
        cfg.ticker = ticker
    if cfg.ticker not in GOLD_TICKERS:
        console.print(
            f"[yellow]Warning:[/yellow] `{cfg.ticker}` is outside the "
            "calibrated gold complex; prompts and risk rules are tuned "
            "for: " + ", ".join(GOLD_TICKERS)
        )

    if cfg.llm_provider == "offline":
        console.print(
            "[dim]Running in [bold]offline[/bold] mode "
            "(no LLM key detected). The pipeline will use a "
            "deterministic heuristic for every agent — useful for "
            "smoke-testing the data layer.[/dim]"
        )

    console.print(
        f"[cyan]→[/cyan] Analysing [bold]{cfg.ticker}[/bold] on "
        f"[bold]{cfg.primary_timeframe}[/bold] (higher tf "
        f"[bold]{cfg.higher_timeframe}[/bold]) using "
        f"[bold]{cfg.llm_provider}[/bold]/[bold]{cfg.deep_llm}[/bold]…"
    )

    pipeline = DayTradingPipeline(cfg=cfg, logger=console.print)
    with console.status("[bold green]Running multi-agent pipeline…", spinner="dots"):
        ctx = pipeline.run(cfg.ticker)

    console.rule("[bold green]Final plan")
    console.print(Markdown(ctx.get("final_plan", "_(empty)_")))

    console.rule("[bold]Risk manager")
    console.print(Markdown(ctx.get("risk_report", "_(empty)_")))

    console.rule("[bold]Research manager")
    console.print(Markdown(ctx.get("research_plan", "_(empty)_")))

    console.print(
        f"\n[dim]Wall-clock: {ctx.get('wall_clock_sec')}s — "
        f"full run saved under {cfg.results_dir}[/dim]"
    )


@app.command()
def info(
    ticker: Optional[str] = typer.Argument(
        None, help="Gold-complex ticker (defaults to XAUUSD=X).",
    ),
) -> None:
    """Print the trading session, macro pulse and calendar — no LLM calls."""
    _banner()
    cfg = load_config()
    if ticker:
        cfg.ticker = ticker

    sess = classify_session()

    table = Table(title="GoldDayTrading status", show_lines=True)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Ticker", cfg.ticker)
    table.add_row("Primary timeframe", cfg.primary_timeframe)
    table.add_row("Higher timeframe", cfg.higher_timeframe)
    table.add_row("Account / risk", f"${cfg.account_usd:,.0f} @ {cfg.risk_per_trade_pct}%")
    table.add_row("Active session", f"{sess.name} ({sess.start_hour_utc:02d}-{sess.end_hour_utc:02d} UTC)")
    table.add_row("LLM provider", cfg.llm_provider)
    console.print(table)

    console.rule("Session")
    console.print(Markdown(session_summary_block()))

    console.rule("Macro pulse")
    console.print(Markdown(macro_pulse_block(fetch_macro_pulse())))

    if cfg.enable_econ_calendar:
        console.rule("Economic calendar (24h)")
        console.print(Markdown(calendar_block(fetch_upcoming_events(24))))


@app.command()
def version() -> None:
    """Print the installed package version."""
    from golddaytrading import __version__
    console.print(f"GoldDayTrading v{__version__}")


def main() -> None:
    """Entry point used by ``python -m golddaytrading``."""
    app()


if __name__ == "__main__":
    main()
