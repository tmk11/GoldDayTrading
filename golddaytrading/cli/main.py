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


# ---------------------------------------------------------------------------
# Backtest command
# ---------------------------------------------------------------------------


@app.command()
def backtest(
    ticker: Optional[str] = typer.Argument(
        None,
        help="Gold-complex ticker (defaults to GDT_DEFAULT_TICKER / XAUUSD=X).",
    ),
    timeframe: str = typer.Option(
        "15m", "--tf", "-t", help="Primary timeframe (1m / 5m / 15m / 1h).",
    ),
    bars: int = typer.Option(
        2000, "--bars", help="How many primary-TF bars to fetch.",
    ),
    warmup: int = typer.Option(
        200, "--warmup", help="Bars consumed before the first decision.",
    ),
    horizon: int = typer.Option(
        32, "--horizon",
        help="Bars to walk forward when resolving each idea.",
    ),
    step: int = typer.Option(
        1, "--step", help="Stride between decision bars.",
    ),
    strategy: str = typer.Option(
        "best_idea", "--strategy",
        help="best_idea | all_ideas | p_up_aligned",
    ),
    min_rr: float = typer.Option(
        None, "--min-rr",
        help="Minimum R:R for a setup to enter the pool. "
             "Defaults to GDT_MIN_RR (1.5).",
    ),
    no_htf: bool = typer.Option(
        False, "--no-htf",
        help="Disable HTF resampling (level pool runs without HTF tie-breaker).",
    ),
) -> None:
    """Walk-forward backtest of the deterministic level-pool layer."""
    _banner()
    from golddaytrading.backtest.replay import run_backtest
    from golddaytrading.backtest.stats import render_full_report
    from golddaytrading.dataflows.intraday_data import fetch_intraday_ohlcv

    cfg = load_config()
    if ticker:
        cfg.ticker = ticker
    if min_rr is not None:
        cfg.min_rr = float(min_rr)

    console.print(
        f"[cyan]→[/cyan] Fetching [bold]{cfg.ticker}[/bold] {timeframe} bars "
        f"(target ≥ {bars}) for backtest…"
    )
    df = fetch_intraday_ohlcv(cfg.ticker, timeframe, bars)
    if df is None or df.empty:
        console.print(
            "[red]Failed to fetch OHLCV.[/red] yfinance was unreachable or "
            "the ticker / timeframe combo is unavailable."
        )
        raise typer.Exit(code=1)

    console.print(
        f"[dim]Loaded {len(df)} bars from "
        f"{df.index[0]:%Y-%m-%d %H:%M} to {df.index[-1]:%Y-%m-%d %H:%M} UTC."
        f"[/dim]"
    )

    htf_factor: Optional[int] = None if no_htf else 4
    with console.status("[bold green]Running walk-forward backtest…",
                        spinner="dots"):
        report = run_backtest(
            df, cfg,
            warmup_bars=warmup,
            step_bars=step,
            horizon_bars=horizon,
            strategy=strategy,
            htf_resample_factor=htf_factor,
            ticker=cfg.ticker,
            timeframe=timeframe,
        )

    console.rule("[bold green]Backtest report")
    console.print(Markdown(render_full_report(report)))


# ---------------------------------------------------------------------------
# Journal command (sub-app)
# ---------------------------------------------------------------------------


journal_app = typer.Typer(
    name="journal",
    help="Inspect / update the SQLite trade journal.",
    no_args_is_help=True,
    add_completion=False,
)


@journal_app.command("stats")
def journal_stats(
    days: int = typer.Option(30, "--days",
                             help="Rolling window in days."),
    ticker: Optional[str] = typer.Option(None, "--ticker"),
) -> None:
    """Show rolling per-setup expectancy from the journal."""
    from golddaytrading.backtest.journal import TradeJournal

    cfg = load_config()
    j = TradeJournal(cfg.journal_db_path)
    block = j.stats_block(days_back=days, ticker=ticker)
    console.print(Markdown(block))


@journal_app.command("list")
def journal_list(
    days: Optional[int] = typer.Option(
        None, "--days", help="Filter to plans logged in the last N days."
    ),
    ticker: Optional[str] = typer.Option(None, "--ticker"),
    limit: int = typer.Option(20, "--limit"),
) -> None:
    """List recent plans (with their current outcome, if any)."""
    from golddaytrading.backtest.journal import TradeJournal

    cfg = load_config()
    j = TradeJournal(cfg.journal_db_path)
    plans = j.list_plans(days_back=days, ticker=ticker, limit=limit)
    if not plans:
        console.print("[dim]No plans logged.[/dim]")
        return

    table = Table(
        title=f"Trade journal — last {len(plans)} plan(s)",
        show_lines=False,
    )
    table.add_column("id", style="bold")
    table.add_column("when (UTC)")
    table.add_column("ticker")
    table.add_column("setup")
    table.add_column("bias")
    table.add_column("entry")
    table.add_column("stop")
    table.add_column("approved")
    table.add_column("outcome")
    for p in plans:
        outs = j.list_outcomes_for_plan(int(p["id"]))
        last_outcome = outs[-1] if outs else {}
        table.add_row(
            str(p["id"]),
            (p["created_at"] or "")[:16],
            p["ticker"] or "?",
            p["setup_id"] or "-",
            p["bias"] or "-",
            f"{p['entry']:.2f}" if p["entry"] is not None else "-",
            f"{p['stop']:.2f}" if p["stop"] is not None else "-",
            "yes" if p["approved"] else "no",
            (
                f"{last_outcome.get('resolution')} "
                f"({last_outcome.get('realised_r'):+.2f}R)"
                if last_outcome and last_outcome.get("realised_r") is not None
                else (last_outcome.get("resolution") or "-")
            ),
        )
    console.print(table)


@journal_app.command("record")
def journal_record(
    plan_id: int = typer.Argument(..., help="Plan id to record an outcome for."),
    resolution: str = typer.Option(
        ..., "--resolution", "-r",
        help="tp1 | tp2 | stop | expired | never_triggered | manual",
    ),
    realised_r: Optional[float] = typer.Option(
        None, "--realised-r", help="Realised R-multiple (signed).",
    ),
    exit_price: Optional[float] = typer.Option(
        None, "--exit-price",
    ),
    notes: str = typer.Option("", "--notes"),
) -> None:
    """Record a trade outcome against a previously-logged plan."""
    from golddaytrading.backtest.journal import (
        ACCEPTED_RESOLUTIONS,
        TradeJournal,
    )

    if resolution not in ACCEPTED_RESOLUTIONS:
        console.print(
            f"[red]Invalid resolution[/red]: must be one of "
            f"{sorted(ACCEPTED_RESOLUTIONS)}"
        )
        raise typer.Exit(code=1)

    cfg = load_config()
    j = TradeJournal(cfg.journal_db_path)
    outcome_id = j.record_outcome(
        plan_id,
        resolution=resolution,
        realised_r=realised_r,
        exit_price=exit_price,
        notes=notes,
    )
    console.print(
        f"[green]Recorded outcome #{outcome_id} for plan #{plan_id}[/green]"
    )


@journal_app.command("path")
def journal_path() -> None:
    """Print the resolved on-disk journal location."""
    cfg = load_config()
    console.print(cfg.journal_db_path)


app.add_typer(journal_app, name="journal")


def main() -> None:
    """Entry point used by ``python -m golddaytrading``."""
    app()


if __name__ == "__main__":
    main()
