# Changelog

All notable changes to GoldDayTrading.

## [0.3.0] - 2026-05-24

P2 — backtest harness and trade journal. The deterministic level-pool
+ quant-baseline foundation now has a way to *measure* its edge and
feed empirical per-setup expectancy back into the LLM prompts.

### Added
- **Walk-forward backtest engine** (`backtest/replay.py`,
  `backtest/outcomes.py`, `backtest/stats.py`). Replays the
  deterministic half of the pipeline (indicators → quant signal →
  level pool) over historical OHLCV, simulates each idea's outcome
  with a conservative same-bar resolution rule (stop wins ties), and
  produces grouped stats (win-rate, expectancy, Sharpe-R, max DD,
  profit factor) by `setup_id`, `session`, `htf_trend`, and
  `resolution`. Three strategies: `best_idea` (top-ranked only,
  matches the live pipeline), `all_ideas` (full pool, more samples),
  and `p_up_aligned` (only ideas whose bias agrees with the quant
  prior). Supports HTF resampling for the level-pool ranker.
- **SQLite trade journal** (`backtest/journal.py`) — auto-logs every
  plan emitted by the live pipeline (chosen idea, guardrail outcome,
  quant signal, regime, session, HTF trend) and lets the trader
  record realised outcomes. The `stats_block` rendering is injected
  into the Research Manager's prompt on subsequent runs so the LLM
  reasons against *empirical* per-setup edge, not just the
  hand-calibrated quant prior.
- **Pipeline auto-logging** — when `cfg.enable_journal` is True
  (default) and a research-chosen idea exists, every run persists a
  row to `~/.golddaytrading/journal.sqlite3` (or
  `$GDT_JOURNAL_DB_PATH`). Failures degrade silently — the journal
  never blocks a run.
- **CLI**: `gdt backtest` and `gdt journal {stats,list,record,path}`.
  `gdt backtest XAUUSD=X --tf 15m --bars 2000 --strategy best_idea`
  fetches via yfinance and prints the full grouped report.
  `gdt journal stats --days 30` prints the markdown block the
  pipeline injects into prompts.
- **New config fields** (with `GDT_*` env-var counterparts):
  `enable_journal`, `inject_journal_stats`, `journal_db_path`,
  `journal_stats_days_back`.
- **Test suite expanded to 74 tests** (`tests/test_p2_backtest.py`):
  outcome simulator (entry trigger, stop priority, same-bar tie
  resolution, expired R math), replay determinism, stats
  aggregation, group-by min_n filter, drawdown sign, journal
  schema lifecycle (log, record, latest-outcome semantics, reset),
  end-to-end auto-log from the offline pipeline.

### Changed
- `RESEARCH_MANAGER` prompt now lists journal stats as a fourth
  input (alongside analyst reports, debate, level pool, quant
  signal) and treats it as a Bayesian prior — high-edge setups
  get a tailwind, but the model is told that recent edge is not
  a guarantee.
- `_render_run_md` includes the trade-journal stats block in the
  per-run markdown for audit.
- The pipeline's `run()` method now opens the journal once at start
  and uses the same handle for both stats injection and the final
  log_plan call.

### Known limitations (P3 territory)
- The backtest passes `macro_pulse={}` — replaying historical macro
  is left to a future P3 ingest (Polygon / Databento). Live runs
  still get the full macro pulse.
- The journal's `stats_block` reconstructs `TradeOutcome` objects
  from DB rows; HTF trend / regime fields persist correctly but
  ATR-based fields are not stored (they were never needed for the
  group-bys we surface).

## [0.2.0] - 2026-05-24

Major accuracy / correctness pass on the day-trading pipeline. The
core change is **the LLM no longer invents trade prices** — entries,
stops, and targets now come from a deterministic level pool, and the
Research Manager's role is to *select* across pre-computed setups.

### Fixed (P0 — bugs that quietly degraded every run)
- **Risk-manager prompt construction** previously dropped the entire
  account / risk / blackout context whenever any guardrail field
  was `None` (a Python operator-precedence bug — `if/else` ternary
  applied to the whole concatenated f-string sequence rather than
  the one line). The prompt is now built from a list of strings,
  with N/A-fallback computed per line.
- **`_extract_level`** now parses prices with thousands separators
  and currency prefixes (`$2,345.60`, `2,345`, `~2345`), which LLMs
  emit constantly in markdown. Previously these silently failed and
  the pipeline produced a SKIP from a parseable plan.
- **Session VWAP** now anchors at 22:00 UTC (the 5pm New York close,
  the FX-desk convention) instead of UTC midnight, and is configurable
  via `GDT_VWAP_ANCHOR_HOUR_UTC`. With `Volume == 0` (the spot-FX
  case) it falls back to a session-anchored typical-price mean rather
  than silently turning into a meaningless 1-volume mean weighted by
  the first bar.

### Added (P1 — accuracy improvements)
- **Quantitative baseline signal** (`signals/quant_baseline.py`) — a
  pure-numpy, calibrated logistic-regression-style scorer producing
  `P(up)`, expected move in ATR units, confidence, and a top-driver
  feature breakdown. Pre-calibrated coefficients encode established
  gold-trading priors (DXY inverse, real-yield drag, RSI extremes
  mean-revert, EMA stack momentum). Output is rendered as a prompt
  block fed to the technical analyst, debate, and Research Manager.
- **Deterministic trade-level pool** (`signals/levels.py`) generates
  up to eight candidate setups (VWAP reclaim/rejection, opening-range
  breakout/breakdown, pivot bounce/rejection, Bollinger mean-reversion
  long/short), each with prices anchored to the indicator snapshot,
  validated for geometry, screened by minimum R:R, and scored using
  the higher-timeframe trend and quant prior.
- **Structured Research Manager output** (`signals/envelope.py`) —
  the RM emits a fenced JSON envelope with `bias`, `conviction`,
  `selected_setup_id`, and `rationale`. The pipeline parses it,
  resolves the selected id against the pool, and downgrades to FLAT
  on hallucinations. The Risk Manager now consumes deterministic
  prices from the chosen `TradeIdea` directly, bypassing free-form
  text parsing on the happy path. Offline / parse-failure runs fall
  back to the highest-ranked pool idea so the pipeline stays
  deterministic end-to-end.
- **Extended indicator battery** — Bollinger Bands(20, 2) with
  bandwidth ("squeeze") detector, Stochastic(14, 3, 3), Camarilla
  pivots, weekly + monthly floor pivots, fractal swing-high/low
  detection, anchored VWAP from the most recent swing high *and*
  swing low. Opening range now anchors on the same NY-close session
  boundary as Session VWAP.
- **Extended macro pulse** — added EURUSD, ^FVX (5Y yield), CL=F
  (crude), BTC-USD, SI=F (silver). Derived series: real-yield proxy
  (inverse of TIP) and 5Y-vs-10Y belly slope.
- **Macro regime classifier** with six regimes
  (`REAL_YIELD_DRIVE`, `USD_WEAKNESS`, `RISK_OFF_HAVEN_BID`,
  `GROWTH_SCARE`, `RISK_ON`, `RANGE_BOUND`); the regime tag drives
  the gold-bias mapping rather than a naive sum of individual driver
  biases. Regime feeds back into the quant baseline as a feature
  with a +0.50 coefficient.
- **Refined prompts** — Macro Pulse Analyst leads with the regime
  tag; Research Manager is required to select from the level pool
  (or return FLAT) and emit JSON; Risk Manager is told to quote the
  guardrail-computed numbers verbatim and never widen a stop to
  fabricate R:R.
- **Test suite** — 48 tests across `test_smoke.py`,
  `test_p0_bugfixes.py` (regression coverage for the three bugs),
  and `test_p1_signals.py` (quant baseline sensitivity, level-pool
  geometry / filtering / ranking, envelope parsing including
  hallucination downgrade, regime classifier across all buckets,
  end-to-end pipeline determinism).

### Changed
- `compute_indicators(df, vwap_anchor_hour_utc=22)` — new optional
  argument, threaded from `GDTConfig.vwap_anchor_hour_utc`.
- `compute_guardrails(..., chosen_idea=None)` — new optional
  argument; when provided, the deterministic levels skip regex
  parsing entirely.
- `MACRO_PULSE_TICKERS` extended with `^FVX`, `EURUSD=X`, `BTC-USD`,
  `CL=F`, `SI=F`.

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
