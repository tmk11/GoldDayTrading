# GoldDayTrading — architecture reference

> Companion to [README.md](../README.md). The README explains *what*
> GoldDayTrading is and *how* to use it. This doc explains *how it
> works internally* so you can extend, replace, or audit any layer.

---

## 1 · Module map

```text
golddaytrading/
├── config.py              # GDTConfig + env-var loader; offline fallback
├── sessions.py            # UTC FX-session classifier (Tokyo / London / NY ...)
├── dataflows/             # Data-fetching layer; never calls LLMs
│   ├── intraday_data.py   # yfinance OHLCV + 4h resampler
│   ├── indicators.py      # EMA / RSI / MACD / ATR / VWAP / OR / pivots
│   ├── macro_pulse.py     # hourly DXY / yields / VIX / TIPS / ES
│   ├── econ_calendar.py   # ForexFactory free JSON + blackout helper
│   └── gold_news.py       # last-12h RSS from Investing.com / Mining.com / Bloomberg
├── llm/
│   └── client.py          # Provider-agnostic chat wrapper (OpenAI/Anthropic/
│                          # Gemini/offline) with deterministic fallback
├── agents/                # LLM agents; each is a callable run(ctx, llm, cfg)
│   ├── prompts.py         # Centralised role prompts (English, localisable)
│   ├── analysts.py        # Technical / Session / Macro / News / Sentiment
│   ├── debate.py          # Bull / Bear + Research Manager synthesis
│   ├── risk_manager.py    # Deterministic guardrails + LLM verdict
│   └── day_trader.py      # Composes the final trade plan
├── graph/
│   └── pipeline.py        # DayTradingPipeline — linear orchestrator
└── cli/
    └── main.py            # Typer + Rich CLI (`gdt analyze` / `gdt info`)
```

---

## 2 · Pipeline stages

```text
                        DayTradingPipeline.run(ticker)
                                     │
       ┌─────────────────────────────┼─────────────────────────────┐
       │                             │                             │
       ▼                             ▼                             ▼
  fetch_intraday_ohlcv         fetch_macro_pulse              fetch_recent_gold_news
  (primary + higher tf)        (DXY / yields / VIX)           (last 12h RSS)

           │                                                       │
           ▼                                                       ▼
  compute_indicators                                        fetch_upcoming_events
  (EMA/RSI/MACD/ATR/VWAP/OR/pivots)                         (24h calendar)

                                     │
                                     ▼
                        ctx = { all blocks rendered }
                                     │
       ┌─────────┬─────────┬─────────┼─────────┬─────────┐
       ▼         ▼         ▼         ▼         ▼
   technical  session   macro    news/      sentiment
   analyst    strategist pulse   catalyst   agent
       └─────────┴─────────┴─────────┴─────────┘
                          │
                          ▼
                    bull / bear  (1 round, ~200 words each)
                          │
                          ▼
                  research_manager   (LONG / SHORT / FLAT + levels)
                          │
                          ▼
        ┌─── compute_guardrails ────┐
        │  position_size            │
        │  R:R vs cfg.min_rr        │
        │  daily-loss-cap budget    │
        │  is_in_blackout(events)   │
        │  hard_block / approved    │
        └────────────┬──────────────┘
                     ▼
                risk_manager     (LLM reads the bundle, writes verdict)
                     │
                     ▼
                 day_trader      (final markdown plan)
```

Each stage writes its output back onto the shared `ctx` dict using a
stable key (`technical_report`, `risk_report`, …). The pipeline is
linear and synchronous — no LangGraph, no async — which keeps the
control flow trivially auditable for an intraday trading system.

---

## 3 · Why a linear pipeline (and not LangGraph)?

The upstream `TradingAgents` framework uses LangGraph because
equity-style decisions can branch on many tools and benefit from
graph-style retries. For day-trading, the workflow is:

1. Always identical sequence (analysts → debate → manager → risk → trader).
2. A single conditional check (blackout window → SKIP) handled by
   `compute_guardrails`, *not* graph routing.
3. Decisions need to land in seconds.

A plain Python function gives you:

- Faster cold-start (no LangGraph imports).
- Trivial unit testing (mock the LLM, hand it a synthetic `ctx`).
- Clearer error surfaces — every stage has a stack trace, no hidden
  graph runtime.

---

## 4 · The offline / heuristic mode

`golddaytrading/llm/client.py` exposes `LLMClient` with a single
`complete(system, user)` method. When no API key is configured, the
factory returns a client whose `complete()` returns a deterministic
heuristic string that:

- Acknowledges the role from the system prompt.
- Echoes back the key indicator / macro lines from the user prompt.
- Suggests a NEUTRAL plan downstream so the Risk Manager refuses
  size on offline runs (you should never trade off the heuristic).

This mode exists so:

1. CI / smoke tests run end-to-end without secrets.
2. New contributors can inspect the *data layer* (which is
   completely independent of the LLM) without paying a penny.
3. The pipeline never crashes on a transient OpenAI outage —
   `LLMClient.complete()` catches provider exceptions and falls
   back to the offline heuristic for that turn only.

---

## 5 · Risk-management contract

The Risk Manager is the only stage that mixes deterministic logic
and LLM judgement, because intraday risk has both:

| Rule                                        | Where it lives                                    |
| ------------------------------------------- | ------------------------------------------------- |
| Position size = (account × risk%) / risk    | `compute_guardrails`                              |
| R:R ≥ `cfg.min_rr`                          | `compute_guardrails`                              |
| Inside a high-impact event blackout         | `econ_calendar.is_in_blackout` + guardrails       |
| Realised + worst-case loss ≤ daily cap      | `compute_guardrails`                              |
| Setup quality / stop placement reasonable   | LLM verdict (Risk Manager prompt)                 |
| Time-in-force vs session end                | LLM verdict                                       |

`compute_guardrails` returns a `RiskGuardrail` dataclass with an
`approved: bool` and a `hard_block: Optional[str]`. The LLM Risk
Manager *sees* this bundle in its prompt and is expected to defer
to a non-`None` `hard_block` — but even if it doesn't, the caller
still has the deterministic `approved` flag to act on.

---

## 6 · Adding a new analyst

1. Write the role prompt in `agents/prompts.py`.
2. Add a function in `agents/analysts.py` that:
   - Pulls the relevant block(s) from `ctx`.
   - Calls `llm.complete(prompts.localise(MY_PROMPT, cfg.output_language), user)`.
   - Returns the assistant text.
3. Register a call in `DayTradingPipeline.run()` between the existing
   analyst block and the debate stage.
4. Update `_render_run_md` in `graph/pipeline.py` so the new section
   shows up in the persisted markdown.

That's the entire surface — there's no graph to register with.

---

## 7 · Adding a new data source

1. Drop a module in `dataflows/` that exposes:
   - `fetch_<thing>(...)` — returns a structured object.
   - `<thing>_block(payload) -> str` — renders a markdown block.
2. Add a `*_block` key to the `ctx` dict in
   `DayTradingPipeline._gather_data`.
3. Reference the block from whichever analyst's user prompt should
   see it (in `agents/analysts.py`).

Failure handling convention: every `fetch_*` returns either valid
data or a sentinel (`None`, `[]`, etc.) and **never** raises. The
matching `*_block` renders a labelled placeholder ("_(feed
unavailable)_") for sentinels so a single dead source never crashes
the run.

---

## 8 · Adding a new LLM provider

`llm/client.py` is intentionally a 200-line file. To wire a new
provider:

1. Add a `_complete_<provider>(client, system, user, model, temp, max_tokens)`
   helper.
2. Add a branch in `build_client(provider)` that lazy-imports the
   SDK, checks for the env var, and returns an `LLMClient`.
3. Add a branch in `LLMClient.complete()` dispatching on
   `self.provider`.

A failed import or a missing key always falls through to the
offline heuristic — no per-provider error handling required.
