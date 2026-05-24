# GoldDayTrading

> **Multi-agent LLM framework for intraday gold (XAU/USD) day trading.**
> Adapted from [tmk11/TradingAgents](https://github.com/tmk11/TradingAgents)
> (a gold-tilted fork of [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents))
> and re-engineered specifically for **day-trading the gold complex** —
> futures (`GC=F` / `MGC=F`), spot pairs (`XAUUSD=X`), and gold ETFs
> (`GLD`, `IAU`) on intraday timeframes.

---

## Why this fork exists

Upstream `TradingAgents` is a brilliant LangGraph framework, but it's
designed for **daily-cadence equity analysis** — it pulls daily
candles, runs a multi-round debate over many minutes, and produces a
"BUY / HOLD / SELL" thesis for a position held for days or weeks.

Day-trading gold is a different sport:

| Concern              | Upstream (daily) | GoldDayTrading (intraday)                  |
| -------------------- | ---------------- | ------------------------------------------- |
| Candles              | 1d               | 1m / 5m / 15m / 1h (+ 4h resample)          |
| Decision horizon     | days–weeks       | minutes–hours, **flat by NY close**         |
| Macro data           | FRED daily       | yfinance hourly DXY / yields / VIX / TIPS   |
| News                 | scraped digest   | last 6–12h RSS only (older = priced in)     |
| Catalysts            | ad-hoc mention   | structured 24h economic calendar + blackout |
| Indicators           | trend-following  | + VWAP, opening range, ATR stops, pivots    |
| Sessions             | n/a              | Tokyo / London / NY / overlap classifier    |
| Risk                 | qualitative      | deterministic R-multiple + daily loss cap   |
| Final output         | research report  | copy-pastable trade plan                    |
| Debate rounds        | up to many       | 1 (configurable)                            |
| External deps        | LangGraph, etc.  | yfinance + pandas + openai (light)          |

The result is a **decision-support system** that produces, on each
run, a single trade plan with entry / stop / TP1 / TP2 / position
size / time-in-force, refusing to fire if a high-impact USD print is
inside the blackout window or if R:R falls below your minimum.

> **Disclaimer.** This is research and decision-support tooling, not
> financial advice. Markets can and will hand you the loss your
> position-sizing assumed they wouldn't. Test on paper first.

---

## Architecture

```text
              ┌─────────────────────── Data layer ───────────────────────┐
              │ Intraday OHLCV (yfinance) │ Indicator battery │ Sessions │
              │ Macro pulse (DXY/yields/VIX/TIP/ES) │ Gold news (RSS)    │
              │           Economic calendar (ForexFactory)               │
              └──────────────────────────────────────────────────────────┘
                                          │
       ┌──────────┬──────────┬──────────┬──────────┐
       ▼          ▼          ▼          ▼          ▼
  Technical    Session     Macro      News /    Sentiment
   Analyst   Strategist    Pulse    Catalyst     (light)
       └──────────┴────┬─────┴──────────┴──────────┘
                       ▼
                  Bull ↔ Bear  (1 round)
                       │
                       ▼
                Research Manager  (LONG / SHORT / FLAT + levels)
                       │
                       ▼
                Risk Manager      (deterministic guardrails + LLM verdict)
                       │
                       ▼
                Day Trader        (final copy-pastable plan)
```

Every stage updates a shared `ctx` dict, so a run is deterministic
given the same inputs and the LLM seed — easy to inspect, replay,
and test.

---

## Install

Requires Python 3.10+.

```bash
git clone https://github.com/<your-fork>/GoldDayTrading.git
cd GoldDayTrading

python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e ".[anthropic]"   # Claude support
pip install -e ".[gemini]"      # Gemini support
pip install -e ".[dev]"         # pytest + pytest-mock
```

Copy the example env file and fill in **only the provider you want
to use** — any single one is enough:

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=... (or ANTHROPIC_API_KEY=..., etc.)
```

If no key is set, `GoldDayTrading` automatically downgrades to an
**offline heuristic** mode so the data pipeline still runs end-to-end
(useful for smoke tests, demos, and CI).

---

## Kiến trúc Agentic Workflow (mới — LangGraph + Graph RAG)

Phiên bản 0.3 bổ sung kiến trúc Agentic Workflow song song với
pipeline tuyến tính cũ. Thiết kế nằm trong package
`golddaytrading.agentic/` và **không thay thế** pipeline cũ —
`gdt analyze` vẫn chạy được như trước (kể cả ở chế độ offline).

Điểm khác biệt chính:

| Tiêu chí | Pipeline cũ (`gdt analyze`) | Workflow mới (`gdt agentic-run`) |
|---|---|---|
| Topology | Tuyến tính | StateGraph có chu trình (LangGraph) |
| Routing | Cố định | Risk Manager **tự chỉ định** node quay lại |
| Tool calling | Python gọi tool trước prompt | LLM tự gọi tool (`bind_tools` + `ToolNode`) |
| Bộ nhớ lịch sử | Không | Graph RAG (NetworkX KG + ChromaDB) |
| LLM | OpenAI/Anthropic/Gemini/offline | **API-only**, OpenAI-compatible |
| Local model | — | Không tải HF — orchestrator CPU thuần |

### Cài đặt extra `agentic`

```bash
pip install -e ".[agentic]"
export OPENAI_API_KEY=...
# (Tuỳ chọn) chỉ định model deep — KHÔNG hardcode trong code:
export GDT_DEEP_LLM=gpt-4o
```

### Chạy workflow

```bash
gdt agentic-run XAUUSD=X --max-iter 3
```

Workflow đi qua các node theo thứ tự (xem `agentic/graph.py`):

```
START → data_gatherer → technical_agent → macro_news_agent → risk_manager
                                                                  │
                                                                  ▼
              ┌──────── route_back (technical | macro_news) ◀────┤
              ▼                                                   │
       (vòng phản biện)                                           │
                                                                  ▼
                                                         memory_consolidator
                                                                  │
                                                                  ▼
                                                                 END
```

* `data_gatherer` — pure Python, fetch OHLCV + macro pulse + lịch
  + RSS news, đẩy vào `AgentState`.
* `technical_agent` — bind tool: `get_live_price`,
  `get_intraday_ohlcv`, `get_higher_timeframe_trend`,
  `get_active_session`, `get_level_pool`. Trả `AgentOutput`.
* `macro_news_agent` — bind tool: `get_macro_pulse`,
  `get_econ_calendar`, `get_gold_news`, **`query_historical_context`**
  (Graph RAG). Trả `AgentOutput`.
* `risk_manager` — không tool, dùng structured output
  `RiskManagerVerdict` để **chọn**: hoặc `finalize` (phát
  `FinalDecision`), hoặc `route_back` (chỉ định node nào quay lại
  + câu hỏi cụ thể).
* `memory_consolidator` — pure Python, không LLM. Chạy ngay trước
  `END` để ghi episode vào Graph RAG (xem mục Self-Learning bên dưới).

### Graph RAG (bộ nhớ lịch sử)

Bên trong `agentic/graph_rag.py`:

* **NetworkX MultiDiGraph** lưu quan hệ vĩ mô (DXY⊥GOLD,
  REAL_YIELD⊥GOLD, EURUSD⊥DXY, FOMC→DXY…). Đã seed sẵn các quan
  hệ prior; mở rộng qua `ingest_correlation()`.
* **ChromaDB persistent client** lưu các "trading episode" được
  embed bằng **OpenAI text-embedding-3-small** qua API. Không tải
  embedding model nội bộ — đúng yêu cầu API-only / no-GPU.
* `query_historical_context(query, k, regime_filter)` trả về top-k
  episode tương tự + cạnh KG liên quan, đã render thành block
  markdown để LLM đọc trực tiếp.

### Sử dụng từ Python

```python
from golddaytrading.agentic import run_agentic_workflow, render_final_report

state = run_agentic_workflow("XAUUSD=X", max_iterations=3)

print(state.final_decision.bias, state.final_decision.confidence)
print(render_final_report(state))
```

### Mở rộng KG / RAG bằng dữ liệu lịch sử

```python
from golddaytrading.agentic import GraphRAG, build_episode_from_macro_pulse
from golddaytrading.dataflows.macro_pulse import fetch_macro_pulse

rag = GraphRAG.default()
pulse = fetch_macro_pulse()
episode = build_episode_from_macro_pulse(
    pulse, gold_chg_1h=-0.42,
    note="CPI trên dự báo 0.1%, gold giảm.",
)
rag.ingest_episode(episode)
print(rag.stats())
```

### Self-Learning loop (auto-ingest mỗi run)

Workflow agentic tự động học từ chính các quyết định của nó. Sau
khi Risk Manager phát hành `FinalDecision`, node `memory_consolidator`
chạy ngay trước `END`:

* Build một `HistoricalEpisode` từ state đã hoàn chỉnh:
  * **narrative** ngắn gồm regime, macro deltas (DXY/US10Y/VIX
    1h), gold price, session, HTF trend, bias + entry/stop;
  * **entity_nodes** = `[GOLD, DXY, US10Y, VIX]` + event KG nodes
    (CPI/NFP/FOMC) khi sự kiện high-impact đang ở < 4h;
  * **metadata** primitive (regime, bias, confidence, entry, RR,
    rsi14, atr14, vwap, macd_hist…) — phục vụ filter sau này.
* Ingest qua `GraphRAG.ingest_episode()` → embed bằng OpenAI API
  → upsert ChromaDB → persist KG.
* Nếu Chroma/embedding API timeout → bắt exception, log
  `errors`, vẫn return `final_decision` bình thường.

Sau N run, Macro Agent ở các phiên sau gọi
`query_historical_context` sẽ retrieve được chính các episode
trước đó → workflow trở nên ngày càng "kinh nghiệm".

### `gdt agentic-info` — visibility cho bộ nhớ

Command CLI mới (không gọi LLM) để inspect Graph RAG:

```bash
gdt agentic-info                          # chỉ stats
gdt agentic-info --sample 5               # + 5 episode mới nhất
gdt agentic-info --sample 20 --top-connected 5
```

Output gồm:

* **Overview**: persist dir, tổng node/edge KG, tổng episode Chroma.
* **KG node types** (asset / macro / event / derived / auto).
* **KG relation types** (inverse / drives / leads_by_30m / …).
* **Top-connected nodes** (degree = in + out) — thường là `GOLD`,
  `DXY`, `REAL_YIELD`, `FOMC_release`.
* **Sample episodes** (khi `--sample N > 0`): timestamp, regime,
  bias, confidence, narrative, entities, metadata.

---

## CLI

The package installs two entry points: `golddaytrading` and the short
alias `gdt`.

### `analyze` — run one full analysis

```bash
gdt analyze                                         # XAUUSD=X on 15m
gdt analyze GC=F --tf 5m --account 25000 --risk-pct 0.5
gdt analyze GLD --tf 1h --provider anthropic --deep-llm claude-3-5-sonnet-20241022
gdt analyze XAUUSD=X --lang Vietnamese              # final plan in Vietnamese
gdt analyze --no-news --no-sentiment                # data + TA + risk only
```

Output ends with a markdown trade plan like:

```markdown
## Gold Day-Trade Plan — XAUUSD=X, 2026-05-23 14:32 UTC

- **Bias:** LONG
- **Setup:** VWAP reclaim during London/NY overlap
- **Entry trigger:** above 2354.20 (5-min close)
- **Stop:** 2349.40   (risk = $50, 1.5 ATR)
- **TP1 / TP2:** 2362.00 / 2370.00   (R:R = 1.6 / 2.4)
- **Position size:** 10.4 units
- **Time-in-force:** until 21:00 UTC (NY close) or 90 min max hold
- **Blackouts:** none in the next 60 min
- **Why now:** (1) DXY -0.18% over 1h, ^TNX -0.12%; (2) reclaim of VWAP
  inside the London-NY overlap; (3) RSI 56 with bullish MACD cross.
- **What kills it:** loss of VWAP and re-entry below 2349.40.
- **Risk Manager verdict:** APPROVE
```

### `info` — quick context dump (no LLM call)

```bash
gdt info               # current session, macro pulse, calendar
gdt info GC=F
```

### `backtest` — measure the deterministic edge

Walks forward through historical bars, builds the level pool at each
step, simulates each idea's outcome, and prints win-rate /
expectancy / Sharpe-R per setup, per session, and per HTF trend. No
LLM calls, fully deterministic.

```bash
gdt backtest XAUUSD=X --tf 15m --bars 2000 --strategy best_idea
gdt backtest GC=F --tf 1h  --bars 1500 --strategy all_ideas
```

`--strategy` can be `best_idea` (top-ranked only, matches live),
`all_ideas` (the whole pool — more samples for stats), or
`p_up_aligned` (only ideas whose bias matches the quant prior).

### `journal` — empirical edge from your live runs

The pipeline auto-logs every plan it produces to a SQLite journal
at `~/.golddaytrading/journal.sqlite3`. Once you've recorded
outcomes, the rolling per-setup stats are injected back into the
Research Manager's prompt so the LLM applies Bayesian reasoning.

```bash
# After a trade resolves, mark its outcome:
gdt journal record 42 --resolution tp1 --realised-r 2.0 --notes "VWAP reclaim"
gdt journal record 41 --resolution stop --realised-r -1.0
gdt journal record 40 --resolution expired --realised-r 0.4

# Inspect rolling 30-day stats:
gdt journal stats --days 30
gdt journal list --limit 10
gdt journal path
```

---

## Python API

```python
from golddaytrading import load_config, DayTradingPipeline

cfg = load_config(
    ticker="XAUUSD=X",
    primary_timeframe="15m",
    account_usd=10_000,
    risk_per_trade_pct=0.5,
    debate_rounds=1,
)
pipeline = DayTradingPipeline(cfg=cfg)
ctx = pipeline.run()

print(ctx["final_plan"])
# Inspect any intermediate artifact:
print(ctx["technical_report"])
print(ctx["risk_report"])
print(ctx["guardrail"])    # deterministic risk bundle
```

Each run is also persisted as a single markdown file under
`~/.golddaytrading/runs/` for later review.

---

## Configuration

All settings can be supplied either via `GDTConfig(...)`, the CLI
flags shown above, or environment variables. Below are the most
useful ones; see [`.env.example`](.env.example) for the full list.

| Env var                     | Default     | What it controls                     |
| --------------------------- | ----------- | ------------------------------------ |
| `GDT_DEFAULT_TICKER`        | `XAUUSD=X`  | Default ticker for `gdt analyze`     |
| `GDT_PRIMARY_TIMEFRAME`     | `15m`       | Primary intraday timeframe           |
| `GDT_HIGHER_TIMEFRAME`      | `1h`        | Trend-confirmation timeframe         |
| `GDT_LLM_PROVIDER`          | `openai`    | `openai` / `anthropic` / `gemini` / `offline` |
| `GDT_DEEP_LLM`              | `gpt-4o`    | Deep-thinking model                  |
| `GDT_QUICK_LLM`             | `gpt-4o-mini` | Quick-thinking model               |
| `GDT_ACCOUNT_USD`           | `10000`     | Account size for position sizing     |
| `GDT_RISK_PER_TRADE_PCT`    | `0.5`       | Max % risk per trade                 |
| `GDT_DAILY_LOSS_LIMIT_PCT`  | `2.0`       | Force flat after this much daily loss |
| `GDT_MIN_RR`                | `1.5`       | Minimum risk:reward ratio            |
| `GDT_ATR_STOP_MULT`         | `1.5`       | Stop = N × ATR(14)                   |
| `GDT_OUTPUT_LANGUAGE`       | `English`   | Output language for the final plan   |
| `GDT_DEBATE_ROUNDS`         | `1`         | Bull/Bear debate rounds              |
| `GDT_ENABLE_ECON_CALENDAR`  | `1`         | Toggle economic-calendar lookup      |
| `GDT_VWAP_ANCHOR_HOUR_UTC`  | `22`        | UTC hour at which session VWAP resets (5pm NY = 22) |
| `GDT_ENABLE_JOURNAL`        | `1`         | Auto-log every plan to the SQLite journal |
| `GDT_INJECT_JOURNAL_STATS`  | `1`         | Feed rolling per-setup stats into the RM prompt |
| `GDT_JOURNAL_STATS_DAYS_BACK` | `30`      | Rolling window for stats injection   |
| `GDT_JOURNAL_DB_PATH`       | `~/.golddaytrading/journal.sqlite3` | Override journal file path |

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The smoke tests build synthetic OHLCV in-process and exercise the
indicator battery, session classifier, risk guardrails, and the
offline LLM fallback — no network, no API keys, no external services.

---

## Differences from upstream `TradingAgents`

| Component             | Upstream                              | GoldDayTrading              |
| --------------------- | ------------------------------------- | --------------------------- |
| Orchestration         | LangGraph                             | Linear pipeline (plain Python) |
| Cadence               | 1-day candles                         | 1m / 5m / 15m / 1h          |
| Macro                 | FRED daily series                     | yfinance hourly pulse       |
| News                  | yfinance/AV news + RSS digest         | last-12h RSS, no auth       |
| Calendar              | n/a                                   | ForexFactory free JSON      |
| Sessions              | n/a                                   | UTC FX-session classifier   |
| Indicators            | StockStats library                    | Pure pandas/numpy           |
| Fundamentals analyst  | included                              | removed (gold has none)     |
| Debate rounds         | up to N                               | 1 (configurable)            |
| Risk                  | qualitative                           | hard guardrails + LLM       |
| Final output          | markdown research report              | structured trade plan       |
| Web UI                | FastAPI + React                       | not yet (v0.2 candidate)    |

---

## Roadmap

- v0.2 — Web UI for queueing intraday runs and reviewing plans.
- v0.3 — Backtest harness (replay 5m bars, score the plans).
- v0.4 — Live paper-trading hook into a broker (OANDA / IBKR).

---

## License

Apache 2.0 — same as the upstream `TradingAgents`. See [LICENSE](LICENSE).
