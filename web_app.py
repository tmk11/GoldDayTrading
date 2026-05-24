from __future__ import annotations

import json
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse

from golddaytrading.config import load_config
from golddaytrading.graph.pipeline import DayTradingPipeline, _render_run_md


app = FastAPI(title="GoldDayTrading")
JOBS_DIR = Path.home() / ".golddaytrading" / "web_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
_executor = ThreadPoolExecutor(max_workers=1)
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _meta_path(job_id: str) -> Path:
    return _job_dir(job_id) / "meta.json"


def _read_meta(job_id: str) -> dict[str, Any]:
    try:
        return json.loads(_meta_path(job_id).read_text(encoding="utf-8"))
    except Exception:
        return {"id": job_id, "status": "missing"}


def _write_meta(job_id: str, meta: dict[str, Any]) -> None:
    with _lock:
        _job_dir(job_id).mkdir(parents=True, exist_ok=True)
        _meta_path(job_id).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


def _append_log(job_id: str, message: str) -> None:
    line = f"[{_now()}] {message}\n"
    with _lock:
        (_job_dir(job_id) / "run.log").open("a", encoding="utf-8").write(line)


def _list_jobs() -> list[dict[str, Any]]:
    jobs = [_read_meta(path.name) for path in JOBS_DIR.iterdir() if path.is_dir()]
    return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)


def _run_pipeline_job(job_id: str) -> None:
    meta = _read_meta(job_id)
    meta.update({"status": "running", "started_at": _now()})
    _write_meta(job_id, meta)

    def logger(*parts: object) -> None:
        _append_log(job_id, " ".join(str(part) for part in parts))

    try:
        cfg = load_config(
            ticker=meta["ticker"],
            primary_timeframe=meta["tf"],
            output_language=meta["lang"],
            debate_rounds=int(meta.get("debate_rounds", 1)),
            debug=True,
        )
        logger(
            "Starting full multi-agent pipeline:",
            cfg.ticker,
            cfg.primary_timeframe,
            cfg.llm_provider,
            cfg.deep_llm,
        )
        ctx = DayTradingPipeline(cfg=cfg, logger=logger).run(cfg.ticker)
        result_md = _render_run_md(ctx, cfg)
        (_job_dir(job_id) / "result.md").write_text(result_md, encoding="utf-8")
        meta.update({
            "status": "done",
            "finished_at": _now(),
            "wall_clock_sec": ctx.get("wall_clock_sec"),
            "provider": ctx.get("llm_provider"),
            "model": ctx.get("llm_model_deep"),
        })
        logger("Finished successfully in", ctx.get("wall_clock_sec"), "seconds")
    except Exception as exc:
        (_job_dir(job_id) / "error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        meta.update({
            "status": "error",
            "finished_at": _now(),
            "error": f"{type(exc).__name__}: {exc}",
        })
        logger("FAILED:", meta["error"])
    finally:
        _write_meta(job_id, meta)


def _status_label(status: str) -> str:
    labels = {
        "queued": "Đang chờ",
        "running": "Đang chạy",
        "done": "Hoàn tất",
        "error": "Lỗi",
        "interrupted": "Bị dừng",
        "missing": "Không thấy",
    }
    return labels.get(status, status or "unknown")


def _short_time(value: object) -> str:
    text = str(value or "")
    if "T" in text:
        text = text.replace("T", " ").replace("+00:00", " UTC")
    return text


def _page(title: str, body: str, *, auto_refresh: bool = False) -> str:
    refresh = "<meta http-equiv='refresh' content='5'>" if auto_refresh else ""
    return f"""
<!doctype html>
<html lang="vi">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
    {refresh}
    <title>{escape(title)}</title>
    <style>
      :root {{
        color-scheme: dark;
        --bg: #070a12;
        --panel: rgba(17, 24, 39, .82);
        --panel-strong: #111827;
        --card: rgba(255, 255, 255, .06);
        --card-border: rgba(255, 255, 255, .12);
        --text: #f8fafc;
        --muted: #94a3b8;
        --gold: #f5c451;
        --gold-2: #ffdf7e;
        --green: #36d399;
        --orange: #fbbf24;
        --red: #fb7185;
        --blue: #60a5fa;
        --shadow: 0 24px 80px rgba(0,0,0,.35);
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        min-height: 100vh;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--text);
        background:
          radial-gradient(circle at top left, rgba(245,196,81,.18), transparent 32rem),
          radial-gradient(circle at 85% 10%, rgba(96,165,250,.16), transparent 28rem),
          linear-gradient(135deg, #070a12 0%, #0f172a 52%, #111827 100%);
      }}
      a {{ color: var(--gold-2); text-decoration: none; }}
      a:hover {{ text-decoration: underline; }}
      .shell {{ width: min(1120px, calc(100% - 32px)); margin: 0 auto; padding: 24px 0 48px; }}
      .hero {{
        border: 1px solid var(--card-border);
        border-radius: 28px;
        padding: 24px;
        background: linear-gradient(135deg, rgba(17,24,39,.94), rgba(30,41,59,.76));
        box-shadow: var(--shadow);
        overflow: hidden;
        position: relative;
      }}
      .hero::after {{
        content: "";
        position: absolute;
        width: 210px; height: 210px; right: -70px; top: -90px;
        background: radial-gradient(circle, rgba(245,196,81,.24), transparent 68%);
        pointer-events: none;
      }}
      .topbar {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 18px; }}
      .brand {{ display: flex; align-items: center; gap: 12px; }}
      .logo {{
        width: 46px; height: 46px; border-radius: 15px;
        display: grid; place-items: center;
        background: linear-gradient(135deg, var(--gold), #b7791f);
        color: #111827; font-size: 24px; font-weight: 900;
        box-shadow: 0 10px 30px rgba(245,196,81,.26);
      }}
      h1 {{ font-size: clamp(28px, 6vw, 48px); line-height: 1.02; margin: 0; letter-spacing: -0.04em; }}
      h2 {{ margin: 0 0 14px; font-size: clamp(20px, 4vw, 28px); letter-spacing: -0.02em; }}
      p {{ color: #dbeafe; line-height: 1.6; }}
      .muted {{ color: var(--muted); }}
      .grid {{ display: grid; grid-template-columns: 1.05fr .95fr; gap: 18px; align-items: start; }}
      .panel, .job-card, .result-card {{
        border: 1px solid var(--card-border);
        border-radius: 22px;
        background: var(--panel);
        backdrop-filter: blur(14px);
        box-shadow: 0 14px 40px rgba(0,0,0,.22);
      }}
      .panel {{ padding: 18px; }}
      .form-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
      label {{ display: grid; gap: 7px; color: var(--muted); font-size: 13px; font-weight: 700; }}
      input, select {{
        width: 100%; min-height: 46px;
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 14px;
        padding: 10px 12px;
        background: rgba(15,23,42,.92);
        color: var(--text);
        font: inherit;
        outline: none;
      }}
      input:focus, select:focus {{ border-color: var(--gold); box-shadow: 0 0 0 3px rgba(245,196,81,.16); }}
      .btn {{
        min-height: 48px;
        border: 0;
        border-radius: 16px;
        padding: 0 18px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        color: #111827;
        background: linear-gradient(135deg, var(--gold-2), var(--gold));
        font-weight: 900;
        font-size: 15px;
        box-shadow: 0 12px 28px rgba(245,196,81,.24);
        cursor: pointer;
      }}
      .btn.secondary {{ color: var(--text); background: rgba(255,255,255,.08); box-shadow: none; border: 1px solid var(--card-border); }}
      .btn.danger {{ color: #fff; background: rgba(251,113,133,.18); box-shadow: none; border: 1px solid rgba(251,113,133,.34); }}
      .submit-row {{ margin-top: 14px; display: flex; gap: 10px; flex-wrap: wrap; }}
      .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }}
      .chip {{ border: 1px solid rgba(255,255,255,.12); border-radius: 999px; padding: 7px 10px; color: #e2e8f0; background: rgba(255,255,255,.06); font-size: 13px; }}
      .job-list {{ display: grid; gap: 12px; margin-top: 14px; }}
      .job-card {{ padding: 16px; display: grid; grid-template-columns: 1fr auto; gap: 12px; }}
      .job-main {{ min-width: 0; }}
      .job-title {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 8px; }}
      .job-id {{ font-weight: 900; font-size: 18px; color: var(--text); }}
      .meta {{ display: flex; flex-wrap: wrap; gap: 8px; color: var(--muted); font-size: 13px; }}
      .actions {{ display: flex; align-items: center; gap: 8px; }}
      .badge {{ display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 900; border: 1px solid transparent; }}
      .badge.queued {{ color: #dbeafe; background: rgba(96,165,250,.16); border-color: rgba(96,165,250,.32); }}
      .badge.running {{ color: #fde68a; background: rgba(251,191,36,.16); border-color: rgba(251,191,36,.34); }}
      .badge.done {{ color: #bbf7d0; background: rgba(54,211,153,.14); border-color: rgba(54,211,153,.32); }}
      .badge.error, .badge.interrupted, .badge.missing {{ color: #fecdd3; background: rgba(251,113,133,.14); border-color: rgba(251,113,133,.32); }}
      .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 18px 0; }}
      .stat {{ border: 1px solid var(--card-border); border-radius: 18px; padding: 13px; background: rgba(255,255,255,.05); }}
      .stat span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 5px; }}
      .stat strong {{ font-size: 15px; word-break: break-word; }}
      pre {{
        white-space: pre-wrap;
        overflow-x: auto;
        background: #020617;
        color: #e5e7eb;
        border: 1px solid rgba(255,255,255,.1);
        padding: 16px;
        border-radius: 18px;
        line-height: 1.55;
        font-size: 14px;
      }}
      .empty {{ text-align: center; padding: 28px 14px; color: var(--muted); border: 1px dashed var(--card-border); border-radius: 20px; }}
      .sticky-actions {{ position: sticky; top: 0; z-index: 2; padding: 10px 0; backdrop-filter: blur(10px); }}
      @media (max-width: 760px) {{
        .shell {{ width: min(100% - 22px, 1120px); padding-top: 12px; }}
        .hero {{ padding: 18px; border-radius: 22px; }}
        .topbar, .grid, .job-card {{ grid-template-columns: 1fr; display: grid; }}
        .form-grid {{ grid-template-columns: 1fr; }}
        .stats {{ grid-template-columns: repeat(2, 1fr); }}
        .actions {{ justify-content: stretch; }}
        .actions .btn, .submit-row .btn {{ flex: 1; width: 100%; }}
        pre {{ font-size: 13px; padding: 13px; }}
      }}
    </style>
  </head>
  <body><main class="shell">{body}</main></body>
</html>
"""


def _mark_interrupted_jobs() -> None:
    for job in _list_jobs():
        if job.get("status") in {"queued", "running"}:
            job["status"] = "interrupted"
            job["finished_at"] = _now()
            job["error"] = "App restarted before this background job finished."
            _write_meta(job["id"], job)
            _append_log(job["id"], "Interrupted by app restart")

@app.on_event("startup")
def startup() -> None:
    _mark_interrupted_jobs()

@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"

@app.get("/", response_class=HTMLResponse)
def home() -> str:
    job_cards = []
    jobs = _list_jobs()
    running_count = sum(1 for job in jobs if job.get("status") in {"queued", "running"})
    done_count = sum(1 for job in jobs if job.get("status") == "done")
    for job in jobs:
        raw_job_id = job.get("id", "")
        job_id = escape(raw_job_id)
        status = escape(job.get("status", "unknown"))
        status_text = escape(_status_label(job.get("status", "unknown")))
        seconds = str(job.get("wall_clock_sec", "") or "—")
        job_cards.append(
            f"""
            <article class="job-card">
              <div class="job-main">
                <div class="job-title">
                  <a class="job-id" href="job/{job_id}">#{escape(raw_job_id[:8])}</a>
                  <span class="badge {status}">● {status_text}</span>
                </div>
                <div class="meta">
                  <span>🏷️ {escape(job.get('ticker', ''))}</span>
                  <span>⏱️ {escape(job.get('tf', ''))}</span>
                  <span>🗣️ {escape(job.get('lang', ''))}</span>
                  <span>💬 {escape(str(job.get('debate_rounds', 1)))} round</span>
                  <span>🕒 {_short_time(job.get('created_at', ''))}</span>
                  <span>⚡ {escape(seconds)}s</span>
                </div>
              </div>
              <div class="actions">
                <a class="btn secondary" href="job/{job_id}">Xem</a>
                <a class="btn danger" href="delete/{job_id}" onclick="return confirm('Xoá job này?')">Xoá</a>
              </div>
            </article>
            """
        )
    jobs_markup = "".join(job_cards) or "<div class='empty'>Chưa có phân tích nào. Tạo job đầu tiên ở form bên trên.</div>"
    body = f"""
    <section class="hero">
      <div class="topbar">
        <div class="brand"><div class="logo">Au</div><div><h1>GoldDayTrading</h1><p class="muted" style="margin:.25rem 0 0">Multi-agent gold day trading pipeline</p></div></div>
      </div>
      <div class="grid">
        <div>
          <h2>Phân tích vàng bằng nhiều AI agent</h2>
          <p>Chạy nền full pipeline gốc: analyst đọc dữ liệu, bull/bear tranh luận, research manager tổng hợp, risk manager kiểm tra, day trader xuất kế hoạch.</p>
          <div class="chips">
            <span class="chip">Không cần giữ tab mở</span>
            <span class="chip">Tự lưu kết quả</span>
            <span class="chip">Xem lại hoặc xoá bất cứ lúc nào</span>
          </div>
        </div>
        <form class="panel" action="start" method="get">
          <h2>Tạo phân tích</h2>
          <div class="form-grid">
            <label>Ticker <input name="ticker" value="GC=F" autocomplete="off" /></label>
            <label>Timeframe
              <select name="tf"><option>15m</option><option>5m</option><option>1m</option><option>1h</option></select>
            </label>
            <label>Ngôn ngữ
              <select name="lang"><option>Vietnamese</option><option>English</option></select>
            </label>
            <label>Debate rounds <input name="debate_rounds" value="1" inputmode="numeric" /></label>
          </div>
          <div class="submit-row"><button class="btn" type="submit">🚀 Analyze nền</button></div>
          <p class="muted" style="margin-bottom:0">Job chạy tuần tự để tránh quá tải API. Dùng GC=F nếu XAUUSD=X không có dữ liệu.</p>
        </form>
      </div>
    </section>

    <section style="margin-top:18px">
      <div class="stats">
        <div class="stat"><span>Tổng job</span><strong>{len(jobs)}</strong></div>
        <div class="stat"><span>Đang chạy/chờ</span><strong>{running_count}</strong></div>
        <div class="stat"><span>Hoàn tất</span><strong>{done_count}</strong></div>
        <div class="stat"><span>Port</span><strong>3333</strong></div>
      </div>
      <div class="panel">
        <h2>Danh sách phân tích</h2>
        <div class="job-list">{jobs_markup}</div>
      </div>
    </section>
    """
    return _page("GoldDayTrading", body)

@app.get("/start")
def start(
    ticker: str = Query("GC=F"),
    tf: str = Query("15m"),
    lang: str = Query("Vietnamese"),
    debate_rounds: int = Query(1),
) -> RedirectResponse:
    job_id = uuid.uuid4().hex
    meta = {
        "id": job_id,
        "status": "queued",
        "ticker": ticker,
        "tf": tf,
        "lang": lang,
        "debate_rounds": max(1, min(int(debate_rounds), 3)),
        "created_at": _now(),
    }
    _write_meta(job_id, meta)
    _append_log(job_id, "Queued")
    _executor.submit(_run_pipeline_job, job_id)
    return RedirectResponse(url=f"job/{job_id}", status_code=303)


@app.get("/job/{job_id}", response_class=HTMLResponse)
def job_detail(job_id: str) -> str:
    meta = _read_meta(job_id)
    status = meta.get("status", "missing")
    auto_refresh = status in {"queued", "running"}
    result_path = _job_dir(job_id) / "result.md"
    log_path = _job_dir(job_id) / "run.log"
    error_path = _job_dir(job_id) / "error.txt"
    result = result_path.read_text(encoding="utf-8") if result_path.exists() else ""
    log = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    error = error_path.read_text(encoding="utf-8") if error_path.exists() else ""
    status_text = escape(_status_label(status))
    waiting_text = "Job chưa xong. Trang sẽ tự refresh mỗi 5 giây khi đang chạy."
    body = f"""
    <div class="sticky-actions">
      <div class="actions">
        <a class="btn secondary" href="../">← Danh sách</a>
        <a class="btn danger" href="../delete/{escape(job_id)}" onclick="return confirm('Xoá job này?')">Xoá job</a>
      </div>
    </div>

    <section class="hero">
      <div class="topbar">
        <div class="brand"><div class="logo">Au</div><div><h1>Job #{escape(job_id[:8])}</h1><p class="muted" style="margin:.25rem 0 0">Full multi-agent pipeline result</p></div></div>
        <span class="badge {escape(status)}">● {status_text}</span>
      </div>
      <div class="stats">
        <div class="stat"><span>Ticker</span><strong>{escape(meta.get('ticker', ''))}</strong></div>
        <div class="stat"><span>Timeframe</span><strong>{escape(meta.get('tf', ''))}</strong></div>
        <div class="stat"><span>Ngôn ngữ</span><strong>{escape(meta.get('lang', ''))}</strong></div>
        <div class="stat"><span>Debate</span><strong>{escape(str(meta.get('debate_rounds', 1)))} round</strong></div>
      </div>
      <div class="stats">
        <div class="stat"><span>Created</span><strong>{escape(_short_time(meta.get('created_at', '')))}</strong></div>
        <div class="stat"><span>Started</span><strong>{escape(_short_time(meta.get('started_at', '')) or '—')}</strong></div>
        <div class="stat"><span>Finished</span><strong>{escape(_short_time(meta.get('finished_at', '')) or '—')}</strong></div>
        <div class="stat"><span>Seconds</span><strong>{escape(str(meta.get('wall_clock_sec', '') or '—'))}</strong></div>
      </div>
      {('<div class="panel"><h2>Lỗi</h2><pre>' + escape(error or meta.get('error', '')) + '</pre></div>') if error or meta.get('error') else ''}
    </section>

    <section class="result-card" style="margin-top:18px; padding:18px">
      <h2>Kết quả full pipeline</h2>
      <p class="muted">Bao gồm final plan, risk manager, research manager, bull/bear debate và các analyst report.</p>
      <pre>{escape(result or waiting_text)}</pre>
    </section>

    <section class="result-card" style="margin-top:18px; padding:18px">
      <h2>Log tiến trình</h2>
      <p class="muted">Theo dõi agent đang chạy tới bước nào.</p>
      <pre>{escape(log or '(no logs)')}</pre>
    </section>
    """
    return _page(f"Job {job_id[:8]}", body, auto_refresh=auto_refresh)

@app.get("/delete/{job_id}")
def delete_job(job_id: str) -> RedirectResponse:
    target = _job_dir(job_id)
    if target.exists() and target.is_dir():
        shutil.rmtree(target)
    return RedirectResponse(url="../", status_code=303)
