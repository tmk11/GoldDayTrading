"""Định nghĩa schema state cho LangGraph workflow.

`AgentState` là "tờ giấy chung" mà mọi node trong đồ thị đọc/ghi.
LangGraph yêu cầu state phải là `TypedDict` hoặc Pydantic model;
ta chọn **Pydantic v2** vì:

* Validate dữ liệu chặt chẽ ngay khi node trả về (phát hiện sớm
  hallucination dạng "stop = entry" hay "confidence > 1").
* JSON-serializable mặc định → dễ ghi log, replay, hoặc gửi qua
  message queue nếu sau này muốn scale ra dạng worker.
* Hỗ trợ field `default_factory` và `Annotated` reducer (cần cho
  `agent_outputs` / `debate_history` để LangGraph biết cách *merge*
  state khi nhiều node cùng cập nhật).

Reducer pattern
---------------

LangGraph `StateGraph` mặc định *thay thế* giá trị field khi node
return. Với các field "tích lũy" (như debate, log của agent) ta
cần `operator.add` reducer để các bản cập nhật được **append** thay
vì ghi đè. Xem `Annotated[..., add_messages]` trong LangGraph docs.

Quy ước field
-------------

* `timestamp`, `asset` — bất biến trong một run.
* `market_data`, `macro_events` — do `data_gatherer` node ghi 1 lần
  ở đầu, nhưng có thể bị refresh nếu Risk Manager yêu cầu re-fetch.
* `agent_outputs` — dict tích lũy: mỗi agent ghi key của mình.
* `debate_history` — list tích lũy theo thứ tự thời gian.
* `final_decision` — chỉ Risk Manager hoặc nhánh `END` mới ghi.
* `iteration` — đếm số vòng lặp đã chạy; hardcap để tránh loop.
* `errors` — log lỗi non-fatal (API timeout, feed unavailable...).
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-schema
# ---------------------------------------------------------------------------


TradingBias = Literal["LONG", "SHORT", "NEUTRAL"]
"""Hướng giao dịch cuối cùng. Dùng `NEUTRAL` thay cho `FLAT` để
khớp với gợi ý trong yêu cầu bài toán; mapping sang pipeline cũ:
NEUTRAL = FLAT (không vào lệnh).
"""


class MarketDataSnapshot(BaseModel):
    """Ảnh chụp dữ liệu thị trường tại một thời điểm.

    Không lưu nguyên DataFrame trong state (DataFrame không
    JSON-serializable và quá nặng). Chỉ giữ những con số và mô tả
    cô đọng mà các agent thực sự cần để suy luận. Toàn bộ DataFrame
    gốc nằm ở "scratchpad" cục bộ trong từng node.
    """

    primary_timeframe: str = Field(..., description="VD: '5m', '15m', '1h'.")
    higher_timeframe: str = Field(..., description="Khung trend cao hơn.")

    # Giá hiện tại + OHLC của nến cuối cùng
    last_price: Optional[float] = None
    last_open: Optional[float] = None
    last_high: Optional[float] = None
    last_low: Optional[float] = None
    last_close: Optional[float] = None
    last_volume: Optional[float] = None
    last_bar_time_utc: Optional[datetime] = None

    # Snapshot indicator (do compute_indicators sinh ra) — chỉ lưu
    # các giá trị trên nến cuối, không phải cả series.
    indicators: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description=(
            "VD: {'ema20': 2350.1, 'rsi14': 58.2, 'atr14': 4.6, "
            "'vwap': 2348.7, 'macd_hist': 0.21}"
        ),
    )

    # Block markdown đã render sẵn (cho LLM đọc trực tiếp).
    ohlcv_block: Optional[str] = None
    indicator_block: Optional[str] = None
    higher_indicator_block: Optional[str] = None

    # Phiên giao dịch hiện tại + tag trend khung cao.
    active_session: Optional[str] = None
    htf_trend: Optional[Literal["up", "down", "chop"]] = None


class MacroEvent(BaseModel):
    """Một sự kiện kinh tế trong calendar (ForexFactory).

    Tách riêng class để dễ validate `when_utc` luôn timezone-aware.
    """

    title: str
    country: str
    impact: Literal["High", "Medium", "Low"]
    when_utc: datetime
    forecast: Optional[str] = None
    previous: Optional[str] = None

    @field_validator("when_utc")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)

    def minutes_until(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now(timezone.utc)
        return (self.when_utc - now).total_seconds() / 60.0


class AgentOutput(BaseModel):
    """Output có cấu trúc của một agent node.

    Mỗi agent (Technical, Macro, News...) trả về một `AgentOutput`
    để Risk Manager có thể **so sánh** giữa các agent một cách
    deterministic (vd: thấy Technical bullish nhưng News bearish thì
    quay lại debate). Free-form `summary` vẫn được giữ để Day
    Trader render plan cuối cùng.
    """

    agent_name: str = Field(..., description="VD: 'technical', 'macro_news'.")
    bias: TradingBias = "NEUTRAL"
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    summary: str = Field(..., description="Markdown ngắn gọn (<=300 từ).")
    key_levels: Dict[str, float] = Field(
        default_factory=dict,
        description="Tuỳ chọn: entry/stop/tp1/tp2 nếu agent có ý tưởng cụ thể.",
    )
    cited_sources: List[str] = Field(
        default_factory=list,
        description="ID/URL của headline hoặc node KG đã tham chiếu.",
    )
    tools_called: List[str] = Field(
        default_factory=list,
        description="Tên các tool agent đã gọi trong lượt này.",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DebateMessage(BaseModel):
    """Một thông điệp trong lịch sử debate / self-reflection."""

    role: Literal["technical", "macro_news", "risk_manager", "system"]
    content: str
    iteration: int = Field(0, ge=0, description="Vòng lặp graph khi message phát sinh.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FinalDecision(BaseModel):
    """Quyết định cuối cùng do Risk Manager phát ra.

    Đây là *contract* mà downstream (UI, broker, journal) ký kết.
    Schema bị ràng buộc kiểu chặt: stop loss và take profit phải
    nhất quán với `bias` (LONG: stop < entry < tp; SHORT: ngược
    lại). Validation thực hiện trong `model_validator`.
    """

    bias: TradingBias
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., description="Tóm tắt lý do <= 200 từ.")
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    rr_ratio: Optional[float] = None
    position_size_units: Optional[float] = None
    time_in_force_minutes: Optional[int] = Field(
        None, ge=1,
        description="Tối đa giữ lệnh (phút). None = đến hết phiên NY.",
    )
    blackout_warning: Optional[str] = None
    contradictions_resolved: List[str] = Field(
        default_factory=list,
        description="Mâu thuẫn nào giữa các agent đã được giải quyết ở vòng này.",
    )

    @model_validator(mode="after")
    def _check_geometry(self) -> "FinalDecision":
        # NEUTRAL không cần entry/stop.
        if self.bias == "NEUTRAL":
            return self
        # LONG / SHORT cần đủ entry và stop.
        if self.entry is None or self.stop_loss is None:
            raise ValueError(
                "LONG/SHORT decision phải có entry và stop_loss."
            )
        if self.entry == self.stop_loss:
            raise ValueError("entry và stop_loss không được trùng nhau.")
        if self.bias == "LONG":
            if self.stop_loss >= self.entry:
                raise ValueError("LONG: stop_loss phải < entry.")
            if self.take_profit_1 is not None and self.take_profit_1 <= self.entry:
                raise ValueError("LONG: take_profit_1 phải > entry.")
        elif self.bias == "SHORT":
            if self.stop_loss <= self.entry:
                raise ValueError("SHORT: stop_loss phải > entry.")
            if self.take_profit_1 is not None and self.take_profit_1 >= self.entry:
                raise ValueError("SHORT: take_profit_1 phải < entry.")
        return self


# ---------------------------------------------------------------------------
# Reducer cho các field tích luỹ
# ---------------------------------------------------------------------------


def _merge_agent_outputs(
    left: Dict[str, AgentOutput],
    right: Dict[str, AgentOutput],
) -> Dict[str, AgentOutput]:
    """Reducer: gộp dict, các key trùng được ghi đè bởi `right` (mới hơn)."""
    out: Dict[str, AgentOutput] = dict(left or {})
    out.update(right or {})
    return out


def _append_errors(
    left: List[str], right: List[str]
) -> List[str]:
    """Reducer: append lỗi mới vào danh sách lỗi tích luỹ."""
    return list(left or []) + list(right or [])


# ---------------------------------------------------------------------------
# State chính
# ---------------------------------------------------------------------------


class AgentState(BaseModel):
    """State dùng chung cho toàn bộ LangGraph workflow.

    Các field dùng `Annotated[..., reducer]` được LangGraph nhận diện
    để **merge** thay vì ghi đè khi node return một `AgentState`
    mới. Field còn lại theo cơ chế thay thế (latest-wins).
    """

    # ------- Bất biến trong run -------
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Thời điểm bắt đầu run (UTC).",
    )
    asset: str = Field(
        "XAUUSD=X",
        description="Symbol giao dịch. Mặc định XAU/USD spot.",
    )

    # ------- Dữ liệu thị trường -------
    market_data: Optional[MarketDataSnapshot] = None
    """Snapshot OHLCV + indicator + session. Có thể bị refresh khi
    Risk Manager yêu cầu re-fetch (vd: dữ liệu đã cũ > N phút)."""

    macro_events: List[MacroEvent] = Field(default_factory=list)
    """Lịch kinh tế 24h tới đã filter high-impact USD/EUR."""

    # Block markdown thô của macro pulse + news (LLM đọc trực tiếp).
    macro_pulse_block: Optional[str] = None
    news_block: Optional[str] = None
    calendar_block: Optional[str] = None

    # ------- Output từ các agent (tích luỹ qua các vòng) -------
    agent_outputs: Annotated[
        Dict[str, AgentOutput], _merge_agent_outputs
    ] = Field(default_factory=dict)

    # ------- Lịch sử debate / self-reflection -------
    debate_history: Annotated[List[DebateMessage], operator.add] = Field(
        default_factory=list
    )

    # ------- Kết quả cuối -------
    final_decision: Optional[FinalDecision] = None

    # ------- Điều khiển graph -------
    iteration: int = Field(
        0, ge=0,
        description=(
            "Số vòng đã đi qua Risk Manager. Conditional edge sẽ "
            "buộc END khi vượt `max_iterations`."
        ),
    )
    max_iterations: int = Field(3, ge=1, le=10)
    next_node: Optional[Literal[
        "technical_agent", "macro_news_agent", "risk_manager", "END"
    ]] = Field(
        None,
        description=(
            "Hint do Risk Manager đặt cho conditional edge. None = "
            "để router quyết định bằng heuristic mặc định."
        ),
    )

    # ------- Telemetry -------
    errors: Annotated[List[str], _append_errors] = Field(default_factory=list)
    llm_provider: str = "openai"
    llm_model_deep: str = "gpt-4o"
    llm_model_quick: str = "gpt-4o-mini"

    # ----------------------------------------------------------------
    # Convenience helpers
    # ----------------------------------------------------------------

    def has_contradiction(self) -> bool:
        """True nếu có ít nhất một cặp agent bias đối nghịch nhau.

        Risk Manager dùng để quyết định route lại Technical/Macro.
        """
        biases = {
            name: out.bias
            for name, out in self.agent_outputs.items()
            if out.bias != "NEUTRAL"
        }
        return ("LONG" in biases.values()) and ("SHORT" in biases.values())

    def consensus_bias(self) -> TradingBias:
        """Bias đa số có trọng số confidence của các agent."""
        score = 0.0
        for out in self.agent_outputs.values():
            if out.bias == "LONG":
                score += out.confidence
            elif out.bias == "SHORT":
                score -= out.confidence
        if score > 0.15:
            return "LONG"
        if score < -0.15:
            return "SHORT"
        return "NEUTRAL"

    def to_summary_dict(self) -> Dict[str, Any]:
        """Tóm tắt nhẹ để log/journal (không kèm các block markdown to)."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "asset": self.asset,
            "iteration": self.iteration,
            "agent_biases": {
                name: {"bias": out.bias, "confidence": out.confidence}
                for name, out in self.agent_outputs.items()
            },
            "final_decision": (
                self.final_decision.model_dump(mode="json")
                if self.final_decision else None
            ),
            "errors": self.errors,
        }

    model_config = {
        "arbitrary_types_allowed": False,
        "extra": "forbid",
    }
