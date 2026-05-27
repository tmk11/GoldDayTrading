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
FinalAction = Literal["LONG", "SHORT", "NO_TRADE", "LONG_SETUP", "SHORT_SETUP"]
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

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_agent_confidence(cls, value):
        if isinstance(value, (int, float)) and value > 1:
            return float(value) / 100.0
        return value

    @field_validator("key_levels", mode="before")
    @classmethod
    def _normalize_key_levels(cls, value):
        if value is None:
            return {}
        if isinstance(value, dict):
            return {
                str(key): float(val)
                for key, val in value.items()
                if isinstance(val, (int, float))
            }
        if isinstance(value, list):
            normalized: Dict[str, float] = {}
            for idx, item in enumerate(value, start=1):
                if isinstance(item, dict):
                    prefix = str(item.get("setup_id") or item.get("setup") or f"setup_{idx}")
                    for key, val in item.items():
                        if isinstance(val, (int, float)):
                            normalized[f"{prefix}.{key}"] = float(val)
                elif isinstance(item, (int, float)):
                    normalized[f"level_{idx}"] = float(item)
            return normalized
        return {}

class DebateCase(BaseModel):
    """Lightweight bull/bear case used as an advisory debate layer."""

    side: Literal["bull", "bear"]
    thesis: str = Field(..., description="Bullish/bearish thesis, concise markdown.")
    supporting_evidence: List[str] = Field(default_factory=list)
    required_confirmation: str = ""
    invalidation_level: Optional[float] = None
    risk_factors: List[str] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("supporting_evidence", "risk_factors", mode="before")
    @classmethod
    def _listify(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("required_confirmation", mode="before")
    @classmethod
    def _stringify_confirmation(cls, value):
        if isinstance(value, list):
            return "; ".join(str(item) for item in value)
        if value is None:
            return ""
        return str(value)

    @field_validator("invalidation_level", mode="before")
    @classmethod
    def _parse_invalidation_level(cls, value):
        if value is None or isinstance(value, (int, float)):
            return value
        text = str(value)
        import re
        match = re.search(r"\d+(?:\.\d+)?", text.replace(",", ""))
        return float(match.group(0)) if match else None

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_confidence(cls, value):
        if isinstance(value, (int, float)) and value > 1:
            return float(value) / 100.0
        return value


class DebateMessage(BaseModel):
    """Một thông điệp trong lịch sử debate / self-reflection."""

    role: Literal[
        "technical", "macro_news", "bull_case", "bear_case",
        "risk_manager", "system",
    ]
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

    bias: TradingBias = "NEUTRAL"
    final_action: FinalAction = "NO_TRADE"
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    confidence_score: int = Field(0, ge=0, le=100)
    rationale: str = Field("", description="Tóm tắt lý do <= 200 từ.")
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit: Optional[float] = None
    take_profit_2: Optional[float] = None
    rr_ratio: Optional[float] = None
    risk_reward: Optional[float] = None
    position_size_units: Optional[float] = None
    position_size_recommendation: Optional[str] = None
    reasons: List[str] = Field(default_factory=list)
    conditions_to_cancel_trade: List[str] = Field(default_factory=list)
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
        if self.final_action == "LONG_SETUP":
            self.bias = "LONG"
        elif self.final_action == "SHORT_SETUP":
            self.bias = "SHORT"
        elif self.bias == "NEUTRAL":
            self.final_action = "NO_TRADE"
        elif self.final_action == "NO_TRADE":
            self.final_action = self.bias
        if self.confidence == 0 and self.confidence_score > 0:
            self.confidence = self.confidence_score / 100.0
        if not self.rationale and self.reasons:
            self.rationale = "; ".join(self.reasons[:3])
        self.confidence_score = int(round(self.confidence * 100))
        if self.take_profit is None:
            self.take_profit = self.take_profit_1
        if self.risk_reward is None:
            self.risk_reward = self.rr_ratio
        if self.position_size_recommendation is None:
            if self.position_size_units is not None:
                self.position_size_recommendation = f"{self.position_size_units} units"
            else:
                self.position_size_recommendation = "Use configured risk-per-trade; no size increase from conviction."
        if not self.reasons:
            self.reasons = [self.rationale]
        if not self.conditions_to_cancel_trade:
            if self.bias == "LONG" and self.stop_loss is not None:
                self.conditions_to_cancel_trade = [f"Cancel/exit if price accepts below {self.stop_loss:.2f}."]
            elif self.bias == "SHORT" and self.stop_loss is not None:
                self.conditions_to_cancel_trade = [f"Cancel/exit if price accepts above {self.stop_loss:.2f}."]
            else:
                self.conditions_to_cancel_trade = ["No trade until confirmation and risk/reward conditions are met."]
        # NEUTRAL/NO_TRADE không cần entry/stop.
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

    @field_validator("confidence", mode="before")
    @classmethod
    def _normalize_decision_confidence(cls, value):
        if isinstance(value, (int, float)) and value > 1:
            return float(value) / 100.0
        return value

    @field_validator(
        "entry", "stop_loss", "take_profit_1", "take_profit",
        "take_profit_2", "rr_ratio", "risk_reward", mode="before"
    )
    @classmethod
    def _coerce_optional_number(cls, value):
        if value is None or isinstance(value, (int, float)):
            return value
        if isinstance(value, dict):
            for key in ("take_profit", "take_profit_1", "tp1", "value", "price"):
                if isinstance(value.get(key), (int, float)):
                    return float(value[key])
            return None
        text = str(value).replace(",", "")
        import re
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(match.group(0)) if match else None


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
    primary_timeframe: str = Field("15m", description="Khung thời gian chính.")
    higher_timeframe: str = Field("1h", description="Khung trend cao hơn.")

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

    # Macro scalar đã extract sẵn (dùng cho Memory Consolidator để
    # tạo narrative gọn cho Graph RAG, KHÔNG phải re-fetch).
    macro_regime: Optional[str] = Field(
        None,
        description=(
            "Regime tag deterministic từ fetch_macro_pulse: "
            "REAL_YIELD_DRIVE / USD_WEAKNESS / RISK_OFF_HAVEN_BID / "
            "GROWTH_SCARE / RISK_ON / RANGE_BOUND."
        ),
    )
    macro_scalars: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "VD: {'dxy_chg_1h': -0.32, 'tnx_chg_1h': 0.05, "
            "'vix_chg_1h': -0.4, 'gold_chg_1h': 0.18}."
        ),
    )

    # ------- Output từ các agent (tích luỹ qua các vòng) -------
    agent_outputs: Annotated[
        Dict[str, AgentOutput], _merge_agent_outputs
    ] = Field(default_factory=dict)

    bull_case: Optional[DebateCase] = None
    bear_case: Optional[DebateCase] = None
    debate_required: bool = False
    debate_reason: str = ""

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
