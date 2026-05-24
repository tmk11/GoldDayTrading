"""Graph RAG Engine — bộ nhớ lịch sử cho các agent.

Module này cung cấp một `GraphRAG` lai giữa hai cấu trúc:

1. **Knowledge Graph (NetworkX)** — lưu các *quan hệ tường minh*
   giữa các thực thể vĩ mô (DXY, US10Y, Gold, VIX, Fed-events,
   v.v.). Mỗi cạnh mang nhãn tương quan ("inverse", "leads_by",
   "regime") và metadata thời điểm. KG cho phép trả lời các câu
   hỏi *cấu trúc* mà thuần vector search không làm được, ví dụ:

       "Liệt kê 5 phiên gần nhất có DXY và US10Y phân kỳ
        (DXY xuống, US10Y lên). Ở mỗi phiên, gold biến động ra sao
        trong 60 phút sau đó?"

2. **Vector Index (ChromaDB persistent client)** — lưu các
   *narrative ngắn* mô tả từng "trading episode" (ví dụ: "Sau CPI
   thấp hơn dự báo 0.2%, DXY -0.4%, gold +1.1% trong 30 phút"),
   embedded bằng **OpenAI text-embedding-3-small** qua API. Vector
   search trả về các episode tương tự về mặt ngữ nghĩa với context
   hiện tại — đó là phần "RAG" thuần tuý.

Hai lớp này phối hợp:

* Macro Agent gọi `query_historical_context(query, k=3)` → ChromaDB
  trả về top-k episode tương tự → mỗi episode mang `node_ids` →
  `expand_via_kg(node_ids)` lấy thêm cạnh KG liên quan → tổng hợp
  thành block markdown đưa vào prompt LLM.

CPU & API-only
--------------

* **KHÔNG** dùng embedding function nội bộ của ChromaDB (vì nó tải
  sentence-transformers / ONNX local). Ta gọi
  `OpenAIEmbeddings.embed_documents(...)` qua HTTPS, đẩy vector
  vào Chroma ở chế độ ``embedding_function=None``.
* NetworkX in-memory + persist sang `pickle` định kỳ. Không cần
  Neo4j hay graph DB ngoài.
* Tất cả I/O vector đi qua `tenacity` retry với backoff exponential
  để chịu được glitch API.

Thư mục mặc định: `~/.golddaytrading/rag/`

* `kg.gpickle`        — NetworkX graph snapshot.
* `chroma/`           — ChromaDB persistent dir.
"""

from __future__ import annotations

import logging
import hashlib
import math
import os
import pickle
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — module chỉ là "soft dependency". Pipeline tuyến tính
# cũ vẫn import được package mà không cần cài extra `agentic`.
# ---------------------------------------------------------------------------


def _import_networkx():
    try:
        import networkx as nx
        return nx
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "GraphRAG cần `networkx`. Cài: pip install -e \".[agentic]\""
        ) from exc


def _import_chroma():
    try:
        import chromadb
        from chromadb.config import Settings
        return chromadb, Settings
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "GraphRAG cần `chromadb`. Cài: pip install -e \".[agentic]\""
        ) from exc


def _import_openai_embeddings():
    """Trả về callable embed(texts) -> list[list[float]] dùng OpenAI API.

    Ưu tiên `langchain_openai.OpenAIEmbeddings` để khớp với phần còn
    lại của workflow agentic. Nếu không có, fallback sang SDK
    `openai` thuần.
    """
    try:
        from langchain_openai import OpenAIEmbeddings
        return ("langchain", OpenAIEmbeddings)
    except ImportError:
        pass
    try:
        from openai import OpenAI
        return ("openai", OpenAI)
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Cần `langchain-openai` hoặc `openai`. "
            "Cài: pip install -e \".[agentic]\""
        ) from exc


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HistoricalEpisode:
    """Một 'trading episode' lịch sử để retrieve về sau.

    Một episode mô tả ngắn gọn một tình huống vĩ mô + phản ứng của
    gold, được embed và lưu trong vector store. Các `entity_nodes`
    là id của các node KG liên quan để expansion.
    """

    episode_id: str
    timestamp: datetime
    narrative: str                      # text được embed
    entity_nodes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    """metadata vd: {'regime': 'REAL_YIELD_DRIVE', 'dxy_chg_1h': 0.32,
    'gold_chg_1h': -0.45, 'event': 'CPI_release'}.
    Chỉ giá trị primitive (str/int/float/bool) để Chroma chấp nhận."""


@dataclass
class CorrelationEdge:
    """Một quan hệ tường minh giữa hai thực thể vĩ mô."""

    source: str                         # node id, vd "DXY"
    target: str                         # node id, vd "GOLD"
    relation: str                       # vd: "inverse", "leads_by_30m"
    weight: float = 1.0                 # mức độ tin cậy / cường độ
    observed_at: Optional[datetime] = None
    note: str = ""


@dataclass
class RAGQueryResult:
    """Output của `query_historical_context` — sẵn sàng đưa vào prompt."""

    episodes: List[HistoricalEpisode]
    related_edges: List[CorrelationEdge]
    rendered_block: str                 # markdown sẵn cho LLM


# ---------------------------------------------------------------------------
# Embedder wrapper (API-only)
# ---------------------------------------------------------------------------


class _OpenAIEmbedder:
    """Wrapper mỏng quanh OpenAI Embeddings API.

    Chỉ làm hai việc: (1) chuẩn hoá interface về `embed(texts)`,
    (2) thêm tenacity retry. Không cache cục bộ — Chroma đã giữ
    vector rồi nên không cần cache thêm.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = os.environ.get("GDT_EMBEDDING_MODEL", model)
        kind, factory = _import_openai_embeddings()
        self._kind = kind
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        api_key = os.environ.get("OPENAI_API_KEY") or None
        if kind == "langchain":
            kwargs = {"model": self.model}
            if base_url:
                kwargs["base_url"] = base_url
            if api_key:
                kwargs["api_key"] = api_key
            self._impl = factory(**kwargs)
        else:
            # openai SDK thuần
            kwargs = {}
            if base_url:
                kwargs["base_url"] = base_url
            if api_key:
                kwargs["api_key"] = api_key
            self._impl = factory(**kwargs)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            return _retry_embed(self._kind, self._impl, self.model, list(texts))
        except Exception as exc:
            if os.environ.get("GDT_ENABLE_HASH_EMBEDDING_FALLBACK", "1") != "1":
                raise
            logger.warning(
                "Embedding API failed (%s); using deterministic hash embeddings.",
                exc,
            )
            return [_hash_embedding(text) for text in texts]

def _hash_embedding(text: str, dimensions: int = 1536) -> List[float]:
    vector = [0.0] * dimensions
    tokens = text.lower().split()
    if not tokens:
        tokens = [text.lower() or "empty"]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _retry_embed(kind: str, impl, model: str, texts: List[str]) -> List[List[float]]:
    """Gọi embedding API với retry. Tách thành function để test mock."""
    try:
        from tenacity import (
            retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
        )
    except ImportError:  # pragma: no cover
        retry = None  # type: ignore

    def _do_call() -> List[List[float]]:
        if kind == "langchain":
            return impl.embed_documents(texts)
        # openai SDK
        rsp = impl.embeddings.create(model=model, input=texts)
        return [d.embedding for d in rsp.data]

    if retry is None:
        return _do_call()

    decorated = retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )(_do_call)
    return decorated()


# ---------------------------------------------------------------------------
# Graph RAG engine
# ---------------------------------------------------------------------------


class GraphRAG:
    """Bộ nhớ lai: Knowledge Graph (NetworkX) + Vector store (Chroma).

    Thread-safe ở mức cơ bản qua một RLock: được thiết kế cho 1
    process orchestrator, không phải cho high-concurrency server.

    Sử dụng tiêu biểu
    -----------------

        rag = GraphRAG.default()
        rag.ingest_correlation(CorrelationEdge("DXY", "GOLD", "inverse"))
        rag.ingest_episode(HistoricalEpisode(
            episode_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            narrative="DXY -0.4% sau CPI thấp hơn dự báo, gold +1.1%.",
            entity_nodes=["DXY", "GOLD", "CPI_release"],
            metadata={"regime": "USD_WEAKNESS"},
        ))
        result = rag.query_historical_context(
            "Gold phản ứng thế nào khi DXY và US10Y phân kỳ?",
            k=3,
        )
        prompt += "\\n\\n" + result.rendered_block
    """

    DEFAULT_DIR = os.path.join(os.path.expanduser("~"), ".golddaytrading", "rag")
    KG_FILE = "kg.gpickle"
    CHROMA_SUBDIR = "chroma"
    CHROMA_COLLECTION = "gold_episodes"

    # ----------------------------------------------------------------
    # Construction
    # ----------------------------------------------------------------

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        embedder: Optional[_OpenAIEmbedder] = None,
    ):
        self.persist_dir = persist_dir or self.DEFAULT_DIR
        os.makedirs(self.persist_dir, exist_ok=True)

        self._lock = threading.RLock()
        self._nx = _import_networkx()
        self.kg = self._load_kg()

        chromadb, Settings = _import_chroma()
        self._chroma_client = chromadb.PersistentClient(
            path=os.path.join(self.persist_dir, self.CHROMA_SUBDIR),
            settings=Settings(anonymized_telemetry=False, allow_reset=False),
        )
        # embedding_function=None: ta tự đẩy vector lên (API-only).
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.CHROMA_COLLECTION,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

        self._embedder = embedder or _OpenAIEmbedder(embedding_model)

    @classmethod
    def default(cls) -> "GraphRAG":
        return cls(persist_dir=os.environ.get("GDT_RAG_DIR") or None)

    # ----------------------------------------------------------------
    # Knowledge Graph layer
    # ----------------------------------------------------------------

    def _load_kg(self):
        path = os.path.join(self.persist_dir, self.KG_FILE)
        if os.path.exists(path):
            try:
                with open(path, "rb") as fh:
                    return pickle.load(fh)
            except Exception as exc:  # pragma: no cover
                logger.warning("KG corrupt (%s) — bắt đầu graph mới.", exc)
        g = self._nx.MultiDiGraph()
        self._seed_kg(g)
        return g

    def _seed_kg(self, g) -> None:
        """Khởi tạo KG với các quan hệ vĩ mô cơ bản đã biết.

        Đây là *prior knowledge* mà các agent có thể tin cậy ngay
        cả khi chưa có episode lịch sử nào được ingest. Nguồn:
        các bài nghiên cứu của World Gold Council, Fed, BIS — kiến
        thức phổ thông không cần citation.
        """
        nodes = {
            "GOLD": {"type": "asset", "label": "Gold spot (XAU/USD)"},
            "DXY": {"type": "macro", "label": "US Dollar Index"},
            "US10Y": {"type": "macro", "label": "US 10Y nominal yield"},
            "US2Y": {"type": "macro", "label": "US 2Y nominal yield"},
            "REAL_YIELD": {"type": "derived", "label": "Real yield proxy (TIP inverse)"},
            "VIX": {"type": "macro", "label": "Equity vol index"},
            "ES": {"type": "macro", "label": "S&P 500 futures"},
            "EURUSD": {"type": "macro", "label": "EUR/USD"},
            "BTC": {"type": "macro", "label": "Bitcoin"},
            "SILVER": {"type": "asset", "label": "Silver futures"},
            "CPI_release": {"type": "event", "label": "US CPI print"},
            "NFP_release": {"type": "event", "label": "US Non-Farm Payrolls"},
            "FOMC_release": {"type": "event", "label": "FOMC decision/minutes"},
        }
        for nid, attrs in nodes.items():
            if nid not in g:
                g.add_node(nid, **attrs)

        priors: List[Tuple[str, str, str, float, str]] = [
            ("DXY", "GOLD", "inverse", 0.85, "Stronger USD ⇒ gold weaker (denominator)."),
            ("US10Y", "GOLD", "inverse", 0.65, "Higher nominal yield ⇒ higher opportunity cost."),
            ("REAL_YIELD", "GOLD", "inverse", 0.90, "Real yields là driver chính của gold post-2018."),
            ("VIX", "GOLD", "positive", 0.55, "Risk-off ⇒ safe-haven bid."),
            ("ES", "GOLD", "weak_inverse", 0.30, "Risk-on equity tape có thể là headwind nhẹ."),
            ("EURUSD", "DXY", "inverse", 0.95, "EURUSD ~57% trọng số DXY."),
            ("CPI_release", "REAL_YIELD", "drives", 0.70, "Số liệu CPI tác động kỳ vọng real yield."),
            ("FOMC_release", "DXY", "drives", 0.75, "FOMC định giá lại USD và toàn curve."),
            ("NFP_release", "US10Y", "drives", 0.65, "Việc làm mạnh ⇒ yield lên."),
            ("SILVER", "GOLD", "leads_late_cycle", 0.50, "Silver thường lead gold giai đoạn cuối rally."),
        ]
        for src, tgt, rel, w, note in priors:
            g.add_edge(src, tgt, relation=rel, weight=w, note=note,
                       observed_at=None, source="prior")

    def _persist_kg(self) -> None:
        path = os.path.join(self.persist_dir, self.KG_FILE)
        try:
            with open(path, "wb") as fh:
                pickle.dump(self.kg, fh)
        except OSError as exc:  # pragma: no cover
            logger.warning("Không lưu được KG (%s).", exc)

    def ingest_correlation(self, edge: CorrelationEdge) -> None:
        """Thêm/cập nhật một quan hệ vào KG.

        Nếu edge cùng (source, target, relation) đã tồn tại, cập
        nhật `weight` bằng running average và ghi đè `observed_at`.
        """
        with self._lock:
            g = self.kg
            for nid in (edge.source, edge.target):
                if nid not in g:
                    g.add_node(nid, type="auto", label=nid)

            existing = None
            for _, _, key, data in g.edges(
                edge.source, edge.target, keys=True, data=True
            ):
                if data.get("relation") == edge.relation:
                    existing = (key, data)
                    break

            if existing:
                key, data = existing
                old_w = float(data.get("weight", 1.0))
                new_w = (old_w + edge.weight) / 2.0
                data["weight"] = new_w
                data["observed_at"] = (edge.observed_at or datetime.now(timezone.utc))
                if edge.note:
                    data["note"] = edge.note
            else:
                g.add_edge(
                    edge.source, edge.target,
                    relation=edge.relation,
                    weight=edge.weight,
                    note=edge.note,
                    observed_at=edge.observed_at or datetime.now(timezone.utc),
                    source="ingested",
                )
            self._persist_kg()

    def ingest_correlations(self, edges: Iterable[CorrelationEdge]) -> int:
        n = 0
        for e in edges:
            self.ingest_correlation(e)
            n += 1
        return n

    def neighbours(self, node_id: str, depth: int = 1) -> List[CorrelationEdge]:
        """Trả về các cạnh kề (đi/đến) trong bán kính `depth`."""
        with self._lock:
            g = self.kg
            if node_id not in g:
                return []
            visited: set[str] = {node_id}
            frontier: List[str] = [node_id]
            edges: List[CorrelationEdge] = []
            for _ in range(max(1, depth)):
                next_frontier: List[str] = []
                for n in frontier:
                    for u, v, data in g.out_edges(n, data=True):
                        edges.append(self._edge_from_data(u, v, data))
                        if v not in visited:
                            visited.add(v)
                            next_frontier.append(v)
                    for u, v, data in g.in_edges(n, data=True):
                        edges.append(self._edge_from_data(u, v, data))
                        if u not in visited:
                            visited.add(u)
                            next_frontier.append(u)
                frontier = next_frontier
            return edges

    @staticmethod
    def _edge_from_data(u: str, v: str, data: dict) -> CorrelationEdge:
        return CorrelationEdge(
            source=u, target=v,
            relation=str(data.get("relation", "related")),
            weight=float(data.get("weight", 1.0)),
            observed_at=data.get("observed_at"),
            note=str(data.get("note", "")),
        )

    # ----------------------------------------------------------------
    # Vector layer (episodes)
    # ----------------------------------------------------------------

    def ingest_episode(self, episode: HistoricalEpisode) -> str:
        """Embed narrative qua API và đẩy vào ChromaDB.

        Cũng tự thêm các `entity_nodes` vào KG nếu chưa tồn tại
        (làm KG tự nở dần theo dữ liệu mới).
        """
        with self._lock:
            # 1. Đảm bảo node tồn tại trong KG
            for nid in episode.entity_nodes:
                if nid not in self.kg:
                    self.kg.add_node(nid, type="auto", label=nid)

            # 2. Embed qua API
            vec = self._embedder.embed([episode.narrative])[0]

            # 3. Sanitize metadata cho Chroma (chỉ primitive)
            meta = {"timestamp": episode.timestamp.isoformat()}
            for k, v in (episode.metadata or {}).items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    meta[k] = v
                else:
                    meta[k] = str(v)
            if episode.entity_nodes:
                # Chroma metadata không nhận list; serialize "|"-joined.
                meta["entity_nodes"] = "|".join(episode.entity_nodes)

            self._collection.upsert(
                ids=[episode.episode_id],
                embeddings=[vec],
                documents=[episode.narrative],
                metadatas=[meta],
            )
            self._persist_kg()
            return episode.episode_id

    def ingest_episodes(self, episodes: Sequence[HistoricalEpisode]) -> int:
        if not episodes:
            return 0
        with self._lock:
            # Batch embed để tiết kiệm API calls
            vectors = self._embedder.embed([ep.narrative for ep in episodes])
            ids, docs, metas = [], [], []
            for ep, vec in zip(episodes, vectors):
                for nid in ep.entity_nodes:
                    if nid not in self.kg:
                        self.kg.add_node(nid, type="auto", label=nid)
                meta = {"timestamp": ep.timestamp.isoformat()}
                for k, v in (ep.metadata or {}).items():
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        meta[k] = v
                    else:
                        meta[k] = str(v)
                if ep.entity_nodes:
                    meta["entity_nodes"] = "|".join(ep.entity_nodes)
                ids.append(ep.episode_id)
                docs.append(ep.narrative)
                metas.append(meta)
            # Upsert thẳng kèm vector đã có sẵn
            self._collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=docs,
                metadatas=metas,
            )
            self._persist_kg()
            return len(episodes)

    # ----------------------------------------------------------------
    # Retrieval — interface chính cho các tool
    # ----------------------------------------------------------------

    def query_historical_context(
        self,
        query: str,
        k: int = 3,
        regime_filter: Optional[str] = None,
        kg_expand_depth: int = 1,
    ) -> RAGQueryResult:
        """Trả về top-k episode giống ngữ nghĩa + cạnh KG liên quan.

        Đây là tool mà Macro Agent sẽ gọi để trả lời các câu như:
        "Gold phản ứng ra sao 3 lần gần nhất khi DXY và US10Y phân kỳ?"

        :param query:           Câu hỏi tự nhiên.
        :param k:               Số episode trả về.
        :param regime_filter:   Nếu truyền, chỉ lọc episode có
                                metadata['regime'] == regime_filter.
        :param kg_expand_depth: Bán kính BFS quanh entity_nodes của
                                các episode để bổ sung cạnh KG.
        """
        with self._lock:
            try:
                qvec = self._embedder.embed([query])[0]
            except Exception as exc:
                logger.warning("Embedding query lỗi: %s", exc)
                return RAGQueryResult([], [], _empty_block(query))

            where: Optional[Dict[str, Any]] = None
            if regime_filter:
                where = {"regime": regime_filter}

            try:
                res = self._collection.query(
                    query_embeddings=[qvec],
                    n_results=max(1, k),
                    where=where,
                )
            except Exception as exc:
                logger.warning("Chroma query lỗi: %s", exc)
                return RAGQueryResult([], [], _empty_block(query))

            episodes = _decode_chroma_result(res)

            related_edges: List[CorrelationEdge] = []
            seen_edge_keys: set[Tuple[str, str, str]] = set()
            for ep in episodes:
                for nid in ep.entity_nodes:
                    for e in self.neighbours(nid, depth=kg_expand_depth):
                        key = (e.source, e.target, e.relation)
                        if key in seen_edge_keys:
                            continue
                        seen_edge_keys.add(key)
                        related_edges.append(e)

            block = _render_block(query, episodes, related_edges)
            return RAGQueryResult(episodes, related_edges, block)

    # ----------------------------------------------------------------
    # Stats / debug
    # ----------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            try:
                count = self._collection.count()
            except Exception:
                count = -1
            return {
                "kg_nodes": self.kg.number_of_nodes(),
                "kg_edges": self.kg.number_of_edges(),
                "vector_episodes": count,
                "persist_dir": self.persist_dir,
            }

    # ----------------------------------------------------------------
    # Visibility helpers (cho CLI `gdt agentic-info`)
    # ----------------------------------------------------------------

    def detailed_stats(self, top_k: int = 10) -> Dict[str, Any]:
        """Phiên bản chi tiết của :meth:`stats` cho CLI.

        Trả về dict gồm:

        * ``kg_nodes`` / ``kg_edges`` (tổng).
        * ``kg_node_types``: dict {type → count} (asset / macro /
          event / derived / auto).
        * ``kg_relation_types``: dict {relation → count} (inverse /
          drives / leads_by_30m / …).
        * ``kg_top_connected``: ``[(node_id, degree), …]`` top
          ``top_k`` node có degree lớn nhất.
        * ``vector_episodes``: tổng số episode trong Chroma.
        * ``persist_dir``: thư mục persist.
        """
        with self._lock:
            g = self.kg

            # Type breakdown
            type_counts: Dict[str, int] = {}
            for _nid, attrs in g.nodes(data=True):
                t = str(attrs.get("type", "unknown"))
                type_counts[t] = type_counts.get(t, 0) + 1

            # Relation breakdown
            rel_counts: Dict[str, int] = {}
            for _u, _v, data in g.edges(data=True):
                r = str(data.get("relation", "?"))
                rel_counts[r] = rel_counts.get(r, 0) + 1

            # Top-connected (degree = in + out vì graph có hướng)
            degrees = [(n, int(d)) for n, d in g.degree()]
            degrees.sort(key=lambda x: -x[1])
            top_connected = degrees[: max(1, top_k)]

            # Vector count
            try:
                vector_count = int(self._collection.count())
            except Exception as exc:  # pragma: no cover
                logger.warning("Chroma count lỗi: %s", exc)
                vector_count = -1

            return {
                "kg_nodes": g.number_of_nodes(),
                "kg_edges": g.number_of_edges(),
                "kg_node_types": type_counts,
                "kg_relation_types": rel_counts,
                "kg_top_connected": top_connected,
                "vector_episodes": vector_count,
                "persist_dir": self.persist_dir,
            }

    def list_recent_episodes(self, limit: int = 10) -> List[HistoricalEpisode]:
        """Trả về N episode mới nhất, sắp xếp giảm dần theo timestamp.

        Chroma 0.5 không hỗ trợ "ORDER BY timestamp" trực tiếp; ta
        dùng :meth:`Collection.get` để lấy toàn bộ rồi sort trong
        Python. Với volume vừa phải (< 100k episode) cách này đủ
        nhanh và đỡ phải maintain index riêng.
        """
        with self._lock:
            limit = max(1, int(limit))
            try:
                res = self._collection.get(
                    include=["documents", "metadatas"],
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Chroma get lỗi: %s", exc)
                return []

            # Schema khác `query()`: ids/documents/metadatas KHÔNG
            # nested theo query rows mà flat.
            ids = res.get("ids") or []
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []

            episodes: List[HistoricalEpisode] = []
            for i, eid in enumerate(ids):
                meta = metas[i] if i < len(metas) and metas[i] else {}
                narrative = docs[i] if i < len(docs) else ""
                ts_raw = meta.get("timestamp")
                try:
                    ts = (
                        datetime.fromisoformat(ts_raw)
                        if ts_raw else datetime.now(timezone.utc)
                    )
                except (TypeError, ValueError):
                    ts = datetime.now(timezone.utc)
                ents_raw = meta.get("entity_nodes") or ""
                ents = [s for s in str(ents_raw).split("|") if s]
                clean_meta = {
                    k: v for k, v in meta.items()
                    if k not in ("timestamp", "entity_nodes")
                }
                episodes.append(HistoricalEpisode(
                    episode_id=str(eid),
                    timestamp=ts,
                    narrative=narrative,
                    entity_nodes=ents,
                    metadata=clean_meta,
                ))

            episodes.sort(key=lambda e: e.timestamp, reverse=True)
            return episodes[:limit]


# ---------------------------------------------------------------------------
# Helpers cấp module
# ---------------------------------------------------------------------------


def _decode_chroma_result(res: Dict[str, Any]) -> List[HistoricalEpisode]:
    """Đổi schema dict từ Chroma sang list[HistoricalEpisode]."""
    out: List[HistoricalEpisode] = []
    if not res:
        return out
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    for i, eid in enumerate(ids):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        narrative = docs[i] if i < len(docs) else ""
        ts_raw = meta.get("timestamp")
        try:
            ts = datetime.fromisoformat(ts_raw) if ts_raw else datetime.now(timezone.utc)
        except (TypeError, ValueError):
            ts = datetime.now(timezone.utc)
        ents_raw = meta.get("entity_nodes") or ""
        ents = [s for s in str(ents_raw).split("|") if s]
        # Loại các meta-keys nội bộ khi trả lại HistoricalEpisode.metadata
        clean_meta = {
            k: v for k, v in meta.items()
            if k not in ("timestamp", "entity_nodes")
        }
        out.append(HistoricalEpisode(
            episode_id=str(eid),
            timestamp=ts,
            narrative=narrative,
            entity_nodes=ents,
            metadata=clean_meta,
        ))
    return out


def _empty_block(query: str) -> str:
    return (
        "### Bộ nhớ lịch sử (Graph RAG)\n"
        f"_Không có episode tương tự nào cho câu truy vấn:_ `{query}`.\n"
    )


def _render_block(
    query: str,
    episodes: List[HistoricalEpisode],
    edges: List[CorrelationEdge],
) -> str:
    """Render kết quả RAG thành block markdown sẵn cho prompt."""
    lines: List[str] = ["### Bộ nhớ lịch sử (Graph RAG)"]
    lines.append(f"_Truy vấn:_ `{query}`")
    if not episodes:
        lines.append("")
        lines.append("_Không tìm thấy tình huống quá khứ tương tự._")
        return "\n".join(lines) + "\n"

    lines.append("")
    lines.append(f"**Top {len(episodes)} tình huống quá khứ tương tự:**")
    for i, ep in enumerate(episodes, 1):
        ts = ep.timestamp.strftime("%Y-%m-%d %H:%M UTC")
        regime = ep.metadata.get("regime") or "—"
        lines.append(
            f"{i}. _[{ts}, regime=`{regime}`]_ {ep.narrative}"
        )

    if edges:
        # Khử trùng lặp + cắt bớt cho gọn
        uniq: Dict[Tuple[str, str, str], CorrelationEdge] = {}
        for e in edges:
            uniq[(e.source, e.target, e.relation)] = e
        top = sorted(uniq.values(), key=lambda e: -e.weight)[:8]

        lines.append("")
        lines.append("**Quan hệ KG liên quan (trọng số ≥):**")
        for e in top:
            lines.append(
                f"- `{e.source}` —[{e.relation}, w={e.weight:.2f}]→ `{e.target}`"
                + (f"  _({e.note})_" if e.note else "")
            )

    lines.append("")
    lines.append(
        "_Hướng dẫn sử dụng:_ coi đây là **prior** Bayesian, không "
        "phải dự báo. Nếu setup hiện tại không giống bất kỳ episode "
        "nào ở trên, hãy nói rõ điều đó trong rationale."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------


def build_episode_from_macro_pulse(
    pulse: Dict[str, Any],
    gold_chg_1h: Optional[float],
    note: str = "",
) -> HistoricalEpisode:
    """Tiện ích: tạo `HistoricalEpisode` từ output `fetch_macro_pulse`.

    Dùng để batch-ingest lịch sử từ pipeline cũ vào RAG.
    """
    regime = pulse.get("__regime__") or "RANGE_BOUND"
    dxy = (pulse.get("DX-Y.NYB") or {}).get("chg_1h")
    tnx = (pulse.get("^TNX") or {}).get("chg_1h")
    vix = (pulse.get("^VIX") or {}).get("chg_1h")

    parts: List[str] = []
    if dxy is not None:
        parts.append(f"DXY 1h Δ {dxy:+.2f}%")
    if tnx is not None:
        parts.append(f"US10Y 1h Δ {tnx:+.2f}%")
    if vix is not None:
        parts.append(f"VIX 1h Δ {vix:+.2f}%")
    if gold_chg_1h is not None:
        parts.append(f"GOLD 1h Δ {gold_chg_1h:+.2f}%")
    narrative = (
        f"[{regime}] " + ", ".join(parts) +
        (f" — {note}" if note else ".")
    )

    metadata: Dict[str, Any] = {"regime": regime}
    if dxy is not None: metadata["dxy_chg_1h"] = float(dxy)
    if tnx is not None: metadata["tnx_chg_1h"] = float(tnx)
    if vix is not None: metadata["vix_chg_1h"] = float(vix)
    if gold_chg_1h is not None: metadata["gold_chg_1h"] = float(gold_chg_1h)

    return HistoricalEpisode(
        episode_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        narrative=narrative,
        entity_nodes=["DXY", "US10Y", "VIX", "GOLD"],
        metadata=metadata,
    )
