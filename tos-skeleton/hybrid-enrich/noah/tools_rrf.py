#!/usr/bin/env python3
"""RRF 에이전트 도구. in-process Search 클래스 + CLI 인터페이스 (claude -p 연동용).

모든 데이터는 이 폴더의 out/ 에서 읽는다 (bak/ 참조 없음).
  뷰      : out/view_{BASE,V6,V9}.jsonl   {"chunk_id","text"}
  덴스벡터: out/vec_{VIEW}.npy (L2 정규화) + out/vec_{VIEW}_ids.json
  원문청크: out/chunks.jsonl              (프리뷰/read 는 항상 원문)
  엘리먼트: out/elements_psection.jsonl

검색 채널
  bm25/{VIEW}  : Okapi BM25 (k1=1.5, b=0.75), 역색인 postings, pickle 캐시
  dense/{VIEW} : ollama bge-m3 질의 임베딩 × 정규화 행렬 내적

암(arm)별 채널 구성은 ARMS 참조. 융합은 RRF k=60.
"""
from __future__ import annotations

import json
import math
import os
import pickle
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

# 코드는 noah/ 에, 데이터는 그 상위 hybrid-enrich/out 에 있다.
HERE = Path(__file__).resolve().parent   # noah/ — 코드 위치
BASE = HERE.parent                       # hybrid-enrich/ — 데이터 루트
OUT = BASE / "out"

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── 설정 ────────────────────────────────────────────────────────────────
MAX_CALLS = 8
RRF_K = 60
CHANNEL_TOPK = 50
FINAL_TOPK = 10
PREVIEW_CHARS = 240

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "bge-m3"

BM25_PICKLE_VERSION = 2

ARMS: dict[str, dict] = {
    # arm        : 채널 목록 [(kind, view), ...]                          clm 도구 노출  slot_and(AND 슬롯) 도구 노출
    "BASE":      {"channels": [("bm25", "BASE"), ("dense", "BASE")], "clm": False, "slot_and": False},
    "V6":        {"channels": [("bm25", "V6"), ("dense", "V6")], "clm": False, "slot_and": False},
    "V9":        {"channels": [("bm25", "V9"), ("dense", "V9")], "clm": False, "slot_and": False},
    # v9 메타 단독 + CLM 태그. v6 를 아직 전량 생성하지 않은 단계에서
    # "메타 + 태그" 조합을 측정하기 위한 arm.
    "V9_CLM":    {"channels": [("bm25", "V9"), ("dense", "V9")], "clm": True, "slot_and": False},
    "BASE_CLM":  {"channels": [("bm25", "BASE"), ("dense", "BASE")], "clm": True, "slot_and": False},
    "V6V9":      {"channels": [("bm25", "V6"), ("dense", "V6"),
                               ("bm25", "V9"), ("dense", "V9")], "clm": False, "slot_and": False},
    "V6V9_CLM":  {"channels": [("bm25", "V6"), ("dense", "V6"),
                               ("bm25", "V9"), ("dense", "V9")], "clm": True, "slot_and": False},
    # AND-slot 검색기(SlotFileSearch v2) 를 CLM 대신 노출하는 arm.
    # retriever axis 비교(AND vs CLM-union)를 위한 것으로, 기본 arm 세트에는 포함하지 않는다.
    "V9_SLOT":   {"channels": [("bm25", "V9"), ("dense", "V9")], "clm": False, "slot_and": True},
    "BASE_SLOT": {"channels": [("bm25", "BASE"), ("dense", "BASE")], "clm": False, "slot_and": True},
}

_KO_TOKEN_RE = re.compile(r"[가-힣a-zA-Z0-9]+")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _tokenize(text: str) -> list[str]:
    return _KO_TOKEN_RE.findall(text.lower())


def view_path(view: str) -> Path:
    return OUT / f"view_{view}.jsonl"


# ── BM25 ────────────────────────────────────────────────────────────────
class BM25:
    """Okapi BM25. 역색인(postings) 기반 — 질의어를 포함한 문서만 채점한다."""

    def __init__(self, docs: list[tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = [d[0] for d in docs]
        self.n = len(docs)
        self.doc_lens: list[int] = []
        # term -> [(doc_idx, tf), ...]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for idx, (_, text) in enumerate(docs):
            terms = _tokenize(text)
            self.doc_lens.append(len(terms))
            for term, tf in Counter(terms).items():
                self.postings[term].append((idx, tf))
        self.postings = dict(self.postings)
        self.avgdl = sum(self.doc_lens) / max(self.n, 1)

    def search(self, query: str, top_k: int = CHANNEL_TOPK) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        k1, b, avgdl = self.k1, self.b, self.avgdl
        for term in _tokenize(query):
            plist = self.postings.get(term)
            if not plist:
                continue
            df = len(plist)
            idf = math.log((self.n - df + 0.5) / (df + 0.5) + 1)
            for idx, tf in plist:
                dl = self.doc_lens[idx]
                scores[idx] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [(self.ids[i], round(sc, 4)) for i, sc in top if sc > 0]

    # pickle 캐시 ---------------------------------------------------------
    def to_state(self) -> dict:
        return {"v": BM25_PICKLE_VERSION, "k1": self.k1, "b": self.b, "ids": self.ids,
                "n": self.n, "doc_lens": self.doc_lens, "avgdl": self.avgdl,
                "postings": self.postings}

    @classmethod
    def from_state(cls, state: dict) -> "BM25":
        obj = cls.__new__(cls)
        obj.k1, obj.b = state["k1"], state["b"]
        obj.ids, obj.n = state["ids"], state["n"]
        obj.doc_lens, obj.avgdl = state["doc_lens"], state["avgdl"]
        obj.postings = state["postings"]
        return obj


def build_or_load_bm25(view: str) -> tuple[BM25 | None, str]:
    """뷰 인덱스를 pickle 캐시에서 로드하거나 새로 만든다. (index, source) 반환."""
    src = view_path(view)
    if not src.exists():
        return None, "missing"
    pkl = OUT / f"bm25_{view}.pkl"
    if pkl.exists() and pkl.stat().st_mtime >= src.stat().st_mtime:
        try:
            with pkl.open("rb") as handle:
                state = pickle.load(handle)
            if state.get("v") == BM25_PICKLE_VERSION:
                return BM25.from_state(state), "cache"
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] bm25 캐시 로드 실패 ({view}): {exc}", file=sys.stderr)
    rows = load_jsonl(src)
    idx = BM25([(r["chunk_id"], r.get("text", "")) for r in rows])
    try:
        with pkl.open("wb") as handle:
            pickle.dump(idx.to_state(), handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] bm25 캐시 저장 실패 ({view}): {exc}", file=sys.stderr)
    return idx, "build"


# ── 덴스 (ollama bge-m3) ────────────────────────────────────────────────
def embed_query(query: str):
    import numpy as np
    payload = json.dumps({"model": EMBED_MODEL, "input": [query],
                          "keep_alive": "30m"}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        vec = np.asarray(json.load(resp)["embeddings"][0], dtype=np.float32)
    return vec / (float(np.linalg.norm(vec)) or 1.0)


# ── Search ──────────────────────────────────────────────────────────────
class Search:
    """모든 인덱스를 지연 로딩한다 (프로세스가 호출마다 새로 뜨므로 콜드스타트 최소화)."""

    def __init__(self, arm: str = "V6V9"):
        self.arm = arm if arm in ARMS else "V6V9"
        self._bm25: dict[str, BM25 | None] = {}
        self._views: dict[str, list[dict]] = {}
        self._chunks: dict[str, str] | None = None
        self._elements: dict[str, str] | None = None
        self._clm = None
        self._clm_loaded = False
        self._slot_and = None
        self._slot_and_loaded = False
        self.notes: list[str] = []

    # 자원 -----------------------------------------------------------------
    def bm25(self, view: str) -> BM25 | None:
        if view not in self._bm25:
            idx, src = build_or_load_bm25(view)
            if idx is None:
                self.notes.append(f"bm25/{view}: 뷰 파일 없음 ({view_path(view).name})")
            self._bm25[view] = idx
        return self._bm25[view]

    def view_rows(self, view: str) -> list[dict]:
        if view not in self._views:
            path = view_path(view)
            self._views[view] = load_jsonl(path) if path.exists() else []
        return self._views[view]

    @property
    def chunks(self) -> dict[str, str]:
        if self._chunks is None:
            path = OUT / "chunks.jsonl"
            self._chunks = {r["chunk_id"]: r["text"] for r in load_jsonl(path)} if path.exists() else {}
        return self._chunks

    @property
    def elements(self) -> dict[str, str]:
        if self._elements is None:
            path = OUT / "elements_psection.jsonl"
            self._elements = {r["element_id"]: r["text"] for r in load_jsonl(path)} if path.exists() else {}
        return self._elements

    @property
    def clm(self):
        if not self._clm_loaded:
            self._clm_loaded = True
            try:
                from clm_filesearch import CLMFileSearch
                self._clm = CLMFileSearch()
            except Exception as exc:  # noqa: BLE001
                self._clm_err = f"clm unavailable: {exc}"
                self._clm = None
        return self._clm

    @property
    def slot_and(self):
        if not self._slot_and_loaded:
            self._slot_and_loaded = True
            try:
                from slot_filesearch import SlotFileSearch
                self._slot_and = SlotFileSearch(variant="v2")
            except Exception as exc:  # noqa: BLE001
                self._slot_and_err = f"slot_filesearch unavailable: {exc}"
                self._slot_and = None
        return self._slot_and

    def preview(self, uid: str) -> str:
        return (self.chunks.get(uid) or self.elements.get(uid) or "")[:PREVIEW_CHARS]

    # 채널 -----------------------------------------------------------------
    def bm25_channel(self, query: str, view: str) -> list[str]:
        idx = self.bm25(view)
        if idx is None:
            return []
        return [uid for uid, _ in idx.search(query, top_k=CHANNEL_TOPK)]

    def dense_channel(self, query: str, view: str) -> list[str]:
        """실패 시 빈 채널 + note. 절대 예외를 올리지 않는다."""
        npy = OUT / f"vec_{view}.npy"
        ids_path = OUT / f"vec_{view}_ids.json"
        if not npy.exists() or not ids_path.exists():
            self.notes.append(f"dense/{view}: 벡터 없음 ({npy.name}) — 채널 생략")
            return []
        try:
            import numpy as np
            matrix = np.load(npy, mmap_mode="r")
            ids = json.loads(ids_path.read_text(encoding="utf-8"))
            if isinstance(ids, dict):  # 구형 포맷 호환
                ids = ids.get("ids", [])
            qv = embed_query(query)
            scores = np.asarray(matrix @ qv)
            k = min(CHANNEL_TOPK, len(scores))
            top = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
            top = top[np.argsort(-scores[top])]
            return [ids[int(i)] for i in top if int(i) < len(ids)]
        except Exception as exc:  # noqa: BLE001
            self.notes.append(f"dense/{view}: 실패 ({exc}) — 채널 생략")
            return []

    # 헤드라인 도구 ---------------------------------------------------------
    def hybrid_search(self, query: str, arm: str | None = None) -> dict:
        """arm 의 모든 채널에서 top-50 → RRF(k=60) 융합 → top-10.

        결과 항목마다 채널별 순위(channels)를 붙여 혼합이 실제로 일어났음을 노출한다.
        """
        arm = arm if arm in ARMS else self.arm
        t0 = time.time()
        fused: dict[str, float] = defaultdict(float)
        provenance: dict[str, dict[str, int]] = defaultdict(dict)
        channel_sizes: dict[str, int] = {}

        for kind, view in ARMS[arm]["channels"]:
            label = f"{kind}/{view}"
            hits = self.bm25_channel(query, view) if kind == "bm25" else self.dense_channel(query, view)
            channel_sizes[label] = len(hits)
            for rank, uid in enumerate(hits, 1):
                fused[uid] += 1.0 / (RRF_K + rank)
                provenance[uid][label] = rank

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:FINAL_TOPK]
        results = [{"id": uid, "rrf": round(score, 6),
                    "channels": provenance[uid], "preview": self.preview(uid)}
                   for uid, score in ranked]
        out: dict = {"arm": arm, "query": query, "rrf_k": RRF_K,
                     "channels": channel_sizes, "results": results,
                     "elapsed_ms": int((time.time() - t0) * 1000)}
        if self.notes:
            out["note"] = "; ".join(dict.fromkeys(self.notes))
        return out

    # 보조 도구 -------------------------------------------------------------
    def slot(self, query: str, slots: dict | None = None, limit: int = 20) -> dict:
        if self.clm is None:
            return {"error": getattr(self, "_clm_err", "clm unavailable")}
        try:
            return self.clm.search(query, slots=slots, limit=limit)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"clm search failed: {exc}"}

    def slotand(self, filters: dict | None = None, raw_regex: str = "", limit: int = 20) -> dict:
        if self.slot_and is None:
            return {"error": getattr(self, "_slot_and_err", "slot_filesearch unavailable")}
        try:
            return self.slot_and.search(filters or {}, raw_regex, limit)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"slotand search failed: {exc}"}

    def grep(self, pattern: str, arm: str | None = None) -> dict:
        """arm 의 (첫) 뷰 파일에 대한 정규식 검색."""
        arm = arm if arm in ARMS else self.arm
        views = list(dict.fromkeys(v for _, v in ARMS[arm]["channels"]))
        rows: list[dict] = []
        used = None
        for view in views:
            rows = self.view_rows(view)
            if rows:
                used = view
                break
        if not rows:
            # 뷰가 아직 없으면 원문 청크로 대체 (그래도 결과를 준다)
            used = "chunks"
            rows = [{"chunk_id": cid, "text": txt} for cid, txt in self.chunks.items()]
        if not rows:
            return {"error": f"no grep corpus for {arm}", "total_matches": 0, "results": []}

        try:
            rx = re.compile(pattern, re.I)
        except re.error:
            rx = re.compile(re.escape(pattern), re.I)

        matched = [r for r in rows if rx.search(r.get("text", ""))]
        total = len(matched)
        if total > 20:
            step = total / 20
            matched = [matched[int(i * step)] for i in range(20)]
        results = []
        for r in matched:
            txt = r.get("text", "")
            m = rx.search(txt)
            s = max(0, (m.start() if m else 0) - 60)
            uid = r.get("chunk_id") or r.get("element_id", "")
            results.append({"id": uid, "match": txt[s:s + 180]})
        out: dict = {"source": used, "total_matches": total, "results": results}
        if total > 20:
            out["note"] = f"total {total} -- sampled 20"
        return out

    def read(self, uid: str) -> dict:
        text = self.elements.get(uid) if uid.startswith("e") else self.chunks.get(uid)
        if text is None:  # 접두사 규칙에서 벗어난 id 대비
            text = self.chunks.get(uid) or self.elements.get(uid)
        return {"id": uid, "text": text[:4000] if text else "", "found": text is not None}

    # 에이전트 디스패치 -----------------------------------------------------
    def dispatch(self, name: str, args: dict, arm: str | None = None) -> dict | list:
        arm = arm if arm in ARMS else self.arm
        if name == "hybrid_search":
            return self.hybrid_search(str(args.get("query", "")), arm)
        if name == "slot_search":
            return self.slot(str(args.get("query", "")), args.get("slots") or None)
        if name == "slot_and_search":
            return self.slotand(args.get("filters") or {}, str(args.get("raw_regex", "")),
                                int(args.get("limit", 20)))
        if name == "grep_search":
            return self.grep(str(args.get("pattern", "")), arm)
        if name == "read_unit":
            return self.read(str(args.get("unit_id", "")))
        return {"error": f"unknown tool: {name}"}


# ── 도구 스키마 ─────────────────────────────────────────────────────────
TOOL_DEFS_ALL = [
    {"type": "function", "function": {
        "name": "hybrid_search",
        "description": "BM25 + 덴스(bge-m3) 다채널 RRF 융합 검색 top-10",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "slot_search",
        "description": "CLM 엘리먼트 슬롯 검색 (계약/주체/역할/조항/표/한정/참조/스키마)",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "slots": {"type": "object", "description": '예: {"contract":"정기특약"}'}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "slot_and_search",
        "description": "AND 슬롯 엘리먼트 검색 (계약/주체/역할/조항/표/한정/참조/스키마, 필드간 AND·필드내 OR, 관련도 순위 없음)",
        "parameters": {"type": "object", "properties": {
            "filters": {"type": "object", "description": '예: {"contract":"질병"}'},
            "raw_regex": {"type": "string"},
            "limit": {"type": "integer"}},
            "required": ["filters"]}}},
    {"type": "function", "function": {
        "name": "grep_search", "description": "뷰 텍스트 정규식 검색",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}},
                       "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "read_unit", "description": "청크(c*) 또는 엘리먼트(e*) 원문 읽기",
        "parameters": {"type": "object", "properties": {"unit_id": {"type": "string"}},
                       "required": ["unit_id"]}}},
]


def tool_defs_for_arm(arm: str) -> list[dict]:
    cfg = ARMS.get(arm, ARMS["V6V9"])
    excluded: set[str] = set()
    if not cfg["clm"]:
        excluded.add("slot_search")
    if not cfg.get("slot_and", False):
        excluded.add("slot_and_search")
    return [t for t in TOOL_DEFS_ALL if t["function"]["name"] not in excluded]


# ── CLI (claude -p 연동용) ──────────────────────────────────────────────
_search: Search | None = None


def _get_search(arm: str) -> Search:
    global _search
    if _search is None:
        _search = Search(arm)
    return _search


def main() -> None:
    arm = os.environ.get("ARM", "V6V9")
    session_dir = os.environ.get("SESSION_DIR", "")

    if len(sys.argv) < 3:
        print(json.dumps({"error": "usage: python tools_rrf.py {hybrid|slot|grep|read} <arg>"},
                         ensure_ascii=False))
        sys.exit(1)

    cmd, arg = sys.argv[1], sys.argv[2]

    if session_dir:
        sd = Path(session_dir)
        sd.mkdir(parents=True, exist_ok=True)
        cf = sd / "calls.jsonl"
        n = 0
        if cf.exists():
            with cf.open(encoding="utf-8") as handle:
                n = sum(1 for line in handle if line.strip())
        if n >= MAX_CALLS:
            print(json.dumps({"error": f"도구 호출 상한({MAX_CALLS}회) 도달"}, ensure_ascii=False))
            sys.exit(0)
        with cf.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"tool": cmd, "arg": arg[:200], "arm": arm},
                                    ensure_ascii=False) + "\n")

    # arm 게이팅 — 이 검사가 없으면 arm 구분이 성립하지 않는다.
    # tool_defs_for_arm() 은 OpenAI 스타일 스키마 목록이라 claude -p 경로에서는
    # 쓰이지 않는다. 실제로 2026-08-14 실험에서 BASE/V9 arm 이 slot 을 각각
    # 114/115 세션에서 호출해 태그 축 비교가 통째로 무효가 됐다. 도구 노출은
    # 프롬프트(문서)와 여기(집행) 두 곳에서 모두 막아야 한다.
    cfg = ARMS.get(arm, ARMS["V6V9"])
    if cmd == "slot" and not cfg["clm"]:
        print(json.dumps({"error": f"'{arm}' arm 에서는 slot 도구를 쓸 수 없다"},
                         ensure_ascii=False))
        sys.exit(0)
    if cmd == "slotand" and not cfg.get("slot_and", False):
        print(json.dumps({"error": f"'{arm}' arm 에서는 slotand 도구를 쓸 수 없다"},
                         ensure_ascii=False))
        sys.exit(0)

    s = _get_search(arm)
    if cmd == "hybrid":
        result = s.hybrid_search(arg, arm)
    elif cmd == "slot":
        try:
            parsed = json.loads(arg)
        except json.JSONDecodeError:
            parsed = arg
        if isinstance(parsed, dict):
            result = s.slot(str(parsed.get("query", "")), parsed.get("slots") or None,
                            int(parsed.get("limit", 20)))
        else:
            result = s.slot(str(parsed))
    elif cmd == "slotand":
        try:
            parsed = json.loads(arg)
        except json.JSONDecodeError:
            parsed = arg
        if isinstance(parsed, dict) and ("filters" in parsed or "raw_regex" in parsed or "limit" in parsed):
            result = s.slotand(parsed.get("filters") or {}, str(parsed.get("raw_regex", "")),
                               int(parsed.get("limit", 20)))
        elif isinstance(parsed, dict):
            # bare {"field": "query", ...} shorthand -> treated directly as filters
            result = s.slotand(parsed)
        else:
            # plain string arg: no per-field filters, just a raw-text regex match
            # (keeps AND-slot semantics intact -- filters stay empty, not unioned)
            result = s.slotand({}, str(parsed))
    elif cmd == "grep":
        result = s.grep(arg, arm)
    elif cmd == "read":
        result = s.read(arg)
    else:
        result = {"error": f"unknown: {cmd}"}

    # 채널 기여도 기록 ---------------------------------------------------
    # hybrid 결과의 채널별 순위 정보는 응답에만 실려 있고 세션 로그에는 남지 않는다.
    # 실험이 끝난 뒤에는 재구성할 수 없으므로(질의 문자열이 같아도 ollama 임베딩
    # 재호출이 필요), 희소/밀집 기여 비율을 이 시점에 세션 디렉터리에 남긴다.
    if session_dir and cmd == "hybrid" and isinstance(result, dict):
        try:
            hits = result.get("results") or []
            per_channel: dict[str, int] = {}
            rr_mass: dict[str, float] = {}
            for hit in hits:
                for ch, rank in (hit.get("channels") or {}).items():
                    per_channel[ch] = per_channel.get(ch, 0) + 1
                    rr_mass[ch] = rr_mass.get(ch, 0.0) + 1.0 / (RRF_K + int(rank))
            total_mass = sum(rr_mass.values()) or 1.0
            sparse = sum(v for k, v in rr_mass.items() if k.startswith("bm25/"))
            dense = sum(v for k, v in rr_mass.items() if k.startswith("dense/"))
            rec = {
                "arm": arm, "query": arg[:200], "n_results": len(hits),
                "channel_sizes": result.get("channels") or {},
                "hits_per_channel": per_channel,
                "rrf_mass": {k: round(v, 6) for k, v in rr_mass.items()},
                "sparse_share": round(sparse / total_mass, 4),
                "dense_share": round(dense / total_mass, 4),
                "notes": result.get("notes") or [],
            }
            with (Path(session_dir) / "channel_stats.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001 — 기록 실패가 검색을 막으면 안 된다
            print(f"[WARN] channel_stats 기록 실패: {exc}", file=sys.stderr)

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
