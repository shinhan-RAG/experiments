#!/usr/bin/env python3
"""armC' 도구: 태그를 concat이 아니라 **별도 BM25 채널**로 쓰고 RRF 융합.

  hybrid "<query>"   → bm25/V9(메타뷰) top50 ⊕ bm25/TAG(태그뷰) top50 을
                       RRF(k=60, tag_weight=1.0) 융합 → top-10 (+scope, 채널 출처)
  grep '<pattern>' | '{"pattern":..,"scope":..}'  → V9 뷰, 특약별 그룹 (tools_rrf2와 동일)
  node '{"query":..,"scope":..}'                  → 특약 범위 내 V9 검색
  read <id>

챔피언 RRF(TAG+META 2채널)의 소비 방식을 이 하네스에 이식한 것.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tools_rrf import load_jsonl, OUT, build_or_load_bm25  # noqa: E402

RRF_K = 60
TAG_W = 1.0
PREVIEW = 200


def norm(s: str) -> str:
    return re.sub(r"[\s\(\)\[\]〔〕·,~%\-]", "", s or "")


class S:
    def __init__(self):
        self.chunks = load_jsonl(OUT / "chunks.jsonl")
        self.cscope = {c["chunk_id"]: (c.get("contract_scope") or "(주계약/공통)") for c in self.chunks}
        self.ctext = {c["chunk_id"]: c["text"] for c in self.chunks}
        self.bm_meta, _ = build_or_load_bm25("V9")
        self.bm_tag, _ = build_or_load_bm25("TAG")

    def hybrid(self, query: str, top: int = 10) -> dict:
        fused: dict[str, float] = defaultdict(float)
        prov: dict[str, dict] = defaultdict(dict)
        for label, idx, w in (("meta", self.bm_meta, 1.0), ("tag", self.bm_tag, TAG_W)):
            if idx is None:
                continue
            for rank, (cid, _) in enumerate(idx.search(query, top_k=50), 1):
                fused[cid] += w / (RRF_K + rank)
                prov[cid][label] = rank
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top]
        return {"results": [{"id": c, "rrf": round(s, 6), "channels": prov[c],
                             "scope": self.cscope.get(c, "?"),
                             "preview": self.ctext.get(c, "")[:PREVIEW]} for c, s in ranked]}

    def grep(self, pattern: str, scope: str = "") -> dict:
        rows = load_jsonl(OUT / "view_V9.jsonl")
        try:
            rx = re.compile(pattern, re.I)
        except re.error:
            rx = re.compile(re.escape(pattern), re.I)
        want = norm(scope)
        by = defaultdict(list)
        total = 0
        for r in rows:
            m = rx.search(r.get("text", ""))
            if not m:
                continue
            sc = self.cscope.get(r["chunk_id"], "?")
            if want and want not in norm(sc):
                continue
            total += 1
            if len(by[sc]) < 3:
                s0 = max(0, m.start() - 40)
                by[sc].append({"id": r["chunk_id"], "match": r["text"][s0:s0 + 140]})
        out = {"total": total, "groups": [{"scope": s2, "hits": h} for s2, h in
                                          sorted(by.items(), key=lambda kv: -len(kv[1]))[:12]]}
        if len(by) > 1 and not scope:
            out["hint"] = '여러 특약에 반복됨 — {"pattern":...,"scope":"특약명일부"} 로 좁혀라'
        return out

    def node(self, query: str, scope: str, top: int = 10) -> dict:
        want = norm(scope)
        cand = {cid for cid, sc in self.cscope.items() if want in norm(sc)}
        scored = [(c, s2) for c, s2 in self.bm_meta.search(query, top_k=400) if c in cand]
        if len(scored) < top:
            extra = [c for c in sorted(cand) if c not in {x for x, _ in scored}]
            scored += [(c, 0.0) for c in extra[: top - len(scored)]]
        return {"note": f"특약 매칭 {len(cand)}청크", "results": [
            {"id": c, "score": s2, "scope": self.cscope.get(c, "?"),
             "preview": self.ctext.get(c, "")[:PREVIEW]} for c, s2 in scored[:top]]}


_S = None


def get() -> S:
    global _S
    if _S is None:
        _S = S()
    return _S


def main() -> None:
    if len(sys.argv) < 3:
        print(json.dumps({"error": "usage: tools_rrf4.py {hybrid|grep|node|read} <arg>"}, ensure_ascii=False)); return
    cmd, arg = sys.argv[1], sys.argv[2]
    s = get()
    if cmd == "hybrid":
        out = s.hybrid(arg)
    elif cmd == "grep":
        try:
            p = json.loads(arg)
        except json.JSONDecodeError:
            p = arg
        out = s.grep(p["pattern"], p.get("scope", "")) if isinstance(p, dict) else s.grep(str(p))
    elif cmd == "node":
        p = json.loads(arg)
        out = s.node(str(p.get("query", "")), str(p.get("scope", "")))
    elif cmd == "read":
        out = {"id": arg, "scope": s.cscope.get(arg, "?"), "text": s.ctext.get(arg, "")[:4000],
               "found": arg in s.ctext}
    else:
        out = {"error": f"unknown cmd {cmd}"}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
