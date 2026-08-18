#!/usr/bin/env python3
"""실험 arm C/D 도구.

arm D (계층 내비게이션): ST 태그를 '지도'로 쓴다.
  browse                → 대분류: 특약 목록 (청크 수)
  browse "<특약명일부>"  → 중분류: 그 특약의 조(article) 목록 + ST 요약(subject/role)
  node '{"query":...,"scope":"특약명일부","article":"조명일부"}'
                        → 소분류: 그 노드 범위 안에서만 META(V9) BM25 검색 top-10
                          article 생략 시 특약 전체(중 레벨)에서 검색 — 백트래킹용
  read <id>             → 원문 + 소속

arm C (메타+ST 재료): VIEW=V9T 로 실행하면 hybrid/grep 이 태그 어휘가 접두된
  view_V9T 를 검색한다 (build_view_v9t.py 로 생성).
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tools_rrf import BM25, load_jsonl, OUT, build_or_load_bm25  # noqa: E402

VIEW = os.environ.get("VIEW", "V9")
PREVIEW = 200


def norm(s: str) -> str:
    return re.sub(r"[\s\(\)\[\]〔〕·,~%\-]", "", s or "")


class Data:
    def __init__(self):
        self.chunks = load_jsonl(OUT / "chunks.jsonl")
        self.cspan = {c["chunk_id"]: (c["char_start"], c["char_end"]) for c in self.chunks}
        self.cscope = {c["chunk_id"]: (c.get("contract_scope") or "(주계약/공통)") for c in self.chunks}
        self.ctext = {c["chunk_id"]: c["text"] for c in self.chunks}
        self.elements = load_jsonl(OUT / "elements_psection.jsonl")
        self.espan = {e["element_id"]: (e["char_start"], e["char_end"]) for e in self.elements}
        self.escope = {e["element_id"]: (e.get("contract_scope") or "(주계약/공통)") for e in self.elements}
        self.tags = load_jsonl(OUT / "element_tags_v2.jsonl")
        # 태그 element_id(e00124__r002) → psection element(e00124) 매핑
        self.tag_of = defaultdict(list)
        for t in self.tags:
            base = t["element_id"].split("__")[0]
            self.tag_of[base].append(t)
        self.bm25, _ = build_or_load_bm25(VIEW)

    def riders(self):
        cnt = defaultdict(int)
        for cid, sc in self.cscope.items():
            cnt[sc] += 1
        return sorted(cnt.items(), key=lambda kv: -kv[1])

    def rider_articles(self, scope_part: str):
        """특약 안의 조 목록 + ST 요약."""
        want = norm(scope_part)
        arts = defaultdict(lambda: {"n": 0, "subject": set(), "role": set(), "span": [None, None]})
        for e in self.elements:
            if want not in norm(self.escope[e["element_id"]]):
                continue
            for t in self.tag_of.get(e["element_id"], []):
                for a in t.get("article") or ["(조항 미상)"]:
                    node = arts[a]
                    node["n"] += 1
                    node["subject"].update((t.get("subject") or [])[:4])
                    node["role"].update((t.get("role") or [])[:4])
                    st, en = self.espan[e["element_id"]]
                    node["span"][0] = st if node["span"][0] is None else min(node["span"][0], st)
                    node["span"][1] = en if node["span"][1] is None else max(node["span"][1], en)
        return arts

    def node_search(self, query: str, scope_part: str, article_part: str = "", top: int = 10):
        want = norm(scope_part)
        cand = {cid for cid, sc in self.cscope.items() if want in norm(sc)}
        note = f"특약 매칭 청크 {len(cand)}개"
        if article_part:
            arts = self.rider_articles(scope_part)
            wa = norm(article_part)
            spans = [v["span"] for a, v in arts.items() if wa in norm(a) and v["span"][0] is not None]
            if spans:
                in_art = {cid for cid in cand
                          if any(self.cspan[cid][0] < en and self.cspan[cid][1] > st for st, en in spans)}
                note += f" → 조 '{article_part}' 매칭 {len(in_art)}개"
                cand = in_art
            else:
                note += f" → 조 '{article_part}' 없음 — 특약 레벨로 백트래킹해서 검색함"
        if not cand:
            return {"note": "해당 노드에 청크 없음 — browse 로 상위 레벨 확인", "results": []}
        scored = [(cid, sc) for cid, sc in self.bm25.search(query, top_k=400) if cid in cand]
        if len(scored) < top:  # BM25 미스 시 문서순 보충
            extra = [cid for cid in sorted(cand) if cid not in {c for c, _ in scored}]
            scored += [(cid, 0.0) for cid in extra[: top - len(scored)]]
        out = [{"id": cid, "score": sc, "scope": self.cscope.get(cid, "?"),
                "preview": self.ctext.get(cid, "")[:PREVIEW]} for cid, sc in scored[:top]]
        return {"note": note, "results": out}


_D = None


def data() -> Data:
    global _D
    if _D is None:
        _D = Data()
    return _D


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: tools_rrf3.py {browse|node|hybrid|grep|read} [arg]"}, ensure_ascii=False)); return
    cmd = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else ""
    d = data()
    if cmd == "browse":
        if not arg:
            out = {"level": "대(특약)", "riders": [{"scope": s, "chunks": n} for s, n in d.riders()[:60]]}
        else:
            arts = d.rider_articles(arg)
            out = {"level": f"중(조) — {arg}",
                   "articles": [{"article": a, "elements": v["n"],
                                 "subjects": sorted(v["subject"])[:6], "roles": sorted(v["role"])[:6]}
                                for a, v in sorted(arts.items(), key=lambda kv: -kv[1]["n"])[:40]]}
            if not arts:
                out["note"] = "특약명 매칭 실패 — browse 로 정확한 특약명 확인"
    elif cmd == "node":
        try:
            p = json.loads(arg)
        except json.JSONDecodeError:
            print(json.dumps({"error": 'node arg must be JSON {"query","scope","article"?}'}, ensure_ascii=False)); return
        out = d.node_search(str(p.get("query", "")), str(p.get("scope", "")), str(p.get("article", "")))
    elif cmd == "hybrid":
        hits = d.bm25.search(arg, top_k=10)
        out = {"view": VIEW, "results": [{"id": c, "score": s, "scope": d.cscope.get(c, "?"),
                                          "preview": d.ctext.get(c, "")[:PREVIEW]} for c, s in hits]}
    elif cmd == "grep":
        try:
            rx = re.compile(arg, re.I)
        except re.error:
            rx = re.compile(re.escape(arg), re.I)
        rows = load_jsonl(OUT / f"view_{VIEW}.jsonl")
        by = defaultdict(list)
        total = 0
        for r in rows:
            m = rx.search(r.get("text", ""))
            if m:
                total += 1
                sc = d.cscope.get(r["chunk_id"], "?")
                if len(by[sc]) < 3:
                    s0 = max(0, m.start() - 40)
                    by[sc].append({"id": r["chunk_id"], "match": r["text"][s0:s0 + 140]})
        out = {"total": total, "groups": [{"scope": s, "hits": h} for s, h in sorted(by.items(), key=lambda kv: -len(kv[1]))[:12]]}
    elif cmd == "read":
        txt = d.ctext.get(arg) or next((e["text"] for e in d.elements if e["element_id"] == arg), "")
        out = {"id": arg, "scope": d.cscope.get(arg, d.escope.get(arg, "?")), "text": (txt or "")[:4000], "found": bool(txt)}
    else:
        out = {"error": f"unknown cmd {cmd}"}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
