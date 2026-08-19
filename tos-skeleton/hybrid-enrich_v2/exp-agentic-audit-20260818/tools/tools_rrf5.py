#!/usr/bin/env python3
"""armD'' 도구: 계층 탐색 보강판.

D' 실패 원인 두 가지를 수정:
  1. 조 지도: ST 태그 요약(판별력 없음) 대신 문서에서 직접 추출한
     조 제목 + 본문 첫 문장 (정규식, 커버리지 구멍 없음)
  2. 하강: 조 선택을 hard 필터가 아니라 soft 가점(boost)으로 —
     짚은 조가 틀려도 특약 내 다른 조의 정답이 순위에 남는다

  browse "<특약명일부>"      → 그 특약의 조 목록 (제목 + 첫 문장 발췌)
  node '{"query","scope","article"?}' → 특약 전체에서 BM25 + 선택 조 가점 top-10
  hybrid "<query>"          → 전역 검색 (특약 미지정 질문용)
  grep / read               → tools_rrf4와 동일
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
from tools_rrf4 import S as S4, norm  # noqa: E402

DOC_CACHE = None
ART_RE = re.compile(r"제\s?\d+(?:-\d+)?\s?조(?:의\s?\d+)?\s?\([^)\n]{2,40}\)")
BOOST = 3.0


def doc_text() -> str:
    global DOC_CACHE
    if DOC_CACHE is None:
        import os
        DOC_CACHE = open(os.environ["DOC_PATH"], encoding="utf-8").read()
    return DOC_CACHE


class S5(S4):
    def __init__(self):
        super().__init__()
        self.cspan = {c["chunk_id"]: (c["char_start"], c["char_end"])
                      for c in load_jsonl(OUT / "chunks.jsonl")}

    def rider_range(self, scope_part: str):
        want = norm(scope_part)
        spans = [self.cspan[cid] for cid, sc in self.cscope.items() if want in norm(sc)]
        if not spans:
            return None
        return min(s[0] for s in spans), max(s[1] for s in spans)

    def articles(self, scope_part: str):
        """특약 범위 안의 조 목록: 제목 + 직후 본문 발췌 (문서에서 직접 추출)."""
        rng = self.rider_range(scope_part)
        if not rng:
            return []
        doc = doc_text()
        seen = {}
        for m in ART_RE.finditer(doc, rng[0], rng[1]):
            title = re.sub(r"\s+", " ", m.group(0))
            if title in seen:
                continue
            snippet = re.sub(r"\s+", " ", doc[m.end():m.end() + 90]).strip()
            seen[title] = {"article": title, "lead": snippet, "pos": m.start()}
        return list(seen.values())

    def node_soft(self, query: str, scope_part: str, article_part: str = "", top: int = 10):
        want = norm(scope_part)
        cand = {cid for cid, sc in self.cscope.items() if want in norm(sc)}
        if not cand:
            return {"note": "특약 매칭 실패 — browse로 이름 확인", "results": []}
        art_spans = []
        note = f"특약 청크 {len(cand)}개"
        if article_part:
            wa = norm(article_part)
            arts = [a for a in self.articles(scope_part) if wa in norm(a["article"])]
            rng = self.rider_range(scope_part)
            for i, a in enumerate(arts):
                end = rng[1]
                art_spans.append((a["pos"], end if i + 1 >= len(arts) else arts[i + 1]["pos"]))
            note += f" | 조 '{article_part}' {'가점 적용' if art_spans else '못 찾음(가점 없이 전체 검색)'}"
        scored = []
        base = {cid: sc for cid, sc in self.bm_meta.search(query, top_k=400) if cid in cand}
        for cid in cand:
            sc = base.get(cid, 0.0)
            if art_spans:
                s = self.cspan[cid]
                if any(s[0] < en and s[1] > st for st, en in art_spans):
                    sc += BOOST
            if sc > 0:
                scored.append((cid, sc))
        scored.sort(key=lambda kv: -kv[1])
        if len(scored) < top:
            extra = [c for c in sorted(cand) if c not in {x for x, _ in scored}]
            scored += [(c, 0.0) for c in extra[: top - len(scored)]]
        return {"note": note, "results": [
            {"id": c, "score": round(s, 3), "scope": self.cscope.get(c, "?"),
             "preview": self.ctext.get(c, "")[:200]} for c, s in scored[:top]]}


_S = None


def get() -> S5:
    global _S
    if _S is None:
        _S = S5()
    return _S


def main() -> None:
    import os as _os
    tf=_os.environ.get("TRACE_FILE")
    if tf and len(sys.argv)>=3:
        with open(tf,"a",encoding="utf-8") as _h:
            _h.write(json.dumps({"cmd":sys.argv[1],"arg":sys.argv[2][:160]},ensure_ascii=False)+"\n")

    if len(sys.argv) < 3:
        print(json.dumps({"error": "usage: tools_rrf5.py {browse|node|hybrid|grep|read} <arg>"}, ensure_ascii=False)); return
    cmd, arg = sys.argv[1], sys.argv[2]
    s = get()
    if cmd == "browse":
        arts = s.articles(arg)
        out = {"rider_query": arg, "articles": [{"article": a["article"], "lead": a["lead"]} for a in arts[:45]]}
        if not arts:
            out["note"] = "조 추출 실패 — 특약명을 더 정확히"
    elif cmd == "node":
        p = json.loads(arg)
        out = s.node_soft(str(p.get("query", "")), str(p.get("scope", "")), str(p.get("article", "")))
    elif cmd == "hybrid":
        out = s.hybrid(arg)
    elif cmd == "grep":
        try:
            p = json.loads(arg)
        except json.JSONDecodeError:
            p = arg
        out = s.grep(p["pattern"], p.get("scope", "")) if isinstance(p, dict) else s.grep(str(p))
    elif cmd == "read":
        out = {"id": arg, "scope": s.cscope.get(arg, "?"), "text": s.ctext.get(arg, "")[:4000], "found": arg in s.ctext}
    else:
        out = {"error": f"unknown cmd {cmd}"}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
