#!/usr/bin/env python3
"""tools_rrf 개선판 (실험용 v2 도구).

변경 3가지 — 전부 결정론:
  1. hybrid 결과에 각 청크의 특약 귀속(scope)을 병기
  2. grep: 균등 샘플 20건 대신 **특약별 그룹 집계**를 반환하고,
     scope 필터로 특정 특약 안에서만 검색 가능
  3. grep 응답에 전체 매칭 수 + "좁히기" 안내를 항상 포함
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tools_rrf import Search, load_jsonl, OUT, ARMS  # noqa: E402

_SCOPE = None


def scope_map() -> dict:
    global _SCOPE
    if _SCOPE is None:
        _SCOPE = {}
        for fname, key in (("chunks.jsonl", "chunk_id"), ("elements_psection.jsonl", "element_id")):
            p = OUT / fname
            if p.exists():
                for r in load_jsonl(p):
                    _SCOPE[r[key]] = r.get("contract_scope") or "(주계약/공통)"
    return _SCOPE


class Search2(Search):
    def hybrid_search(self, query: str, arm: str | None = None) -> dict:
        out = super().hybrid_search(query, arm)
        sm = scope_map()
        for item in out.get("results", []):
            item["scope"] = sm.get(item["id"], "?")
        return out

    def grep2(self, pattern: str, scope: str = "", arm: str | None = None) -> dict:
        arm = arm if arm in ARMS else self.arm
        views = list(dict.fromkeys(v for _, v in ARMS[arm]["channels"]))
        rows = []
        for view in views:
            rows = self.view_rows(view)
            if rows:
                break
        if not rows:
            rows = [{"chunk_id": cid, "text": txt} for cid, txt in self.chunks.items()]
        try:
            rx = re.compile(pattern, re.I)
        except re.error:
            rx = re.compile(re.escape(pattern), re.I)
        sm = scope_map()
        norm = lambda s: re.sub(r"[\s\(\)\[\]·,~%]", "", s)
        want = norm(scope) if scope else ""
        by_scope: dict[str, list] = defaultdict(list)
        total = 0
        for r in rows:
            m = rx.search(r.get("text", ""))
            if not m:
                continue
            uid = r.get("chunk_id") or r.get("element_id", "")
            sc = sm.get(uid, "?")
            if want and want not in norm(sc):
                continue
            total += 1
            if len(by_scope[sc]) < 3:
                s0 = max(0, m.start() - 50)
                by_scope[sc].append({"id": uid, "match": r["text"][s0:s0 + 160]})
        groups = [{"scope": sc, "n": sum(1 for r in rows if (r.get("chunk_id") or r.get("element_id","")) and sm.get(r.get("chunk_id") or r.get("element_id",""))==sc and rx.search(r.get("text",""))), "hits": hits}
                  for sc, hits in sorted(by_scope.items(), key=lambda kv: -len(kv[1]))]
        out = {"pattern": pattern, "scope_filter": scope or None,
               "total_matches": total, "n_scopes": len(by_scope),
               "groups": groups[:15]}
        if len(by_scope) > 1 and not scope:
            out["hint"] = ("동일 문구가 여러 특약에 반복된다. 질문이 특정 특약을 다루면 "
                           "grep 인자를 JSON으로 {\"pattern\": ..., \"scope\": \"특약명 일부\"} 로 좁혀라.")
        return out


_CACHE: dict[str, Search2] = {}


def _get(arm: str) -> Search2:
    if arm not in _CACHE:
        _CACHE[arm] = Search2(arm)
    return _CACHE[arm]


def main() -> None:
    import os
    arm = os.environ.get("ARM", "V6V9")
    if len(sys.argv) < 3:
        print(json.dumps({"error": "usage: python3 tools_rrf2.py {hybrid|grep|read} <arg>"}, ensure_ascii=False))
        sys.exit(1)
    cmd, arg = sys.argv[1], sys.argv[2]
    s = _get(arm)
    if cmd == "hybrid":
        result = s.hybrid_search(arg, arm)
    elif cmd == "grep":
        try:
            parsed = json.loads(arg)
        except json.JSONDecodeError:
            parsed = arg
        if isinstance(parsed, dict):
            result = s.grep2(str(parsed.get("pattern", "")), str(parsed.get("scope", "")), arm)
        else:
            result = s.grep2(str(parsed), "", arm)
    elif cmd == "read":
        result = s.read(arg)
        result["scope"] = scope_map().get(arg, "?")
    else:
        result = {"error": f"unknown cmd {cmd}"}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
