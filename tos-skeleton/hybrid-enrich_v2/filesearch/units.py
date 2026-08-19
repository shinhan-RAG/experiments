#!/usr/bin/env python3
"""제출 id → 채점 단위 변환. 태그 검색 결과(e*, 항)·조(j*)·메타 검색 결과(c*, vector_search 청크)를 하나의 순위로 받는다.
채점은 조(u2jo) 단위: 각 id 를 소속 조로 사상(청크는 겹침이 가장 큰 조)하고 중복 조는 첫 등장만 남긴다."""
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent


class Units:
    def __init__(self, jo_path=None, chunks_path=None):
        self.J = [json.loads(l) for l in open(jo_path or HERE / "out/elements_u2jo.jsonl", encoding="utf-8")]
        self.jid = {u["element_id"]: u for u in self.J}
        self.m2j = {m: u for u in self.J for m in u["members"]}
        self._starts = [u["char_start"] for u in self.J]
        cp = Path(chunks_path) if chunks_path else HERE.parent / "vector_search" / "out" / "chunks.jsonl"
        self.C = {}
        if cp.exists():
            for l in open(cp, encoding="utf-8"):
                c = json.loads(l); self.C[c["chunk_id"]] = c

    def jo_of_span(self, c0, c1):
        import bisect
        i = bisect.bisect_right(self._starts, c0) - 1
        best, bo = None, 0
        for k in range(max(0, i - 2), min(len(self.J), i + 6)):
            u = self.J[k]; o = min(c1, u["char_end"]) - max(c0, u["char_start"])
            if o > bo:
                best, bo = u, o
        return best

    def resolve(self, ids):
        """→ 채점 단위(조) 순위 리스트(중복 제거). 각 항목은 char_start/char_end 를 가진 dict."""
        out, seen = [], set()
        for x in ids:
            u = None
            if x.startswith("j"):
                u = self.jid.get(x)
            elif x.startswith("e"):
                u = self.m2j.get(x)
            elif x.startswith("c") and x in self.C:
                c = self.C[x]; u = self.jo_of_span(c["char_start"], c["char_end"])
                if u is None:
                    u = {"element_id": x, "char_start": c["char_start"], "char_end": c["char_end"]}
            if u is not None and u["element_id"] not in seen:
                seen.add(u["element_id"]); out.append(u)
        return out
