#!/usr/bin/env python3
"""태그 전용 어휘 색인 — 0점 문서를 순위에서 제외한다.

왜 별도 파일인가
    meta-search-v4/retrieve.py의 `Channel.rank`는 이렇게 되어 있다.

        s = self.bm.score(question)
        return np.argsort(-s)[:pool].tolist()

    0점을 거르지 않는다. 지금까지는 문제가 드러나지 않았다 — 그쪽 view는 항상
    `메타 + 본문`이라 거의 모든 청크가 질문과 최소 한 글자쌍은 겹쳐 0점이 아니었다.

    v7의 어휘 채널은 다르다. **태그 단독 색인**(본문 없음, 평균 32자)이라 대부분의
    문서가 질문과 아무 것도 공유하지 않아 정확히 0점이다. 그 상태로 argsort하면
    0점 문서들이 **인덱스 순서대로** pool개까지 반환되고, RRF는 그것을 순위로
    믿고 가산점을 준다. 5,335개 중 800개를 뽑으면 그 대부분이 순수 잡음이다.

    잡음이 등가중으로 들어가면 융합은 반드시 진다. dr-dci가 FiQA에서 관측한
    "신규 21건 얻고 117건 밀려남"의 더 나쁜 버전이 된다. 그래서 이 파일이 먼저다.

retrieve.py는 건드리지 않는다
    08-07 등록본이 그 코드로 재현되어야 하므로, BM25 수식만 import해서 쓰고
    순위 산출만 여기서 다시 정의한다. 수식이 두 벌로 갈라지지 않게 하려는 것이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_V4 = Path(__file__).resolve().parent.parent / "metajson-v6" / "meta-search-v4"
if str(_V4) not in sys.path:
    sys.path.insert(0, str(_V4))

from common import OUT, read_jsonl  # noqa: E402
from retrieve import BM25, bigram, tok  # noqa: E402


class LexIndex:
    """view_{cs}_{arm}__meta.jsonl(메타 단독) 위의 BM25.

    `__meta` view를 쓰는 것이 핵심이다. 본체 view(`view_{cs}_{arm}.jsonl`)는
    태그 + 본문이고, 그 구성은 semtag 트랙이 8개 스키마로 전부 반증했다.
    """

    def __init__(self, chunkset: str, arm: str, ids: list[str], tokenizer=bigram):
        path = OUT / f"view_{chunkset}_{arm}__meta.jsonl"
        if not path.exists():
            raise SystemExit(
                f"메타 view가 없습니다: {path}\n"
                f"  -> build_views.py --chunkset {chunkset} --fields --arms {arm}")
        texts = {r["chunk_id"]: r["text"] for r in read_jsonl(path)}
        missing = [c for c in ids if c not in texts]
        if missing:
            raise SystemExit(f"{arm}: chunk_id {len(missing)}개가 view에 없습니다")
        self.docs = [texts[c] for c in ids]
        self.n_nonempty = sum(1 for d in self.docs if d.strip())
        self.bm = BM25(self.docs, tokenizer=tokenizer)

    def rank(self, question: str, pool: int) -> list[int]:
        """0점 문서를 뺀 상위 pool개. 동점은 인덱스 순으로 결정론적."""
        s = self.bm.score(question)
        nz = np.flatnonzero(s > 0)
        if nz.size == 0:
            return []
        order = nz[np.argsort(-s[nz], kind="stable")]
        return order[:pool].tolist()

    def coverage(self, questions: list[str], pool: int) -> dict:
        """이 채널이 실제로 몇 개나 돌려주는지. 0에 가까우면 융합에 기여할 수 없다."""
        counts = [len(self.rank(q, pool)) for q in questions]
        arr = np.asarray(counts, dtype=float)
        return {
            "n_questions": len(counts),
            "docs_nonempty": self.n_nonempty,
            "mean_returned": round(float(arr.mean()), 1) if counts else 0.0,
            "median_returned": int(np.median(arr)) if counts else 0,
            "zero_return_questions": int((arr == 0).sum()),
        }


__all__ = ["LexIndex", "BM25", "bigram", "tok"]
