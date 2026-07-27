"""
정답 span 위치 → 각 청킹의 gold 청크(qrels) 유도 — 오프셋 구간 겹침.

qa_meta.supporting_spans[].occurrences 는 QA 생성 시 원본(parents.jsonl)에서
찾은 '모든' 등장 구간 [(s,e),...] 이다. 텍스트 재매칭 없이 이 기록된 구간과
corpus 청크의 [char_start,char_end] 가 겹치는지로 gold 를 정한다.
- 중복 등장(정답 문장이 여러 곳)도 모든 등장 위치의 청크를 gold 로(합집합).
- 재매칭이 없어 first-occurrence 편향/매칭실패가 없다.

사용: python derive_qrels.py <variant>   (smoke | element)
출력: data/aihub/<variant>/{queries.jsonl, qrels.jsonl}
"""
import json, sys
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
AIHUB = BASE / "data" / "aihub"
QA = AIHUB / "qa"


def overlaps(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    cdir = AIHUB / variant

    corpus = [json.loads(l) for l in open(cdir / "corpus.jsonl")]
    by_parent = defaultdict(list)
    for d in corpus:
        by_parent[d["parent_id"]].append((d["_id"], d["char_start"], d["char_end"]))

    qa = [json.loads(l) for l in open(QA / "qa_meta.jsonl")]
    queries = [json.loads(l) for l in open(QA / "queries.jsonl")]

    qrels, unmatched, per_q = [], [], []
    for m in qa:
        chunks = by_parent.get(m["parent_id"], [])
        gold = []
        for span in m["supporting_spans"]:
            for (s0, s1) in span.get("occurrences", []):
                for cid, cs, ce in chunks:
                    if overlaps(s0, s1, cs, ce):
                        gold.append(cid)
        gold = list(dict.fromkeys(gold))
        if gold:
            per_q.append(len(gold))
            for cid in gold:
                qrels.append({"query-id": int(m["qid"]), "corpus-id": cid, "score": 2})
        else:
            unmatched.append(m["qid"])

    with open(cdir / "qrels.jsonl", "w", encoding="utf-8") as f:
        for x in qrels:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(cdir / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            if int(q["_id"]) not in unmatched:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

    avg = sum(per_q) / len(per_q) if per_q else 0
    print(f"[{variant}] 질문 {len(qa)} → gold {len(qa)-len(unmatched)}, 누락 {len(unmatched)}")
    print(f"  질의당 gold 청크 평균 {avg:.1f} | qrels {len(qrels)}행 → {cdir/'qrels.jsonl'}")
    if unmatched:
        print(f"  ⚠️ occurrences가 비어 gold 없는 질문: {unmatched}")


if __name__ == "__main__":
    main()
