"""
QA → qrels 유도 (parent 단위 gold).

법률 aihub 데이터는 정답 라벨이 '문서(parent)' 단위다(질문 → 판례/문서). 그래서
gold도 parent 단위로 기록한다: qrels.corpus-id = 질문의 parent_id.
- run_experiment는 검색된 청크를 parent로 사상(to_parent_ids)해 이 gold와 비교하므로,
  512·element 어느 청킹이든 동일 parent gold로 평가된다(청킹 무관 재사용).
- 청킹별 '어느 청크를 얼마나 정확히' 가져왔는지는 span 기반 coverage/density가 담당
  (qa_meta.supporting_spans, 청크 gold 불필요).
텍스트 재매칭·오프셋 겹침으로 청크 gold를 만들지 않으므로 매칭 실패/1글자 겹침 문제가 없다.

사용: python derive_qrels.py <variant>   (smoke | element)
출력: data/aihub/<variant>/{queries.jsonl, qrels.jsonl}  (qrels: parent gold)
"""
import json, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
AIHUB = BASE / "data" / "aihub"
QA = AIHUB / "qa"


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    cdir = AIHUB / variant

    # corpus에 실제 존재하는 parent만 gold로 인정
    corpus_parents = {json.loads(l)["parent_id"] for l in open(cdir / "corpus.jsonl")}

    qa = [json.loads(l) for l in open(QA / "qa_meta.jsonl")]
    queries = [json.loads(l) for l in open(QA / "queries.jsonl")]

    qrels, unmatched = [], []
    for m in qa:
        pid = m["parent_id"]
        if pid in corpus_parents:
            qrels.append({"query-id": int(m["qid"]), "corpus-id": pid, "score": 2})
        else:
            unmatched.append(m["qid"])

    with open(cdir / "qrels.jsonl", "w", encoding="utf-8") as f:
        for x in qrels:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(cdir / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            if int(q["_id"]) not in unmatched:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"[{variant}] 질문 {len(qa)} → parent gold {len(qrels)}, 누락 {len(unmatched)}")
    print(f"  qrels {len(qrels)}행 (질의당 1 parent) → {cdir/'qrels.jsonl'}")
    if unmatched:
        print(f"  ⚠️ parent가 corpus에 없는 질문: {unmatched}")


if __name__ == "__main__":
    main()
