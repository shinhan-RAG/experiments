"""
4단계: 품질 게이트 + BEIR 최종 병합.

- 기계 검사: span 실존율, gold 청크 id 유효성, 질문 중복(정규화 유사도)
- 분포 리포트: 트랙/사업구분/문서유형/element_type/문서크기 버킷
- 통과분을 BEIR 포맷으로 병합: queries.jsonl / qrels.jsonl / qa_meta.jsonl

주의(회의록 18장): 코드 작성자가 QA까지 만들면 실험에 유리한 데이터가 될 수 있음.
→ 이 게이트는 기계 검증만 수행. 최종 확정 전 반드시 타인이 샘플 검수할 것.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# 질문에 정식 문서명의 괄호 부가정보가 그대로 들어간 경우 (사용자 피드백: 부자연스러움)
OVERDETAIL_RE = re.compile(r"무배당|해약환급금|해지환급금 미지급|\(갱신형\)|배당, ")

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "raw" / "shinhan-tos"


def norm_q(q: str) -> str:
    return "".join(q.split()).rstrip("?.!")


def main():
    chunks = {}
    with open(DATA_DIR / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks[c["_id"]] = c
    doc_meta = {}
    with open(DATA_DIR / "docs_selected.json", encoding="utf-8") as f:
        for d in json.load(f)["docs"]:
            doc_meta[d["name"]] = d

    records = []
    for src in ["qa_aug_raw.jsonl", "qa_new_raw.jsonl"]:
        p = DATA_DIR / src
        if p.exists():
            with open(p, encoding="utf-8") as f:
                records.extend(json.loads(l) for l in f if l.strip())
    print(f"원시 QA: {len(records)}건")

    kept, fails, seen_q = [], Counter(), set()
    for r in records:
        chunk = chunks.get(r["gold_chunk_id"])
        if chunk is None:
            fails["gold_chunk_없음"] += 1
            continue
        # span 실존 재검증 (verbatim 또는 공백정규화 매칭)
        ok_spans = []
        comp = " ".join(chunk["text"].split())
        for s in r.get("supporting_spans", []):
            if s["text"] in chunk["text"] or " ".join(s["text"].split()) in comp:
                ok_spans.append(s)
        if not ok_spans:
            fails["span_원문_불일치"] += 1
            continue
        if len(r.get("question", "")) < 10 or len(r.get("answer", "")) < 5:
            fails["질문·정답_너무_짧음"] += 1
            continue
        if OVERDETAIL_RE.search(r["question"]):
            fails["질문에_정식문서명_과다"] += 1
            continue
        nq = norm_q(r["question"])
        if nq in seen_q:
            fails["질문_중복"] += 1
            continue
        seen_q.add(nq)
        r["supporting_spans"] = ok_spans
        kept.append(r)

    print(f"통과: {len(kept)}건, 탈락: {dict(fails)}")

    # ---- 분포 리포트 ----
    def dist(key_fn, label):
        print(f"\n[{label}]")
        for k, v in Counter(key_fn(r) for r in kept).most_common():
            print(f"  {k}: {v}")

    dist(lambda r: r["source"], "트랙(source)")
    dist(lambda r: r["qa_type"], "사업구분(qa_type)")
    dist(lambda r: r["element_type"], "element_type")
    dist(lambda r: doc_meta.get(r["doc"], {}).get("type", "?"), "문서 유형")
    dist(lambda r: doc_meta.get(r["doc"], {}).get("bucket", "?"), "문서 크기 버킷")
    per_doc = Counter(r["doc"] for r in kept)
    print(f"\n[문서 커버리지] {len(per_doc)}개 문서, 최대 편중 {per_doc.most_common(1)}")

    # ---- BEIR 병합 ----
    with open(DATA_DIR / "queries.jsonl", "w", encoding="utf-8") as fq, \
         open(DATA_DIR / "qrels.jsonl", "w", encoding="utf-8") as fr, \
         open(DATA_DIR / "qa_meta.jsonl", "w", encoding="utf-8") as fm:
        for i, r in enumerate(kept, start=1):
            qid = str(i)
            fq.write(json.dumps({"_id": qid, "text": r["question"]},
                                ensure_ascii=False) + "\n")
            fr.write(json.dumps({"query-id": i, "corpus-id": r["gold_chunk_id"],
                                 "score": 2}, ensure_ascii=False) + "\n")
            fm.write(json.dumps({"qid": qid, "question": r["question"],
                                 "answer": r["answer"],
                                 "gold_chunk_id": r["gold_chunk_id"],
                                 "doc": r["doc"], "element_type": r["element_type"],
                                 "source": r["source"], "qa_type": r["qa_type"],
                                 "orig_no": r.get("orig_no"),
                                 "supporting_spans": r["supporting_spans"]},
                                ensure_ascii=False) + "\n")
    print(f"\nBEIR 산출: queries/qrels/qa_meta {len(kept)}건 → {DATA_DIR}")
    print("※ 최종 확정 전 사람 샘플 검수 필요 (질문↔정답↔span 일관성 10건 이상)")


if __name__ == "__main__":
    main()
