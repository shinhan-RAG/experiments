"""
3b단계: 청크 샘플링 신규 QA 생성 트랙 (목표 ~60개).

build_shinhan_qa_v2.py 방식 이식 (LLM만 OpenAI → claude CLI 교체):
- 문서×element_type 층화 샘플링 (문서당 상한 PER_DOC_CAP — 대형 문서 편중 방지)
- claude CLI(haiku)가 청크를 보고 질문·정답·verbatim 근거 생성 (usable 필터 포함)
- locate_spans로 근거 실존 검증 → 실패 시 폐기

출력: data/raw/shinhan-tos/qa_new_raw.jsonl
"""
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tos_llm import call_claude_json, locate_spans

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "raw" / "shinhan-tos"
OUT = DATA_DIR / "qa_new_raw.jsonl"

SEED = 42
TARGET_N = 340   # 증강 트랙이 소스 소진으로 160에서 마감 → 총 500 맞춤
PER_DOC_CAP = 45   # 후보 풀 소진으로 상향 (대형 문서가 추가분 공급)
MIN_CHUNK_CHARS = 300   # 정보성 없는 짧은 청크 제외
WORKERS = 6

PROMPT = """당신은 보험 문서 검색 평가셋을 만드는 전문가입니다.
아래는 신한라이프 보험 문서에서 뽑은 한 청크입니다.

이 청크에 근거해서 실제 상담원/고객이 물어볼 법한 **구체적인 질문 1개**를 만들고,
정답과 **정답의 근거가 되는 원문 문장을 청크에서 그대로(verbatim) 인용**하세요.

규칙:
- 질문은 이 청크를 봐야만 답할 수 있을 만큼 구체적(상품명·항목·조건·수치 특정).
- 단, 질문에 청크 원문 표현을 그대로 베끼지 말고 **고객이 쓸 법한 일상 어휘**로 바꿔 쓸 것
  (검색 난이도 확보 — 어휘 간극 유지).
- 질문에 파일명·정식 문서명 전체를 넣지 말 것. "(무배당...)", "(갱신형)", "해약환급금 미지급형" 같은
  괄호 부가정보 금지. 고객이 실제 부르듯 짧은 상품명으로 지칭 (예: "신한통합건강보장보험 원").
  상품 특정이 불필요한 질문이면 상품명 없이 물어도 됨.
- evidence는 반드시 **청크 원문에 그대로 존재하는 문자열**이어야 함(요약·변형 금지). 1~3개.
- 청크가 표지/목차/인사말/OCR깨짐 등 정보성이 없으면 usable=false.
- 아래 JSON만 출력:
{"usable": true/false, "question": "...", "answer": "...", "evidence": ["원문 인용1", "원문 인용2"]}

[문서명] {doc}
[섹션] {section}
[청크]
{chunk}
"""


def main():
    chunks_by_key = defaultdict(list)
    with open(DATA_DIR / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if len(c["text"]) >= MIN_CHUNK_CHARS:
                chunks_by_key[(c["doc"], c["element_type"])].append(c)

    # 층화 샘플링: (문서, element_type) 셀을 돌며 1개씩, 문서당 상한까지
    rng = random.Random(SEED)
    cells = sorted(chunks_by_key.keys())
    for k in cells:
        rng.shuffle(chunks_by_key[k])
    picked, per_doc, idx = [], defaultdict(int), defaultdict(int)
    progressed = True
    while progressed and len(picked) < TARGET_N * 2:   # 폐기율 감안 여유분
        progressed = False
        for k in cells:
            doc, _ = k
            if per_doc[doc] >= PER_DOC_CAP or idx[k] >= len(chunks_by_key[k]):
                continue
            picked.append(chunks_by_key[k][idx[k]])
            idx[k] += 1
            per_doc[doc] += 1
            progressed = True
            if len(picked) >= TARGET_N * 2:
                break
    print(f"샘플 청크: {len(picked)}개, 문서 수: {len(per_doc)}")

    done = set()
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["gold_chunk_id"])

    def work(chunk):
        prompt = (PROMPT.replace("{doc}", chunk["doc"][:80])
                  .replace("{section}", (chunk["section"] or "")[:80])
                  .replace("{chunk}", chunk["text"][:4000]))
        res = call_claude_json(prompt)
        if not res or not res.get("usable"):
            return None
        spans = locate_spans(chunk["text"], res.get("evidence"))
        if not spans:
            return None
        return {"source": "new", "orig_no": None, "qa_type": "신규",
                "question": res["question"], "answer": res["answer"],
                "gold_chunk_id": chunk["_id"], "doc": chunk["doc"],
                "element_type": chunk["element_type"], "supporting_spans": spans}

    n_ok = len(done)
    with open(OUT, "a", encoding="utf-8") as f, ThreadPoolExecutor(WORKERS) as ex:
        todo = [c for c in picked if c["_id"] not in done]
        futs = {ex.submit(work, c): c["_id"] for c in todo}
        for fut in as_completed(futs):
            if n_ok >= TARGET_N:
                break
            rec = fut.result()
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_ok += 1
                print(f"  ok {n_ok}/{TARGET_N}  {futs[fut]}")
            else:
                print(f"  discard {futs[fut]}")
    print(f"완료: {n_ok}건 → {OUT}")


if __name__ == "__main__":
    main()
