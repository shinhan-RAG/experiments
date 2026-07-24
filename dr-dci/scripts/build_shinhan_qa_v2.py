"""
신한 corpus.jsonl → 평가셋 v2 (원본 실험 방법 유지 + 청크단위·span 지표 지원).

원본 실험이 보려던 것(augmentation 적층 효과)은 그대로. 바뀐 것:
- gold = 정답이 든 '청크'(문서 아님)  → qrels 기반 Recall@k/nDCG
- supporting span = 정답 근거가 되는 원문 문장(그대로 인용) → coverage/density/F1
두 라벨을 한 번의 생성으로 확보(근거 문장은 gold 청크 안에 있음).

출력:
  data/raw/shinhan/queries.jsonl    {_id, text}
  data/raw/shinhan/qrels.jsonl      {query-id, corpus-id(gold 청크), score}
  data/raw/shinhan/qa_meta.jsonl    {qid, question, answer, gold_chunk_id, doc,
                                     element_type, supporting_spans:[{text,char_start,char_end}]}
"""
import json, os, random, time, urllib.request
from pathlib import Path
from collections import defaultdict

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "shinhan"
API_KEY = None
for line in (Path(__file__).resolve().parent.parent / ".env").read_text().splitlines():
    if line.startswith("OPENAI_API_KEY="):
        API_KEY = line.split("=", 1)[1].strip()
MODEL = "gpt-4o-mini"
SEED = 42
N_QUERIES = 50
PER_DOC_CAP = 4

PROMPT = """당신은 보험 문서 검색 평가셋을 만드는 전문가입니다.
아래는 신한라이프 보험 문서에서 뽑은 한 청크입니다.

이 청크에 근거해서 실제 상담원/고객이 물어볼 법한 **구체적인 질문 1개**를 만들고,
정답과 **정답의 근거가 되는 원문 문장을 청크에서 그대로(verbatim) 인용**하세요.

규칙:
- 질문은 이 청크를 봐야만 답할 수 있을 만큼 구체적(상품명·항목·조건·수치 특정).
- evidence는 반드시 **청크 원문에 그대로 존재하는 문자열**이어야 함(요약·변형 금지). 1~3개.
- 청크가 표지/목차/인사말/OCR깨짐 등 정보성이 없으면 usable=false.
- 아래 JSON만 출력:
{"usable": true/false, "question": "...", "answer": "...", "evidence": ["원문 인용1", "원문 인용2"]}

[문서명] {doc}
[섹션] {section}
[청크]
{chunk}
"""


def call_llm(chunk, doc, section):
    body = json.dumps({
        "model": MODEL, "temperature": 0, "max_tokens": 600,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content":
            PROMPT.replace("{doc}", doc[:80]).replace("{section}", section[:80])
                  .replace("{chunk}", chunk[:4000])}],
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.load(r)
            return json.loads(out["choices"][0]["message"]["content"])
        except Exception as e:
            if attempt == 3:
                print("  LLM fail:", str(e)[:120]); return None
            time.sleep(2 * (attempt + 1))


def locate_spans(chunk_text, evidences):
    """evidence 문자열을 청크 원문에서 찾아 char offset 부여. 못 찾으면 제외."""
    spans = []
    for ev in evidences or []:
        ev = (ev or "").strip()
        if len(ev) < 8:
            continue
        idx = chunk_text.find(ev)
        if idx == -1:  # 공백 정규화 후 재시도
            norm = " ".join(ev.split())
            comp = " ".join(chunk_text.split())
            j = comp.find(norm)
            if j == -1:
                continue
            spans.append({"text": ev, "char_start": None, "char_end": None, "verbatim": False})
            continue
        spans.append({"text": ev, "char_start": idx, "char_end": idx + len(ev), "verbatim": True})
    return spans


def main():
    assert API_KEY, "OPENAI_API_KEY 없음"
    corpus = [json.loads(l) for l in open(RAW / "corpus.jsonl")]
    rng = random.Random(SEED)
    by_doc = defaultdict(list)
    for d in corpus:
        by_doc[d["doc"]].append(d)
    docs = list(by_doc); rng.shuffle(docs)
    for d in docs: rng.shuffle(by_doc[d])
    order, used = [], defaultdict(int)
    while len(order) < len(corpus):
        moved = False
        for d in docs:
            if used[d] < min(PER_DOC_CAP, len(by_doc[d])):
                order.append(by_doc[d][used[d]]); used[d] += 1; moved = True
        if not moved:
            break

    queries, qrels, meta = [], [], []
    qid = 0
    for doc in order:
        if len(queries) >= N_QUERIES:
            break
        res = call_llm(doc["text"], doc["doc"], doc["section"])
        if not res or not res.get("usable") or not res.get("question"):
            continue
        spans = locate_spans(doc["text"], res.get("evidence"))
        # 최소 1개는 원문 그대로(verbatim)여야 함 — 근거 span의 신뢰성 확보
        if not any(s["verbatim"] for s in spans):
            continue
        qid += 1; q = str(qid)
        queries.append({"_id": q, "text": res["question"].strip()})
        qrels.append({"query-id": qid, "corpus-id": doc["_id"], "score": 2})
        meta.append({
            "qid": q, "question": res["question"].strip(), "answer": res.get("answer", "").strip(),
            "gold_chunk_id": doc["_id"], "doc": doc["doc"], "element_type": doc["element_type"],
            "supporting_spans": spans,
        })
        vb = sum(s["verbatim"] for s in spans)
        print(f"  {len(queries)}/{N_QUERIES} q{qid} [{doc['element_type']}] span:{len(spans)}(verbatim {vb}): {res['question'][:45]}")

    with open(RAW / "queries.jsonl", "w") as f:
        for x in queries: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(RAW / "qrels.jsonl", "w") as f:
        for x in qrels: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(RAW / "qa_meta.jsonl", "w") as f:
        for x in meta: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    tot_span = sum(len(m["supporting_spans"]) for m in meta)
    vb = sum(s["verbatim"] for m in meta for s in m["supporting_spans"])
    print(f"\n{len(queries)}개 질문, span {tot_span}개 (원문정확 {vb}/{tot_span})")


if __name__ == "__main__":
    main()
