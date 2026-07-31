"""paper-ha corpus.jsonl → 평가셋 50문항 (build_shinhan_qa_v2 방식 + 병렬화 + 요소 층화).

바뀐 점:
- 표/차트/그림 캡션 청크에서 최소 CAPTION_MIN(15)문항을 강제 할당
  → "요소 태그가 검색에 도움이 되는가"를 질의 축에서도 측정할 수 있게 한다.
- ThreadPoolExecutor 병렬 호출(기본 16)로 후보 전체를 한 번에 생성 → 수 분 내 완료.
- gold = 청크(qrels), supporting span = 청크 원문 verbatim 인용 (신한 v2와 동일 계약).

출력: data/raw/paper-ha/queries.jsonl / qrels.jsonl / qa_meta.jsonl

  python scripts/build_paper_ha_qa.py [--concurrency 16]
"""
import argparse
import json
import random
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "paper-ha"
MODEL = "gpt-4o-mini"
SEED = 42
N_QUERIES = 50
CAPTION_MIN = 15          # 표/차트/그림 캡션 청크 최소 문항 수
PER_DOC_CAP = 4
CAPTION_CANDIDATES = 45   # 리젝트 대비 후보 여유분
TEXT_CANDIDATES = 110

API_KEY = None
for line in (Path(__file__).resolve().parent.parent / ".env").read_text().splitlines():
    if line.startswith("OPENAI_API_KEY="):
        API_KEY = line.split("=", 1)[1].strip()

PROMPT = """당신은 학술논문 검색 평가셋을 만드는 전문가입니다.
아래는 한국 학술논문(인문학·예술체육학)에서 뽑은 한 청크입니다.
(주의: 원문에 어절 간 공백이 빠진 구간이 있을 수 있습니다. 인용은 그 표기 그대로 하세요.)

이 청크에 근거해서 연구자/학생이 논문을 검색하며 물어볼 법한 **구체적인 질문 1개**를 만들고,
정답과 **정답의 근거가 되는 원문 문장을 청크에서 그대로(verbatim) 인용**하세요.

규칙:
- 질문은 이 청크를 봐야만 답할 수 있을 만큼 구체적(인물·개념·주장·수치·표/그림 내용 특정).
- 질문에 "이 청크/이 논문/위 글" 같은 지시어를 쓰지 말고, 독립적으로 검색 가능한 질문으로.
- evidence는 반드시 **청크 원문에 그대로 존재하는 문자열**이어야 함(요약·변형·공백수정 금지). 1~3개.
- 청크가 목차/서지사항/인사말 등 정보성이 없으면 usable=false.
- 아래 JSON만 출력:
{"usable": true/false, "question": "...", "answer": "...", "evidence": ["원문 인용1", "원문 인용2"]}

[논문] {doc}
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
    for attempt in range(4):
        try:
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions", data=body,
                headers={"Authorization": f"Bearer {API_KEY}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.load(r)
            return json.loads(out["choices"][0]["message"]["content"])
        except Exception as e:
            if attempt == 3:
                print("  LLM fail:", str(e)[:120])
                return None
            time.sleep(2 * (attempt + 1))


def locate_spans(chunk_text, evidences):
    """evidence를 청크 원문에서 찾아 char offset 부여 (신한 v2와 동일)."""
    spans = []
    for ev in evidences or []:
        ev = (ev or "").strip()
        if len(ev) < 8:
            continue
        idx = chunk_text.find(ev)
        if idx == -1:
            norm = " ".join(ev.split())
            comp = " ".join(chunk_text.split())
            if comp.find(norm) == -1:
                continue
            spans.append({"text": ev, "char_start": None, "char_end": None, "verbatim": False})
            continue
        spans.append({"text": ev, "char_start": idx, "char_end": idx + len(ev), "verbatim": True})
    return spans


def pick_candidates(corpus, rng):
    caption_pool = [d for d in corpus
                    if d["element_type"] in ("table", "chart", "figure") and len(d["text"]) >= 80]
    text_pool = [d for d in corpus
                 if d["element_type"] == "text" and len(d["text"]) >= 200]
    rng.shuffle(caption_pool)
    rng.shuffle(text_pool)
    return caption_pool[:CAPTION_CANDIDATES], text_pool[:TEXT_CANDIDATES]


def main():
    assert API_KEY, "OPENAI_API_KEY 없음 (dr-dci/.env)"
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    corpus = [json.loads(l) for l in open(RAW / "corpus.jsonl", encoding="utf-8")]
    rng = random.Random(SEED)
    captions, texts = pick_candidates(corpus, rng)
    candidates = captions + texts
    print(f"후보: 캡션 {len(captions)} + 본문 {len(texts)} = {len(candidates)} "
          f"(동시 {args.concurrency})")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(
            lambda d: call_llm(d["text"], d["doc"], d["section"]), candidates))
    print(f"LLM 생성 완료: {time.time() - t0:.0f}s")

    def usable(doc, res):
        if not res or not res.get("usable") or not res.get("question"):
            return None
        spans = locate_spans(doc["text"], res.get("evidence"))
        if not any(s["verbatim"] for s in spans):
            return None
        return spans

    queries, qrels, meta = [], [], []
    used_per_doc = defaultdict(int)
    qid = 0

    def take(doc, res, spans):
        nonlocal qid
        qid += 1
        q = str(qid)
        queries.append({"_id": q, "text": res["question"].strip()})
        qrels.append({"query-id": qid, "corpus-id": doc["_id"], "score": 2})
        meta.append({
            "qid": q, "question": res["question"].strip(),
            "answer": (res.get("answer") or "").strip(),
            "gold_chunk_id": doc["_id"], "doc": doc["doc"],
            "element_type": doc["element_type"], "supporting_spans": spans,
        })
        used_per_doc[doc["doc"]] += 1
        vb = sum(s["verbatim"] for s in spans)
        print(f"  {len(queries)}/{N_QUERIES} q{qid} [{doc['element_type']}] "
              f"span:{len(spans)}(verbatim {vb}): {res['question'][:45]}")

    n_cap = len(captions)
    # 1) 캡션 청크에서 CAPTION_MIN 우선 확보
    for doc, res in zip(captions, results[:n_cap]):
        if sum(1 for m in meta if m["element_type"] != "text") >= CAPTION_MIN:
            break
        if used_per_doc[doc["doc"]] >= PER_DOC_CAP:
            continue
        spans = usable(doc, res)
        if spans:
            take(doc, res, spans)
    n_caption_q = len(queries)
    if n_caption_q < CAPTION_MIN:
        print(f"  [!] 캡션 문항 {n_caption_q} < 목표 {CAPTION_MIN} (후보 소진)")

    # 2) 본문 청크로 50까지 채움
    for doc, res in zip(texts, results[n_cap:]):
        if len(queries) >= N_QUERIES:
            break
        if used_per_doc[doc["doc"]] >= PER_DOC_CAP:
            continue
        spans = usable(doc, res)
        if spans:
            take(doc, res, spans)

    if len(queries) < N_QUERIES:
        print(f"  [!] {len(queries)}문항으로 종료 (목표 {N_QUERIES}, 후보 수를 늘릴 것)")

    with open(RAW / "queries.jsonl", "w", encoding="utf-8") as f:
        for x in queries:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(RAW / "qrels.jsonl", "w", encoding="utf-8") as f:
        for x in qrels:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(RAW / "qa_meta.jsonl", "w", encoding="utf-8") as f:
        for x in meta:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    by_et = defaultdict(int)
    for m in meta:
        by_et[m["element_type"]] += 1
    tot_span = sum(len(m["supporting_spans"]) for m in meta)
    vb = sum(s["verbatim"] for m in meta for s in m["supporting_spans"])
    print(f"\n{len(queries)}문항, 요소 분포 {dict(by_et)}, "
          f"span {tot_span}개 (verbatim {vb}/{tot_span})")


if __name__ == "__main__":
    main()
