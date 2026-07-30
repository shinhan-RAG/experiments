"""shinhan-uw corpus.jsonl → 평가셋 (질의 / qrels / span gold).

build_shinhan_qa_v2.py 와 **생성 방식은 동일하게** 유지한다 — 같은 프롬프트,
같은 verbatim span 검증, 같은 qrels score. gold 생성 절차를 바꾸면 기존 신한
결과와 방법론이 어긋나기 때문이다. 달라진 것은 표본 추출 정책뿐이다.

표본 추출이 왜 달라야 하는가 (C8)
---------------------------------
gold 청크 563건 중 `건강 전상품`(316) + `간편예외질환`(137)이 80%다. 비례
추출하면 사실상 스프레드시트 두 개만 평가하게 된다. 그래서
  1) 문서별 round-robin — 7문서가 고르게 들어간다
  2) element_type 층화 — 표 청크가 89%지만 비표도 최소 쿼터를 보장한다
     (비표 후보가 27건뿐이라 쿼터는 상한이 아니라 목표다. 실제 층 분포는
      manifest 에 그대로 기록한다)
  3) distractor(`shinhan-legacy::`)는 gold 후보에서 제외

출력:
  data/raw/shinhan-uw/queries.jsonl   {_id, text}
  data/raw/shinhan-uw/qrels.jsonl     {query-id, corpus-id, score}
  data/raw/shinhan-uw/qa_meta.jsonl   {qid, question, answer, gold_chunk_id, doc,
                                       section_path, element_type, supporting_spans}
  data/raw/shinhan-uw/qa_manifest.json

  python scripts/build_shinhan_uw_qa.py
  python scripts/build_shinhan_uw_qa.py --n 50 --dry-run    # LLM 호출 없이 후보만 확인
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
RAW = BASE_DIR / "data" / "raw" / "shinhan-uw"
MODEL = "gpt-4o-mini"
SEED = 42
N_QUERIES = 50
PER_DOC_CAP = 10          # 문서 7개 × 10 = 70 슬롯 (50을 채우기에 충분한 여유)
MIN_CHUNK_CHARS = 150     # 이보다 짧은 청크는 구체적 질문을 만들 근거가 못 된다
# 층 목표. 비표 후보가 27건뿐이므로 상한이 아니라 '최소 확보 목표'로 쓴다.
STRATUM_TARGET = {"table": 35, "non_table": 15}

API_KEY = None
for line in (BASE_DIR / ".env").read_text(encoding="utf-8").splitlines():
    if line.startswith("OPENAI_API_KEY="):
        API_KEY = line.split("=", 1)[1].strip()

# build_shinhan_qa_v2.py 와 동일한 프롬프트 (문서 종류 표현만 언더라이팅에 맞춤)
PROMPT = """당신은 보험 문서 검색 평가셋을 만드는 전문가입니다.
아래는 신한라이프 언더라이팅 실무문서에서 뽑은 한 청크입니다.

이 청크에 근거해서 실제 상담원/설계사/심사자가 물어볼 법한 **구체적인 질문 1개**를 만들고,
정답과 **정답의 근거가 되는 원문 문장을 청크에서 그대로(verbatim) 인용**하세요.

규칙:
- 질문은 이 청크를 봐야만 답할 수 있을 만큼 구체적(상품명·담보명·질환명·조건·수치 특정).
- evidence는 반드시 **청크 원문에 그대로 존재하는 문자열**이어야 함(요약·변형 금지). 1~3개.
- 청크가 표지/목차/작성지침/OCR깨짐 등 정보성이 없으면 usable=false.
- 표 청크라면 특정 행의 값을 묻는 질문이 좋습니다(예: 특정 담보의 가입한도).
- 아래 JSON만 출력:
{"usable": true/false, "question": "...", "answer": "...", "evidence": ["원문 인용1", "원문 인용2"]}

[문서명] {doc}
[섹션] {section}
[청크]
{chunk}
"""


def call_llm(chunk: str, doc: str, section: str):
    body = json.dumps({
        "model": MODEL, "temperature": 0, "max_tokens": 600,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content":
                      PROMPT.replace("{doc}", doc[:80])
                            .replace("{section}", section[:80])
                            .replace("{chunk}", chunk[:4000])}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                out = json.load(resp)
            return json.loads(out["choices"][0]["message"]["content"])
        except Exception as exc:                      # noqa: BLE001
            if attempt == 3:
                print("  LLM fail:", str(exc)[:120])
                return None
            time.sleep(2 * (attempt + 1))
    return None


def locate_spans(chunk_text: str, evidences) -> list[dict]:
    """evidence 문자열을 청크 원문에서 찾아 char offset 부여. 못 찾으면 제외.

    build_shinhan_qa_v2.locate_spans 와 동일 동작 — span 지표 비교 가능성을
    위해 바꾸지 않는다.
    """
    spans = []
    for ev in evidences or []:
        ev = (ev or "").strip()
        if len(ev) < 8:
            continue
        idx = chunk_text.find(ev)
        if idx == -1:                                 # 공백 정규화 후 재시도
            norm = " ".join(ev.split())
            comp = " ".join(chunk_text.split())
            if comp.find(norm) == -1:
                continue
            spans.append({"text": ev, "char_start": None, "char_end": None,
                          "verbatim": False})
            continue
        spans.append({"text": ev, "char_start": idx, "char_end": idx + len(ev),
                      "verbatim": True})
    return spans


def stratum_of(chunk: dict) -> str:
    return "table" if chunk["element_type"] == "table" else "non_table"


def build_candidate_order(corpus: list[dict]) -> list[dict]:
    """문서별 round-robin + 층 쿼터 순서로 후보를 정렬한다."""
    rng = random.Random(SEED)
    pool = [c for c in corpus
            if "distractor_source" not in c
            and len(c["text"]) >= MIN_CHUNK_CHARS]
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for chunk in pool:
        by_doc[chunk["doc"]].append(chunk)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    for doc in docs:
        rng.shuffle(by_doc[doc])

    order: list[dict] = []
    used: Counter = Counter()
    quota = dict(STRATUM_TARGET)
    # 1차: 층 쿼터를 지키며 문서별 round-robin
    progressed = True
    while progressed:
        progressed = False
        for doc in docs:
            if used[doc] >= PER_DOC_CAP:
                continue
            for chunk in by_doc[doc]:
                if chunk in order:
                    continue
                stratum = stratum_of(chunk)
                if quota.get(stratum, 0) <= 0:
                    continue
                order.append(chunk)
                quota[stratum] -= 1
                used[doc] += 1
                progressed = True
                break
    # 2차: 쿼터를 다 쓴 뒤에도 후보가 필요하므로 남은 것을 같은 순서로 덧붙인다
    #      (LLM 이 usable=false 를 내면 후보가 더 필요하다)
    remaining = [c for c in pool if c not in order]
    remaining.sort(key=lambda c: (used[c["doc"]], c["_id"]))
    return order + remaining


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_QUERIES)
    ap.add_argument("--dry-run", action="store_true",
                    help="LLM 호출 없이 후보 선정 결과만 출력")
    args = ap.parse_args()

    corpus = [json.loads(line) for line in
              open(RAW / "corpus.jsonl", encoding="utf-8")]
    order = build_candidate_order(corpus)
    head = order[: args.n]
    print(f"=== shinhan-uw 평가셋 빌드 (목표 {args.n}질의) ===")
    print(f"  gold 후보 {len(order):,}건 (>= {MIN_CHUNK_CHARS}자, distractor 제외)")
    print(f"  앞 {len(head)}건 문서 분포: "
          f"{dict(Counter(c['doc'][:22] for c in head))}")
    print(f"  앞 {len(head)}건 층 분포  : "
          f"{dict(Counter(stratum_of(c) for c in head))}")
    if args.dry_run:
        return 0

    if not API_KEY:
        raise SystemExit("OPENAI_API_KEY 없음 (.env 확인)")

    queries, qrels, meta = [], [], []
    attempted = rejected = 0
    for chunk in order:
        if len(queries) >= args.n:
            break
        attempted += 1
        res = call_llm(chunk["text"], chunk["doc"], chunk.get("section_path", ""))
        if not res or not res.get("usable") or not res.get("question"):
            rejected += 1
            continue
        spans = locate_spans(chunk["text"], res.get("evidence"))
        # 최소 1개는 원문 그대로여야 한다 — 근거 span 의 신뢰성 확보
        if not any(s["verbatim"] for s in spans):
            rejected += 1
            continue
        qid = str(len(queries) + 1)
        queries.append({"_id": qid, "text": res["question"].strip()})
        qrels.append({"query-id": int(qid), "corpus-id": chunk["_id"], "score": 2})
        meta.append({
            "qid": qid,
            "question": res["question"].strip(),
            "answer": (res.get("answer") or "").strip(),
            "gold_chunk_id": chunk["_id"],
            "doc": chunk["doc"],
            "section_path": chunk.get("section_path", ""),
            "element_type": chunk["element_type"],
            "supporting_spans": spans,
        })
        vb = sum(s["verbatim"] for s in spans)
        print(f"  {len(queries):>2}/{args.n} q{qid} [{chunk['element_type']:6s}] "
              f"span {len(spans)}(verbatim {vb}): {res['question'][:48]}")

    for name, rows in (("queries.jsonl", queries), ("qrels.jsonl", qrels),
                       ("qa_meta.jsonl", meta)):
        with open(RAW / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    corpus_ids = {c["_id"] for c in corpus}
    gold_ids = {e["corpus-id"] for e in qrels}
    assert gold_ids <= corpus_ids, "corpus 에 없는 gold 가 있다"
    assert not any("::" in g for g in gold_ids), "distractor 가 gold 로 뽑혔다"
    assert len(gold_ids) == len(qrels), "같은 청크가 두 번 gold 가 됐다"

    total_spans = sum(len(m["supporting_spans"]) for m in meta)
    verbatim = sum(s["verbatim"] for m in meta for s in m["supporting_spans"])
    manifest = {
        "dataset": "shinhan-uw",
        "generated_by": "scripts/build_shinhan_uw_qa.py",
        "model": MODEL,
        "seed": SEED,
        "queries": len(queries),
        "candidates_attempted": attempted,
        "candidates_rejected": rejected,
        "min_chunk_chars": MIN_CHUNK_CHARS,
        "per_doc_cap": PER_DOC_CAP,
        "stratum_target": STRATUM_TARGET,
        "stratum_actual": dict(Counter(m["element_type"] for m in meta)),
        "by_doc": dict(Counter(m["doc"] for m in meta)),
        "supporting_spans": {"total": total_spans, "verbatim": verbatim},
        "limitations": [
            "gold 는 gpt-4o-mini 단일 생성이며 사람 검수가 없다.",
            "질의당 gold 청크 1건이므로 recall 이 1/50 = 2%p 단위로만 움직인다. "
            "'효과 없음'과 '표본 부족으로 판정 불가'를 구분해 서술할 것.",
            "비표 후보가 27건뿐이라 층 목표(비표 15)를 채우지 못할 수 있다. "
            "실제 층 분포는 stratum_actual 을 볼 것.",
            "질문이 gold 청크로부터 생성되었으므로 어휘 중복이 크다 — 검색이 "
            "실제보다 쉬울 수 있다(기존 신한·법률 평가셋과 동일한 한계).",
        ],
    }
    with open(RAW / "qa_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n  === 자기 검증 ===")
    print(f"    [OK] 질의 {len(queries)}건, gold 전부 corpus 에 존재, distractor 0건")
    print(f"    span {total_spans}개 (원문정확 {verbatim}/{total_spans})")
    print(f"    층 분포: {manifest['stratum_actual']}")
    print(f"    문서 분포: {manifest['by_doc']}")
    print(f"    후보 시도 {attempted} / 거절 {rejected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
