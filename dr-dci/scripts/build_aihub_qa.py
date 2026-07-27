"""
AI-Hub corpus → 청킹-무관 평가셋 (512·element 둘 다에 재사용).

- 층화 샘플링: 도메인(의료/법률) × 카테고리 문서수 비례 배분, 문서당 1문항
- 정답을 '원본 문서(parent) 속 근거 span'으로 기록 → 청킹과 무관
- gold 청크는 실험 시 각 청킹에서 span 포함 청크로 자동 유도(build 단계선 span만 확정)

출력: data/aihub/qa/
  queries.jsonl   {_id, text}
  qa_meta.jsonl   {qid, question, answer, parent_id, domain, category,
                   supporting_spans:[{text,char_start,char_end}]}
※ qrels는 청킹별로 파생(별도 스크립트) — span이 어느 청크에 들어가는지로 gold 결정.
"""
import json, os, re, random, time, urllib.request
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "aihub" / "qa"
PARENTS = BASE / "data" / "aihub" / "parents.jsonl"          # 원본 텍스트 단일 기준
META = BASE / "data" / "aihub" / "smoke" / "parent_meta.jsonl"
API_KEY = next((l.split("=",1)[1].strip() for l in (BASE/".env").read_text().splitlines()
                if l.startswith("OPENAI_API_KEY=")), None)
MODEL = "gpt-4o-mini"; SEED = 42; N_QUERIES = 50

PROMPT = """당신은 법률·의료 문서 검색 평가셋을 만드는 전문가입니다.
아래는 한 문서(일부)입니다. 이 내용에 근거해 실무자가 물어볼 **구체적 질문 1개**를 만들고,
정답과 **정답 근거가 되는 원문 문장을 그대로(verbatim) 인용**하세요.

규칙:
- 질문은 이 문서를 봐야만 답할 수 있을 만큼 구체적(사건·조항·수치·개념 특정).
- evidence는 반드시 **원문에 그대로 존재하는 문자열**(요약·변형 금지), 1~2개.
- 정보성 없으면(목차/형식만) usable=false.
- 아래 JSON만 출력:
{"usable": true/false, "question":"...", "answer":"...", "evidence":["원문 인용"]}

[도메인] {domain}  [분야] {category}
[문서]
{text}
"""


def call_llm(text, domain, category):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": 500,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": PROMPT
            .replace("{domain}", domain).replace("{category}", category[:40])
            .replace("{text}", text[:4000])}]}).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(json.load(r)["choices"][0]["message"]["content"])
        except Exception as e:
            if a == 3:
                print("  LLM fail:", str(e)[:100]); return None
            time.sleep(2 * (a + 1))


def _ns_map(raw):
    o, idx = [], []
    for i, ch in enumerate(raw):
        if not ch.isspace():
            o.append(ch); idx.append(i)
    return "".join(o), idx


def locate_all(raw, raw_ns, idx, ev):
    """evidence의 원본 내 '모든' 등장 구간 [(s,e),...]. 공백차이 무시. 중복 정답 대비."""
    occ = []
    ev = (ev or "").strip()
    if len(ev) < 8:
        return occ
    # 1) 그대로 모든 위치
    start = 0
    while True:
        i = raw.find(ev, start)
        if i < 0:
            break
        occ.append((i, i + len(ev))); start = i + 1
    if occ:
        return occ
    # 2) 공백제거 후 모든 위치 → 원본 오프셋 매핑
    evns = re.sub(r"\s+", "", ev)
    if len(evns) < 8:
        return occ
    start = 0
    while True:
        j = raw_ns.find(evns, start)
        if j < 0:
            break
        occ.append((idx[j], idx[j + len(evns) - 1] + 1)); start = j + 1
    return occ


def main():
    assert API_KEY
    OUT.mkdir(parents=True, exist_ok=True)
    # 원본 텍스트 단일 기준 (parents.jsonl)
    fulltext = {json.loads(l)["parent_id"]: json.loads(l)["text"] for l in open(PARENTS)}
    meta = {json.loads(l)["parent_id"]: json.loads(l) for l in open(META)}

    # 층화: 도메인×카테고리 비례. 문서당 1문항, 카테고리별 최소 1 보장.
    rng = random.Random(SEED)
    by_key = defaultdict(list)
    for pid, m in meta.items():
        by_key[(m["domain"], m["category"])].append(pid)
    keys = list(by_key)
    total = sum(len(v) for v in by_key.values())
    # 비례 배분 + 최소 1
    alloc = {}
    for k in keys:
        alloc[k] = max(1, round(N_QUERIES * len(by_key[k]) / total))
    # 합을 N_QUERIES에 맞춤
    while sum(alloc.values()) > N_QUERIES:
        k = max(alloc, key=lambda x: alloc[x]); alloc[k] -= 1
    while sum(alloc.values()) < N_QUERIES:
        k = max(keys, key=lambda x: len(by_key[x])); alloc[k] += 1

    queries, qa_meta = [], []
    qid = 0
    for k in sorted(keys, key=lambda x: -len(by_key[x])):
        pool = by_key[k][:]; rng.shuffle(pool)
        need = alloc[k]; made = 0
        for pid in pool:
            if made >= need or len(queries) >= N_QUERIES:
                break
            m = meta[pid]
            raw = fulltext[pid]
            res = call_llm(raw, m["domain"], m["category"])
            if not res or not res.get("usable") or not res.get("question"):
                continue
            raw_ns, idx = _ns_map(raw)
            spans = []
            for ev in res.get("evidence") or []:
                occ = locate_all(raw, raw_ns, idx, ev)
                if occ:
                    spans.append({"text": ev.strip(), "occurrences": occ})
            if not spans:                       # 원본에서 근거를 못 찾으면 제외
                continue
            qid += 1; q = str(qid)
            queries.append({"_id": q, "text": res["question"].strip()})
            qa_meta.append({"qid": q, "question": res["question"].strip(),
                            "answer": res.get("answer", "").strip(), "parent_id": pid,
                            "domain": m["domain"], "category": m["category"],
                            "supporting_spans": spans})
            made += 1
            print(f"  {len(queries)}/{N_QUERIES} [{m['domain']}/{m['category'][:12]}] {res['question'][:38]}")

    with open(OUT / "queries.jsonl", "w", encoding="utf-8") as f:
        for x in queries: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(OUT / "qa_meta.jsonl", "w", encoding="utf-8") as f:
        for x in qa_meta: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"\n{len(queries)}문항 | 도메인:", dict(Counter(m['domain'] for m in qa_meta)),
          "| 커버 카테고리:", len({m['category'] for m in qa_meta}))


if __name__ == "__main__":
    main()
