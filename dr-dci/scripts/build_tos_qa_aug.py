"""
3a단계: 359 QA 증강 트랙 (목표 ~60개).

원리 — "약관은 공통 뼈대 + 문서별 값만 다르다":
1. 정답셋_359_최종_v4.xlsx에서 (질문, 정답, 근거 인용) 추출 (stdlib zipfile+XML — pandas 없음)
2. 근거 인용문을 대상 약관의 청크에서 공백 정규화 문자열 매칭으로 탐색
   → 같은 조항이 실존하는 (QA, 대상 청크) 쌍만 증강 후보
3. claude CLI(haiku)로 질문·정답을 대상 문서에 맞게 치환 (상품명·특약명·수치)
4. 근거는 반드시 청크 원문 verbatim → locate_spans 검증 실패 시 폐기

출력: data/raw/shinhan-tos/qa_aug_raw.jsonl (트랙별 중간 산출물, gate에서 병합)
"""
import json
import random
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree as ET

from tos_llm import call_claude_json, locate_spans

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "raw" / "shinhan-tos"
XLSX = BASE.parents[0] / "정답셋_359_최종_v4.xlsx"
OUT = DATA_DIR / "qa_aug_raw.jsonl"

SEED = 42
TARGET_N = 200
PER_DOC_CAP = 70       # 한 문서에 QA 과다 집중 방지 (매칭 쌍 소진으로 상향)
MATCH_PREFIX = 30      # 근거 인용 앞부분 n자(공백 제거)로 매칭
WORKERS = 6

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
T_TAG = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"

PROMPT = """당신은 보험 문서 검색 평가셋을 만드는 전문가입니다.
아래 [원본 QA]는 다른 상품의 약관으로 만든 질문·정답입니다.
[대상 문서]의 [청크]에 같은 성격의 조항이 있으므로, 원본 질문을 **대상 문서에 맞게 이식**하세요.

규칙:
- 질문은 원본의 의도·유형을 유지하되 상품명·특약명·조건을 대상 문서 기준으로 자연스럽게 바꿀 것.
- 질문에 파일명·정식 문서명 전체를 넣지 말 것. "(무배당...)", "(갱신형)", "해약환급금 미지급형" 같은
  괄호 부가정보 금지. 고객이 실제 부르듯 짧은 상품명으로 지칭 (예: "신한통합건강보장보험 원",
  "또받는생활비암보험"). 상품 특정이 불필요한 질문이면 상품명 없이 물어도 됨.
- 정답은 **오직 [청크] 내용에만 근거**해 작성. 원본 정답을 베끼지 말 것 (값이 다를 수 있음).
- evidence는 **[청크] 원문에 그대로 존재하는 문자열** 1~2개 (요약·변형 금지).
- 청크만으로 그 질문에 답할 수 없으면 usable=false.
- 아래 JSON만 출력:
{"usable": true/false, "question": "...", "answer": "...", "evidence": ["원문 인용1"]}

[원본 QA]
질문: {orig_q}
정답: {orig_a}

[대상 문서] {doc}
[청크]
{chunk}
"""


def read_xlsx_rows():
    z = zipfile.ZipFile(XLSX)
    sh = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = sh.find("a:sheetData", NS)

    def val(c):
        v = c.find("a:v", NS)
        if v is None:
            it = c.find("a:is", NS)
            return "".join(t.text or "" for t in it.iter(T_TAG)) if it is not None else ""
        return v.text or ""

    out = []
    for r in list(rows)[1:]:
        d = {c.get("r")[0]: val(c) for c in r.findall("a:c", NS)}
        evs = []
        for line in (d.get("G") or "").split("\n"):
            line = line.strip()
            if line.startswith("[L") and "]" in line:
                q = line.split("]", 1)[1].strip()
                if len(q) >= 15:
                    evs.append(q)
        out.append({"no": d.get("A"), "biz": d.get("B", ""), "q": d.get("C", ""),
                    "a": d.get("D", ""), "conf": d.get("E", ""), "evs": evs})
    return out


def main():
    qa_rows = [r for r in read_xlsx_rows() if r["conf"] == "확정" and r["evs"]]
    print(f"증강 소스 QA: {len(qa_rows)}건 (확정 + 근거 보유)")

    chunks_by_doc = defaultdict(list)
    doc_meta = {}
    with open(DATA_DIR / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks_by_doc[c["doc"]].append(c)
    with open(DATA_DIR / "docs_selected.json", encoding="utf-8") as f:
        for d in json.load(f)["docs"]:
            doc_meta[d["name"]] = d
    # 증강 대상: 약관 유형 문서 (공통 뼈대 가설이 성립하는 범위)
    targets = [doc for doc, m in doc_meta.items() if m["type"] == "약관"]
    print(f"대상 약관 문서: {len(targets)}건")

    # 사전 매칭: (QA, 문서) → 근거 인용이 실존하는 청크
    norm = lambda s: "".join(s.split())
    norm_chunks = {doc: [(c, norm(c["text"])) for c in chunks_by_doc[doc]] for doc in targets}
    rng = random.Random(SEED)
    rng.shuffle(qa_rows)

    pairs, per_doc = [], defaultdict(int)
    for row in qa_rows:
        for doc in sorted(targets, key=lambda d: per_doc[d]):  # 커버리지 균등화
            if per_doc[doc] >= PER_DOC_CAP:
                continue
            hit = None
            for ev in row["evs"]:          # 근거 인용 아무거나 실존하면 매칭
                key = norm(ev)[:MATCH_PREFIX]
                if len(key) < 20:
                    continue
                hit = next((c for c, nt in norm_chunks[doc] if key in nt), None)
                if hit is not None:
                    break
            if hit is not None:
                pairs.append((row, doc, hit))
                per_doc[doc] += 1
                break
        if len(pairs) >= TARGET_N * 2:   # LLM 폐기율 감안 여유분
            break
    print(f"매칭된 (QA,문서,청크) 쌍: {len(pairs)}  문서별: {dict(per_doc)}")

    done_keys = set()
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done_keys.add((r["orig_no"], r["doc"]))

    def work(row, doc, chunk):
        prompt = (PROMPT.replace("{orig_q}", row["q"]).replace("{orig_a}", row["a"][:800])
                  .replace("{doc}", doc[:80]).replace("{chunk}", chunk["text"][:3500]))
        res = call_claude_json(prompt)
        if not res or not res.get("usable"):
            return None
        spans = locate_spans(chunk["text"], res.get("evidence"))
        if not spans:
            return None
        return {"source": "aug", "orig_no": row["no"], "qa_type": row["biz"],
                "question": res["question"], "answer": res["answer"],
                "gold_chunk_id": chunk["_id"], "doc": doc,
                "element_type": chunk["element_type"], "supporting_spans": spans}

    n_ok = len(done_keys)
    with open(OUT, "a", encoding="utf-8") as f, ThreadPoolExecutor(WORKERS) as ex:
        todo = [(r, d, c) for r, d, c in pairs if (r["no"], d) not in done_keys]
        futs = {ex.submit(work, r, d, c): (r["no"], d) for r, d, c in todo}
        for fut in as_completed(futs):
            if n_ok >= TARGET_N:
                break
            rec = fut.result()
            tag = futs[fut]
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                n_ok += 1
                print(f"  ok {n_ok}/{TARGET_N}  #{tag[0]} → {tag[1][:40]}")
            else:
                print(f"  discard #{tag[0]} → {tag[1][:40]}")
    print(f"완료: {n_ok}건 → {OUT}")


if __name__ == "__main__":
    main()
