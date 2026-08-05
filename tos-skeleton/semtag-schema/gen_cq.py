#!/usr/bin/env python3
"""실험 10 — 합성 고객 구어 질의셋 생성 (고객 데이터 부재 가정 하의 대체 평가셋).

누수 통제(사전 등록):
- 변환기 입력 = 질문 텍스트만. 문서·gold·별칭 사전(Q_BRIDGE)·태그 어휘 일절 미제공
  (사전을 주면 사전이 잡을 표현을 스스로 만들게 되어 순환 검증이 됨).
- 특약·상품명은 보존 지시 — 질문의 대상 식별자가 깨지면 gold가 무효화됨.
- 생성 후 전량 동결(cq_frozen.jsonl + SHA) → 동결본만 채점. dev/test 분할은 평가
  시점에 기존 qid 분할 재사용(gold 재사용 — 신규 라벨링 없음).
- 실패 폴백: 검증(길이·한글·특약명 보존) 2회 불통과 시 원 질문 유지 + fallback 표기.

한계(정직 기록): 합성 구어가 실제 고객 발화 분포와 같다는 보장은 없음 — 결론은
"합성 고객 구어 분포"에 한정하며 실측 일반화는 미검증으로 병기한다.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out/exp10"
CACHE = OUT / "cq"

PROMPT = """보험 고객이 콜센터 상담원에게 전화로 묻는 것처럼, 아래 질문을 일상 구어체로
다시 써라. 규칙:
1. 약관 전문용어(예: 지급사유, 면책, 보장개시일 같은 딱딱한 용어)는 고객이 쓸 법한
   일상 표현으로 바꿔라.
2. 특약 이름·상품 이름은 그대로 유지하라 (예: "암진단특약"은 그대로).
3. 질문이 묻는 내용(의미)은 바꾸지 마라. 새 정보를 추가하지 마라.
4. 한 문장, 반말 금지, 자연스러운 존댓말 구어.

질문: {q}

한 줄 JSON만 출력: {{"q": "다시 쓴 질문"}}"""

TK = re.compile(r"[가-힣A-Za-z0-9()\[\]·]+특약")


def valid(orig, cq):
    if not cq or not (8 <= len(cq) <= 140):
        return False
    if not re.search(r"[가-힣]", cq):
        return False
    for tk in TK.findall(orig):
        if tk not in cq:
            return False
    return True


def gen_one(row):
    qid, orig = str(row["qid"]), row["q"]
    f = CACHE / f"{qid}.json"
    if f.exists():
        d = json.loads(f.read_text())
        if d.get("cq"):
            return qid, d
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    cq, fallback = "", False
    for attempt in range(2):
        try:
            r = subprocess.run(["claude", "-p", "--model", "haiku"],
                               input=PROMPT.format(q=orig),
                               capture_output=True, text=True, timeout=90, env=env)
            m = re.findall(r"\{[^{}]*\"q\"[\s\S]*?\}", r.stdout)
            cand = str(json.loads(m[-1]).get("q", "")).strip() if m else ""
            if valid(orig, cand):
                cq = cand
                break
        except Exception:
            pass
    if not cq:
        cq, fallback = orig, True
    d = {"qid": qid, "q_orig": orig, "cq": cq, "fallback": fallback}
    f.write_text(json.dumps(d, ensure_ascii=False))
    return qid, d


def main():
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in open(HERE / "out/gold_mapped.jsonl", encoding="utf-8")]
    print(f"input questions={len(rows)}", flush=True)
    done = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (qid, d) in enumerate(ex.map(gen_one, rows), 1):
            done[qid] = d
            if i % 20 == 0:
                print(f"progress {i}/{len(rows)}", flush=True)
    ordered = [done[str(r["qid"])] for r in rows]
    fb = sum(1 for d in ordered if d["fallback"])
    frozen = OUT / "cq_frozen.jsonl"
    frozen.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in ordered) + "\n")
    sha = hashlib.sha256(frozen.read_bytes()).hexdigest()
    (OUT / "cq_frozen.sha256").write_text(sha + "\n")
    print(f"DONE n={len(ordered)} fallback={fb} sha={sha[:16]}", flush=True)


if __name__ == "__main__":
    main()
