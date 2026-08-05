#!/usr/bin/env python3
"""실험 11 재순위 하네스 — listwise permutation (PREREG_EXP11.md 사전 등록 준수).

결정성 한계(명시): LLM 응답은 재계산 재현이 아니라 **캐시 replay** 재현이다.
전 요청·응답을 out/exp11/cache.jsonl에 동결하며, 동일 캐시로는 전 파이프라인이
바이트 동일하게 재생된다. 기존 실험들의 "2-pass SHA 일치"(순수 재계산)와 성질이
다르므로 동급으로 서술하지 않는다.

usage:
  python3 exp11_rerank.py dry            # 토큰 추정(LLM 호출 없음)
  python3 exp11_rerank.py R1|R2|R3|R4    # 런 실행
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import semtag_experiment as SE
from context_v2 import annotate_v2

HERE = Path(__file__).parent
OUT = HERE / "out/exp11"
DOC = ("/Users/donggyu/Documents/논문/신한라이프/신한RAG_QA셋_원본문서_20260710/원본문서/md/"
       "판매약관_신한(간편가입)통합건강보험 원(ONE)(무배당, 해약환급금 미지급형)_260507.md")
MODEL = "haiku"
COST_CAP_INPUT_TOKENS = 8_000_000  # 사전 등록 상한(커밋 전 확정) — 실측 usage 초과 시 중단
RUNS = {"R1": dict(K=20, reverse=False), "R2": dict(K=50, reverse=False),
        "R3": dict(K=100, reverse=False), "R4": dict(K=50, reverse=True)}

PROMPT = """당신은 보험약관 검색 결과의 재순위기다. 질문에 대한 정답 조항일 가능성이
높은 순서로 아래 후보들을 재배열하라. 각 후보는 "주소 헤더 + 본문 일부"다.

질문: {q}

{cands}

규칙: 후보 내용에 근거해 판단하고, 번호 순서(제시 순서)에 의존하지 마라.
한 줄 JSON만 출력: {{"order": [가장 관련 높은 후보 번호부터, {n}개 전부]}}"""

_cache_lock = threading.Lock()
_usage_lock = threading.Lock()
_usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cache_hits": 0}


def code_commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=HERE).stdout.strip()


def load_world():
    lines = SE.nfc(Path(DOC).read_text(encoding="utf-8", errors="ignore")).splitlines()
    els = annotate_v2(SE.split_elements(lines), lines)
    idx = {e["eid"]: i for i, e in enumerate(els)}
    mapped = {str(m["qid"]): m for m in
              (json.loads(l) for l in open(HERE / "out/gold_mapped.jsonl", encoding="utf-8"))}
    return els, idx, mapped


def header(e, ord_):
    h = f"{e.get('scope') or '주계약'} > {e.get('pyeon') or ''} {e.get('jo') or ''} [{e['type']}] #{ord_}"
    h = re.sub(r"\s+", " ", h).strip()
    return h[:80]


def body(e):
    return re.sub(r"\s+", " ", e["text"]).strip()[:400]


def cand_block(els, idx, eids):
    parts = []
    for i, eid in enumerate(eids, 1):
        e = els[idx[eid]]
        parts.append(f"[{i}] {header(e, idx[eid])}\n{body(e)}")
    return "\n\n".join(parts)


def cache_get(key):
    f = OUT / "cache.jsonl"
    if not f.exists():
        return None
    for l in f.read_text(encoding="utf-8").splitlines():
        d = json.loads(l)
        if d["key"] == key:
            return d["response"]
    return None


def cache_put(key, prompt, response):
    with _cache_lock:
        with (OUT / "cache.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "model": MODEL,
                                 "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest(),
                                 "prompt": prompt, "response": response},
                                ensure_ascii=False) + "\n")


def llm(prompt):
    key = hashlib.sha256((MODEL + "\x1f" + prompt).encode()).hexdigest()
    hit = cache_get(key)
    if hit is not None:
        with _usage_lock:
            _usage["cache_hits"] += 1
        return hit
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    r = subprocess.run(["claude", "-p", "--model", MODEL, "--output-format", "json"],
                       input=prompt, capture_output=True, text=True, timeout=180, env=env)
    text, usage = r.stdout, {}
    try:
        d = json.loads(r.stdout)
        text = d.get("result", r.stdout)
        usage = d.get("usage", {}) or {}
    except Exception:
        pass
    with _usage_lock:
        _usage["calls"] += 1
        _usage["input_tokens"] += usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
        _usage["output_tokens"] += usage.get("output_tokens", 0)
        if _usage["input_tokens"] > COST_CAP_INPUT_TOKENS:
            raise SystemExit(f"비용 상한 초과({_usage['input_tokens']}) — 중단")
    cache_put(key, prompt, text)
    return text


def parse_order(text, n):
    m = re.findall(r"\{[^{}]*\"order\"[\s\S]*?\}", text)
    if not m:
        return None
    try:
        arr = json.loads(m[-1])["order"]
        seen, out = set(), []
        for x in arr:
            x = int(x)
            if 1 <= x <= n and x not in seen:
                seen.add(x)
                out.append(x)
        out += [i for i in range(1, n + 1) if i not in seen]
        return out
    except Exception:
        return None


def rerank_window(q, els, idx, eids):
    """창 1개 listwise 재배열 — 실패 시 1회 재시도, 그래도 실패면 원 순서(기록)."""
    prompt = PROMPT.format(q=q, cands=cand_block(els, idx, eids), n=len(eids))
    for _ in range(2):
        order = parse_order(llm(prompt), len(eids))
        if order:
            return [eids[i - 1] for i in order], False
    return list(eids), True


def rerank_question(q, els, idx, cands, K, reverse):
    eids = list(cands[:K])
    if reverse:
        eids = eids[::-1]
    fails = 0
    if K <= 50:
        out, f = rerank_window(q, els, idx, eids)
        fails += f
    else:  # K=100: RankGPT sliding — 하단부터 window 20 / stride 10, 창 순서 고정
        out = list(eids)
        start = len(out) - 20
        while True:
            w = out[start:start + 20]
            ranked, f = rerank_window(q, els, idx, w)
            fails += f
            out[start:start + 20] = ranked
            if start == 0:
                break
            start = max(0, start - 10)
    return out, fails


def main():
    mode = sys.argv[1]
    els, idx, mapped = load_world()
    devq = [json.loads(l) for l in open(OUT / "candidates_K100.jsonl", encoding="utf-8")]
    if mode == "dry":
        tot = 0
        for run, cfg in RUNS.items():
            chars = 0
            for row in devq:
                q = mapped[row["qid"]]["q"]
                K = cfg["K"]
                if K <= 50:
                    chars += len(PROMPT.format(q=q, cands=cand_block(els, idx, row["candidates"][:K]), n=K))
                else:
                    for _ in range(9):
                        chars += len(PROMPT.format(q=q, cands=cand_block(els, idx, row["candidates"][:20]), n=20))
            print(f"[dry] {run}: {chars:,}자 → 추정 {int(chars*0.45):,}~{chars:,} 입력토큰")
            tot += chars
        print(f"[dry] 총 {tot:,}자 → 추정 {int(tot*0.45):,}~{tot:,} 토큰 (상한 {COST_CAP_INPUT_TOKENS:,})")
        return

    cfg = RUNS[mode]
    results, fail_total = {}, 0

    def work(row):
        qid = row["qid"]
        cands = row["candidates"]
        out, fails = rerank_question(mapped[qid]["q"], els, idx, cands, cfg["K"], cfg["reverse"])
        return qid, {"input": cands[:cfg["K"]][::-1] if cfg["reverse"] else cands[:cfg["K"]],
                     "output": out, "parse_fails": fails}

    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, (qid, r) in enumerate(ex.map(work, devq), 1):
            results[qid] = r
            fail_total += r["parse_fails"]
            if i % 16 == 0:
                print(f"{mode} {i}/{len(devq)} (usage in={_usage['input_tokens']:,})", flush=True)

    payload = {"run": mode, "config": cfg, "model": MODEL, "code_commit": code_commit(),
               "usage": dict(_usage), "parse_fail_windows": fail_total,
               "results": results}
    f = OUT / f"run_{mode}.json"
    f.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"{mode} DONE usage={_usage} parse_fails={fail_total} "
          f"sha={hashlib.sha256(f.read_bytes()).hexdigest()[:16]}", flush=True)


if __name__ == "__main__":
    main()
