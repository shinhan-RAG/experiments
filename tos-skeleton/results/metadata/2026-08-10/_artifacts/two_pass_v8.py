# -*- coding: utf-8 -*-
"""v8 재현 검증 — two_pass_check.py 와 **같은 해시 방식**으로, v8 뷰 경로에 맞춰 돌린다.

two_pass_check.py 를 쓸 수 없는 이유: 그 스크립트는 build_views.py 로 뷰를 재생성하는데
v8 뷰는 build_views_v8.py 가 만든다. 해시 함수(content_sha/combined)는 그대로 가져와
결과를 08-07/08-09 등록본과 같은 잣대로 비교할 수 있게 한다.

보증 범위: 메타데이터(meta_v8.jsonl)를 고정한 이후 구간 = 뷰조립 -> 검색 -> 채점.
LLM 생성 자체는 비결정적이라 보증 대상이 아니다 (08-09 등록본과 같은 방침).
"""
import subprocess, sys, os, json, hashlib
from pathlib import Path

MS = Path(r"c:/Users/Atdev-pc/Desktop/wiki/experiments/tos-skeleton/hybrid-enrich/metajson-v6/meta-search-v4")
OUT = MS / "out"
HERE = Path(__file__).parent
CS = "fixed600"
ARMS = ["V8_CONTRAST", "V8_PLUS"]
EVAL = "eval_v8_2pass.json"
VOLATILE = {"latency_ms", "took_ms", "elapsed", "timestamp"}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def content_sha(p: Path) -> str:
    if not (p.name.startswith("run_") and p.suffix == ".jsonl"): return sha(p)
    h = hashlib.sha256()
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            row = {k: v for k, v in json.loads(line).items() if k not in VOLATILE}
            h.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def combined(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode("utf-8")); h.update(content_sha(p).encode("ascii"))
    return h.hexdigest()


def run(cmd, cwd):
    print("  $", " ".join(cmd[:4]), "...", flush=True)
    r = subprocess.run([sys.executable, *cmd], cwd=str(cwd), capture_output=True)
    if r.returncode:
        print(r.stdout.decode("utf-8", "replace")[-1200:]); raise SystemExit(f"실패: {cmd}")


def one_pass():
    run(["build_views_v8.py"], HERE)
    for a in ARMS:
        run(["retrieve.py", "--chunkset", CS, "--arm", a, "--backend", "dense", "--topk", "100"], MS)
    run(["eval.py", "--chunkset", CS, "--out", EVAL,
         "--runs", *[f"run_{CS}_{a}_dense.jsonl" for a in ARMS]], MS)


targets = [OUT / f"run_{CS}_{a}_dense.jsonl" for a in ARMS] + [OUT / EVAL]

print("== 1회차 ==")
one_pass()
p1 = combined(targets)
print(f"1회차 결합 sha256 = {p1}")

print("== 2회차 (산출물 삭제 후 동일 명령) ==")
for p in targets: p.unlink()
one_pass()
p2 = combined(targets)
print(f"2회차 결합 sha256 = {p2}")

ok = p1 == p2
print(f"\n결정적: {ok}   two_pass_sha = {p1[:8] if ok else 'N/A'}")
(OUT / "two_pass_v8.json").write_text(json.dumps({
    "chunkset": CS, "arms": ARMS, "pass1": p1, "pass2": p2,
    "deterministic": ok, "two_pass_sha": p1[:8] if ok else None,
    "scope": "메타데이터 고정 이후 구간(뷰조립->검색->채점). LLM 생성은 비결정적이라 보증 대상 아님.",
}, ensure_ascii=False, indent=2), encoding="utf-8")
