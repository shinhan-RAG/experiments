# -*- coding: utf-8 -*-
"""
질문 태그(qtags) 생성 — codex CLI / gpt-5.6-luna 판.

filesearch/qtag_llm.py 의 프롬프트·출력 스키마를 그대로 재사용하고 호출부만 codex 로 교체.
입력은 질문 텍스트와 특약명 사전뿐 — 문서·gold·정답은 주지 않는다(순환 방지).

출력: out/qtags_luna.jsonl
  {qid, q_sha, contract[], role[], subject[], qualifier[], schema[], model, parse_ok}

사용:
  python gen_qtags_codex.py --gold out/gold_v7_train_strict.jsonl --workers 6
"""
import argparse, json, re, subprocess, sys, hashlib, io, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parents[2] / "tos-skeleton" / "hybrid-enrich_v2"
FS = R / "filesearch"
sys.path.insert(0, str(FS))
from qtag_llm import PROMPT, ROLES, ROLE_DESC          # noqa: E402  프롬프트 동일성 보장

OUT = HERE / "out"; OUT.mkdir(exist_ok=True)


def _codex_bin():
    """Windows 의 npm shim 은 bash 스크립트라 subprocess 가 못 찾는다 — .cmd 를 쓴다."""
    import shutil, os
    for c in ("codex.cmd", "codex"):
        p = shutil.which(c)
        if p and (os.name != "nt" or p.lower().endswith(".cmd")):
            return p
    p = Path(os.environ.get("APPDATA", "")) / "npm" / "codex.cmd"
    if p.exists():
        return str(p)
    return "codex"


CODEX = _codex_bin()


def call_codex(model, prompt, cwd, timeout=300):
    # 프롬프트가 길어(특약 141개) 커맨드라인 인자 한도를 넘으므로 stdin 으로 넘긴다.
    r = subprocess.run(
        [CODEX, "exec", "-m", model, "--skip-git-repo-check", "-"],
        input=prompt, capture_output=True, text=True, timeout=timeout, cwd=str(cwd),
        encoding="utf-8", errors="replace",
    )
    txt = r.stdout or ""
    blocks = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", txt, re.S)
    for b in reversed(blocks):                          # 마지막 JSON 블록 채택
        try:
            d = json.loads(b)
            if isinstance(d, dict) and ("contract" in d or "role" in d or "subject" in d):
                return d, txt[-400:]
        except Exception:
            continue
    return {}, txt[-400:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(OUT / "gold_v7_train_strict.jsonl"))
    ap.add_argument("--tags", default=str(FS / "out" / "tags_u2_rules.jsonl"))
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--out", default=str(OUT / "qtags_luna.jsonl"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n", type=int, default=-1)
    a = ap.parse_args()

    contracts = sorted({json.loads(l)["contract_key"]
                        for l in open(a.tags, encoding="utf-8")} - {""})
    roles = "\n".join("- %s: %s" % (r, ROLE_DESC[r]) for r in ROLES)
    out_path = Path(a.out)

    done = {}
    if out_path.exists():
        for l in out_path.open(encoding="utf-8"):
            try:
                d = json.loads(l); done[d["qid"]] = d
            except Exception:
                pass

    Q = [json.loads(l) for l in open(a.gold, encoding="utf-8")]
    if a.n > 0:
        Q = Q[:a.n]
    todo = [q for q in Q if q["qid"] not in done]
    print("model=%s contracts=%d total=%d cached=%d todo=%d"
          % (a.model, len(contracts), len(Q), len(Q) - len(todo), len(todo)), flush=True)

    contracts_block = "\n".join(contracts)
    t0 = time.time()

    def work(q):
        p = PROMPT.format(roles=roles, contracts=contracts_block, question=q["q"])
        try:
            js, raw = call_codex(a.model, p, HERE)
        except Exception as e:
            js, raw = {}, "ERROR %s" % e
        return {"qid": q["qid"],
                "q_sha": hashlib.sha256(q["q"].encode()).hexdigest()[:12],
                "contract": [c for c in (js.get("contract") or []) if c in contracts],
                "role": [r for r in (js.get("role") or []) if r in ROLES],
                "subject": [str(s)[:60] for s in (js.get("subject") or [])][:6],
                "qualifier": [str(s)[:60] for s in (js.get("qualifier") or [])][:6],
                "schema": [s for s in (js.get("schema") or []) if s in ("table", "formula")],
                "model": a.model, "parse_ok": bool(js)}

    ok = 0
    with ThreadPoolExecutor(a.workers) as ex, out_path.open("a", encoding="utf-8") as f:
        for i, rec in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            ok += bool(rec["parse_ok"])
            if i % 10 == 0 or i == len(todo):
                print("  %d/%d  parse_ok=%d  %.0fs  last=%s %s"
                      % (i, len(todo), ok, time.time() - t0, rec["qid"],
                         (rec["contract"][:1] or [""])[0][:28]), flush=True)
    print("done parse_ok=%d/%d" % (ok, len(todo)), flush=True)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
