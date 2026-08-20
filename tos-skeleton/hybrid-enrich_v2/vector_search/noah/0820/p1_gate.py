#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1(fanout) 오프라인 검증 — LLM 0회.

  gate 모드:  플래그 없는 arm 으로 search/submit 을 실행해 출력을 캡처(--capture DIR)하거나
              기준 캡처와 바이트 비교(--ref DIR). 수정 전 코드에서 --capture, 수정 후 --ref.
  eval 모드:  gold_v4_train60 60문항 각각을 fanout off/on 두 번 검색(서브프로세스, 실제 CLI 경로)
              → page1 40건을 조(jo)로 해석 → scoring.score 로 R@5/R@10/R@40 비교표 출력.

사용: PYTHONUTF8=1 python p1_gate.py gate --capture out/p1_gate_ref
      PYTHONUTF8=1 python p1_gate.py gate --ref out/p1_gate_ref
      PYTHONUTF8=1 python p1_gate.py eval
"""
import argparse, json, os, subprocess, sys, tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
sys.path.insert(0, str(FS))

ARM_BASE = {"tags": "tags_u2_rules.jsonl", "mode": "clm", "lex": "count", "w": {"contract": 2},
            "router": "llm", "qtags": "out/qtags_haiku.jsonl", "facet": 1}
# 게이트 submit 경로용: s_all 의 submit 플래그(meta 는 ChunkHybridSearch 기동을 피해 제외)
ARM_GATE_SUBMIT = dict(ARM_BASE, submit_pad=1, submit_ref_merge=1, alias_cond=5.0)

ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def load_gold():
    return [json.loads(l) for l in open(HERE / "out/gold_v4_train60.jsonl", encoding="utf-8")]


def tool(args, qid, arm, sess):
    env = dict(ENV, SEMTAG_SESSION=str(sess), SEMTAG_QID=qid, SEMTAG_ARM=json.dumps(arm, ensure_ascii=False))
    r = subprocess.run([sys.executable, str(HERE / "agent_tools.py")] + args,
                       capture_output=True, env=env, cwd=str(HERE))
    if r.returncode != 0:
        raise RuntimeError(f"agent_tools 실패 qid={qid} args={args}\n{r.stderr.decode('utf-8', 'replace')[-800:]}")
    return r.stdout


def run_gate(a):
    gold = load_gold()[:5]
    outdir = Path(a.capture or a.ref)
    cap = bool(a.capture)
    if cap:
        outdir.mkdir(parents=True, exist_ok=True)
    n_diff = 0
    with tempfile.TemporaryDirectory() as td:
        for i, g in enumerate(gold):
            sess = Path(td) / f"s{i}"
            outs = {f"search_{i}.json": tool(["search", "--q", g["q"]], g["qid"], ARM_BASE, sess)}
            ids = [it["id"] for it in json.loads(outs[f"search_{i}.json"].decode("utf-8")).get("results", [])[:4]]
            if ids:
                outs[f"submit_{i}.json"] = tool(["submit", "--ids", ",".join(ids)], g["qid"], ARM_GATE_SUBMIT, sess)
            for name, b in outs.items():
                p = outdir / name
                if cap:
                    p.write_bytes(b)
                elif p.read_bytes() != b:
                    n_diff += 1
                    print(f"[게이트 불일치] {name}")
    if cap:
        print(f"기준 캡처 저장: {outdir}")
    else:
        print("게이트: " + ("통과 — 바이트 단위 동일" if n_diff == 0 else f"실패 — {n_diff}건 불일치"))
        if n_diff:
            sys.exit(1)


def search_jos(g, arm, U):
    """search 1회 → page1 40건 id → 조 단위 순위(units.Units.resolve, 중복 조 제거)."""
    with tempfile.TemporaryDirectory() as td:
        out = json.loads(tool(["search", "--q", g["q"]], g["qid"], arm, Path(td) / "s").decode("utf-8"))
    ids = [it["id"] for it in out.get("results", [])]
    return U.resolve(ids), ids


def run_eval(a):
    from units import Units
    from scoring import score
    gold = load_gold()
    U = Units()
    arm_off, arm_on = ARM_BASE, dict(ARM_BASE, fanout=1)

    def one(g):
        r_off, ids_off = search_jos(g, arm_off, U)
        r_on, ids_on = search_jos(g, arm_on, U)
        return (g, score(r_off, g["groups"], ks=(5, 10, 40)), score(r_on, g["groups"], ks=(5, 10, 40)),
                ids_off != ids_on)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(one, gold))

    def agg(idx, sub=None):
        sel = [r for r in rows if (sub is None or r[3] == sub)]
        n = len(sel)
        return n, {k: sum(r[idx][k] for r in sel) / n for k in ("R@5", "R@10", "R@40")} if n else {}

    n_changed = sum(1 for r in rows if r[3])
    print(f"\n== P1 fanout 오프라인 비교 (train60, page1 40건 → 조 순위, scoring.score) ==")
    print(f"페이지가 실제로 바뀐 문항: {n_changed}/60")
    for label, sub in (("전체 60문항", None), ("fanout 발화(페이지 변경) 문항", True)):
        n_off, m_off = agg(1, sub)
        _, m_on = agg(2, sub)
        if not n_off:
            continue
        print(f"\n[{label}] n={n_off}")
        print(f"{'지표':<8}{'off':>10}{'on':>10}{'Δ':>10}")
        for k in ("R@5", "R@10", "R@40"):
            print(f"{k:<8}{m_off[k]:>10.4f}{m_on[k]:>10.4f}{m_on[k] - m_off[k]:>+10.4f}")
    # 문항별 변화 상세(악화 문항 확인용)
    worse = [(r[0]["qid"], r[1]["R@5"], r[2]["R@5"]) for r in rows if r[2]["R@5"] < r[1]["R@5"]]
    better = [(r[0]["qid"], r[1]["R@5"], r[2]["R@5"]) for r in rows if r[2]["R@5"] > r[1]["R@5"]]
    print(f"\nR@5 개선 {len(better)}문항, 악화 {len(worse)}문항")
    for qid, o, n_ in worse:
        print(f"  악화: {qid} {o:.2f}→{n_:.2f}")
    for qid, o, n_ in better:
        print(f"  개선: {qid} {o:.2f}→{n_:.2f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    mx = g.add_mutually_exclusive_group(required=True)
    mx.add_argument("--capture", help="기준 출력 캡처 저장 디렉터리(수정 전 코드에서 실행)")
    mx.add_argument("--ref", help="기준 캡처와 바이트 비교")
    e = sub.add_parser("eval")
    e.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    (run_gate if a.cmd == "gate" else run_eval)(a)


if __name__ == "__main__":
    main()
