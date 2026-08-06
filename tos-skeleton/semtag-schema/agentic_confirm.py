#!/usr/bin/env python3
"""에이전틱 확정 측정 — 태그 스키마 s0(현행형) vs s1k2(주소성 강화) paired.

PR #18 러너 결함 수정: error 세션은 재실행 시 재시도(캐시 안 함), 집계 파일은
append+최신 우선(클로버 없음). grep+read 전용, 세션당 도구 캡 10, 동일 프롬프트.

usage: python3 agentic_confirm.py --doc <md> --gold out/gold_mapped.jsonl \
         --dev-qids <qa100_gold.jsonl> --out out/agentic --arms s0,s1k2 \
         [--model haiku] [--workers 8] [--limit N] [--smoke]
"""
import argparse
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import semtag_experiment as SE
from semtag_h3 import build_tag_v2
from context_v2 import annotate_v2

PROMPT = """당신은 보험 약관 문서에서 질문의 근거 element를 찾는 검색 에이전트다.

도구는 Bash로 다음 두 명령만 사용한다 (다른 명령 금지):
  python3 tools.py grep '<패턴>'   # 뷰 파일 전체에서 rg -i 검색. 각 줄 = eid | [태그] | 원문
  python3 tools.py read <eid>      # 해당 element 원문 전체 열람

규칙: 도구 호출은 최대 10회. 뷰 줄 앞부분의 [특약]/[경로]/[역할]/[키] 마커를 패턴에
활용할 수 있다. 충분히 확인되면 즉시 종료하라.

질문: {q}

마지막 출력은 반드시 아래 한 줄 JSON만:
{{"ranked_eids": ["e00123", ...], "reason": "한 문장"}}
(ranked_eids는 근거 가능성 순 최대 10개)"""

TOOLS_PY = r'''
import json, os, subprocess, sys
_here = os.path.dirname(os.path.abspath(__file__))
_cfg = json.load(open(os.path.join(_here, "cfg.json")))
VIEW = _cfg["view"]; SDIR = _here
CAP = 10
calls = os.path.join(SDIR, "calls.txt")
n = sum(1 for _ in open(calls)) if os.path.exists(calls) else 0
if n >= CAP:
    print(json.dumps({"error": "call cap reached — 지금까지 정보로 답하라"})); sys.exit(0)
open(calls, "a").write("\t".join(sys.argv[1:3]) + "\n")
cmd = sys.argv[1]
if cmd == "grep":
    pat = sys.argv[2]
    r = subprocess.run(["rg", "-i", "-e", pat, "--", VIEW],
                       capture_output=True, text=True, timeout=30)
    lines = r.stdout.splitlines()
    if len(lines) > 20:
        step = len(lines) / 20
        lines = [lines[int(i * step)] for i in range(20)]
        note = f"(전체 {len(r.stdout.splitlines())}건 중 등간격 20건)"
    else:
        note = f"({len(lines)}건)"
    print(note)
    for l in lines:
        print(l[:300])
elif cmd == "read":
    eid = sys.argv[2]
    for l in open(VIEW):
        if l.startswith(eid + " | "):
            print(l[:4000]); break
    else:
        print(json.dumps({"error": "eid not found"}))
'''


def build_view(els, schema, path):
    with open(path, "w", encoding="utf-8") as f:
        for e in els:
            tag = build_tag_v2(e, "s2q") if schema == "s2q" else SE.build_tag(e, schema)
            flat = re.sub(r"\s+", " ", e["text"])[:3000]
            f.write(f"{e['eid']} | {tag} | {flat}\n")


def run_one(item, arm, cfg):
    qid = item["qid"]
    sdir = Path(cfg["out"]) / "sessions" / f"{qid}_{arm}"
    done_f = sdir / "result.json"
    if done_f.exists():
        prev = json.loads(done_f.read_text())
        if prev.get("status") == "ranked":          # error는 재시도
            return prev
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "calls.txt").unlink(missing_ok=True)
    (sdir / "tools.py").write_text(TOOLS_PY, encoding="utf-8")
    (sdir / "cfg.json").write_text(json.dumps(
        {"view": str(cfg["views"][arm])}), encoding="utf-8")
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)   # 구독 인증 폴백(크레딧 계정 회피)
    t0 = time.time()
    try:
        r = subprocess.run(
            ["claude", "-p", "--model", cfg["model"], "--output-format", "json",
             "--max-turns", "24",
             "--allowedTools", "Bash(python3 tools.py:*)",
             "--disallowedTools", "Read,Write,Edit,Grep,Glob,WebFetch,WebSearch,Task"],
            input=PROMPT.format(q=item["q"]), capture_output=True, text=True,
            timeout=300, cwd=str(sdir), env=env)
        out = json.loads(r.stdout)
        text = str(out.get("result", ""))
        m = re.findall(r"\{[^{}]*\"ranked_eids\"[\s\S]*?\}", text)
        rec = json.loads(m[-1]) if m else None
        if not isinstance(rec, dict) or "ranked_eids" not in rec:
            rec = {"status": "error", "error": "no_json", "raw": text[-200:]}
        else:
            rec = {"status": "ranked",
                   "ranked_eids": [str(x) for x in rec["ranked_eids"]][:10],
                   "reason": str(rec.get("reason", ""))[:200]}
        rec.update({"turns": out.get("num_turns"),
                    "cost_usd": out.get("total_cost_usd")})
    except Exception as e:
        rec = {"status": "error", "error": str(e)[:150]}
    rec.update({"qid": qid, "arm": arm, "model": cfg["model"],
                "duration_ms": int((time.time() - t0) * 1000)})
    done_f.write_text(json.dumps(rec, ensure_ascii=False))
    return rec


def evaluate(rows, gold_by_qid, arms):
    latest = {}
    for r in rows:
        latest[(r["qid"], r["arm"])] = r
    per = {a: {} for a in arms}
    for (qid, arm), r in latest.items():
        g = set(gold_by_qid[qid]["gold"])
        ranked = r.get("ranked_eids") or []
        rk = next((i + 1 for i, x in enumerate(ranked[:10]) if x in g), None)
        per[arm][qid] = {"r5": int(bool(rk and rk <= 5)), "r10": int(bool(rk)),
                        "mrr": (1 / rk) if rk else 0.0,
                        "err": int(r["status"] != "ranked")}
    out = {}
    common = set.intersection(*(set(per[a]) for a in arms))
    for a in arms:
        rs = [per[a][q] for q in sorted(common)]
        out[a] = {"n": len(rs),
                  "r5": round(sum(r["r5"] for r in rs) / len(rs), 4),
                  "r10": round(sum(r["r10"] for r in rs) / len(rs), 4),
                  "mrr10": round(sum(r["mrr"] for r in rs) / len(rs), 4),
                  "errors": sum(r["err"] for r in rs)}
    a, b = arms
    deltas = [per[b][q]["mrr"] - per[a][q]["mrr"] for q in sorted(common)]
    win = sum(1 for q in common if per[b][q]["r5"] > per[a][q]["r5"])
    loss = sum(1 for q in common if per[b][q]["r5"] < per[a][q]["r5"])
    out["paired"] = {"delta_mrr": round(sum(deltas) / len(deltas), 4),
                     "ci95": [round(x, 4) for x in SE.boot_ci(deltas)],
                     "r5_transitions": {"improve": win, "regress": loss}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--dev-qids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="s0,s1k2")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--use-test", action="store_true",
                    help="dev 제외 = 동결 test 문항으로 실행")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    arm_schema = {"s0": "tag.s0", "s1k2": "tag.s1k2", "s2q": "s2q"}
    arms = args.arms.split(",")

    lines = SE.nfc(Path(args.doc).read_text(encoding="utf-8",
                                            errors="ignore")).splitlines()
    els = annotate_v2(SE.split_elements(lines), lines)
    views = {}
    for a in arms:
        p = out / f"view_{a}.txt"
        if not p.exists():
            build_view(els, arm_schema[a], p)
        views[a] = p.resolve()
    tooldir = out / "tooldir"
    tooldir.mkdir(exist_ok=True)
    (tooldir / "tools.py").write_text(TOOLS_PY, encoding="utf-8")

    gold = [json.loads(l) for l in open(args.gold, encoding="utf-8")]
    dev_qids = {json.loads(l)["qid"] for l in open(args.dev_qids)}
    items = [g for g in gold if (g["qid"] not in dev_qids) == args.use_test]
    if args.limit:
        items = items[:args.limit]
    gold_by_qid = {g["qid"]: g for g in items}
    cfg = {"out": out, "views": views, "tooldir": str(tooldir),
           "model": args.model}

    jobs = [(it, a) for it in items for a in arms]
    print(f"jobs={len(jobs)} (q={len(items)} × arms={arms}) model={args.model}",
          flush=True)
    rows = []
    agg = out / "agentic_rows.jsonl"
    with open(agg, "a", encoding="utf-8") as f, \
         ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, it, a, cfg): (it["qid"], a)
                for it, a in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            rec = fu.result()
            rows.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(jobs):
                ok = sum(1 for r in rows if r["status"] == "ranked")
                print(f"  {i}/{len(jobs)} ranked={ok}", flush=True)

    all_rows = [json.loads(l) for l in open(agg, encoding="utf-8")]
    res = evaluate([r for r in all_rows
                    if (r["qid"], r["arm"]) in {(it["qid"], a) for it, a in jobs}],
                   gold_by_qid, arms)
    res["total_cost_usd"] = round(sum(r.get("cost_usd") or 0 for r in rows), 2)
    (out / "agentic_eval.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
