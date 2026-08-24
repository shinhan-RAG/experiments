#!/usr/bin/env python3
"""결정론 노출 게이트 — 주어진 qid 들의 첫 하이브리드 검색(호스트와 동일 조건)에서 gold JO 가
top-K 에 노출되는지 잰다. LLM 0회. 개선 전/후 paired 비교용."""
import argparse, json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", required=True, help="쉼표 구분 qid 또는 json 파일")
    ap.add_argument("--arm", default="c29_membership_dual_evidence_hybrid")
    ap.add_argument("--gold", default=str(FS / "out/gold_train_scoped_u3_reviewed_overlay_v19.jsonl"))
    ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--tag", default="", help="세션 디렉터리 접미(대조 기록용)")
    a = ap.parse_args()
    qids = json.load(open(a.qids)) if a.qids.endswith(".json") else [x.strip() for x in a.qids.split(",")]
    arms = json.loads((HERE / "arms.json").read_text())
    arm = arms[a.arm]
    G = {json.loads(l)["qid"]: json.loads(l) for l in open(a.gold)}
    J = [json.loads(l) for l in open(FS / "out/elements_u3jo.jsonl")]
    exposed = 0
    rows = []
    for qid in qids:
        g = G[qid]
        gj = {u["element_id"] for u in J
              if any(u["char_start"] < m["c1"] and u["char_end"] > m["c0"] for gr in g["groups"] for m in gr["members"])}
        sess = HERE / "out" / "exposure_check" / (qid + (("_" + a.tag) if a.tag else ""))
        sess.mkdir(parents=True, exist_ok=True)
        for f in ("calls.jsonl",):
            if (sess / f).exists():
                (sess / f).unlink()
        env = dict(os.environ, SEMTAG_SESSION=str(sess), SEMTAG_QID=qid,
                   SEMTAG_ARM=json.dumps(arm, ensure_ascii=False))
        p = subprocess.run([sys.executable, str(HERE / "agent_tools.py"), "search", "--q", g["q"]],
                           capture_output=True, text=True, env=env, cwd=str(HERE), timeout=300)
        try:
            resp = json.loads(p.stdout)
        except Exception:
            rows.append({"qid": qid, "error": p.stderr[-200:]}); continue
        order = []
        def walk(o):
            if isinstance(o, dict):
                jid = o.get("jo") or (o.get("id") if str(o.get("id", "")).startswith("j") else None)
                if jid and jid not in order:
                    order.append(jid)
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(resp.get("results") or resp)
        ranks = [i + 1 for i, jid in enumerate(order[: a.k]) if jid in gj]
        hit = bool(ranks)
        exposed += hit
        rows.append({"qid": qid, "exposed": hit, "ranks": ranks[:3], "fallback": resp.get("fallback_status", "")})
        print(qid, "exposed" if hit else "MISS", ranks[:3], flush=True)
    print(json.dumps({"n": len(qids), "exposed": exposed, "k": a.k, "arm": a.arm, "tag": a.tag}, ensure_ascii=False))
    out = HERE / "out" / "exposure_check" / f"summary{('_' + a.tag) if a.tag else ''}.json"
    json.dump({"rows": rows, "exposed": exposed, "n": len(qids)}, open(out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
