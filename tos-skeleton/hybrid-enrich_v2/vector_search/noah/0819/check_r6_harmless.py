#!/usr/bin/env python3
"""R6 무해성 게이트 — intent_role 이 발화하지 않은 질문의 첫 페이지가 기준 arm 과
byte 단위로 같은지 확인(check_r1_harmless.py 와 동일 방식).

기준 arm(r1v)과 신규 arm(r6)은 intent_role 키 하나만 다르므로, 두 arm 을 같은
조건(LLM 0회)으로 각각 1회 검색하고 첫 페이지 40칸의 (id, jo) 순서를 JSON
직렬화해 byte 비교한다. intent_role_subquery 가 발화하지 않은 질문에서 차이가
하나라도 있으면 FAIL 이다.
"""
import argparse, json, os, random, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"


def page_signature(resp):
    """첫 페이지 노출 ID·순서만 남긴 정규 직렬화(점수·미리보기 제외)."""
    return json.dumps([[item.get("id", ""), item.get("jo", "")]
                       for item in (resp.get("results") or [])],
                      ensure_ascii=False, sort_keys=True).encode("utf-8")


def run(arm, arm_name, qid, question, tag):
    sess = HERE / "out" / "harmless_check" / f"{qid}_{tag}"
    sess.mkdir(parents=True, exist_ok=True)
    if (sess / "calls.jsonl").exists():
        (sess / "calls.jsonl").unlink()
    env = dict(os.environ, SEMTAG_SESSION=str(sess), SEMTAG_QID=qid,
               SEMTAG_ARM=json.dumps(arm, ensure_ascii=False))
    proc = subprocess.run([sys.executable, str(HERE / "agent_tools.py"), "search", "--q", question],
                          capture_output=True, text=True, env=env, cwd=str(HERE), timeout=300)
    return json.loads(proc.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="r1v_verify_identity_reference_hybrid")
    ap.add_argument("--arm", default="r6_intent_role_hybrid")
    ap.add_argument("--gold", default=str(FS / "out/gold_train_scoped_u3_reviewed_overlay_v20.jsonl"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--exclude", default=str(HERE / "out/exposure_check_qids_R6.json"))
    a = ap.parse_args()

    arms = json.loads((HERE / "arms.json").read_text())
    pool = [json.loads(l) for l in open(a.gold, encoding="utf-8")]
    excluded = set(json.load(open(a.exclude))) if a.exclude and Path(a.exclude).exists() else set()
    pool = [g for g in pool if g["qid"] not in excluded]
    pool.sort(key=lambda g: g["qid"])
    sample = random.Random(a.seed).sample(pool, min(a.n, len(pool)))

    rows, fired, mismatch_silent, mismatch_fired = [], 0, 0, 0
    for g in sample:
        base = run(arms[a.base], a.base, g["qid"], g["q"], "base")
        new = run(arms[a.arm], a.arm, g["qid"], g["q"], "r6")
        intent = new.get("intent_role_subquery") or {}
        did_fire = bool(intent.get("fired"))
        same = page_signature(base) == page_signature(new)
        fired += did_fire
        if not same:
            if did_fire:
                mismatch_fired += 1
            else:
                mismatch_silent += 1
        rows.append({"qid": g["qid"], "fired": did_fire, "identical": same,
                     "intent_roles": intent.get("roles") or [],
                     "intent_vocabulary": [sub.get("vocabulary")
                                           for sub in (intent.get("subqueries") or [])],
                     "base_page": len(base.get("results") or []),
                     "new_page": len(new.get("results") or [])})
        print(g["qid"], "FIRED" if did_fire else "silent",
              "identical" if same else "DIFFERENT", flush=True)

    summary = {"n": len(rows), "fired": fired, "silent": len(rows) - fired,
               "silent_byte_identical": len(rows) - fired - mismatch_silent,
               "silent_mismatch": mismatch_silent, "fired_changed": mismatch_fired,
               "pass": mismatch_silent == 0, "base_arm": a.base, "arm": a.arm, "seed": a.seed}
    print(json.dumps(summary, ensure_ascii=False))
    out = HERE / "out" / "harmless_check" / "summary_r6.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return 0 if mismatch_silent == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
