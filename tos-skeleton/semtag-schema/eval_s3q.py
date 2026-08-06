#!/usr/bin/env python3
"""s3q 평가 — s0 + [묻는말](규칙 브리지 + LLM 생성·필터·동결분), concat 채널.
dev/test paired vs A0·s0·s2q. 전 구간 결정적(동결 입력)."""
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
import semtag_experiment as SE
from semtag_h3 import build_tag_v2, bridge_terms
from context_v2 import annotate_v2

doc = sys.argv[1]; devq = sys.argv[2]
lines = SE.nfc(Path(doc).read_text(encoding="utf-8", errors="ignore")).splitlines()
els = annotate_v2(SE.split_elements(lines), lines)
frozen = json.loads(Path("out/qgen_frozen.json").read_text(encoding="utf-8"))
gen = frozen["tags"]
mapped = [json.loads(l) for l in open("out/gold_mapped.jsonl", encoding="utf-8")]
dev_qids = {json.loads(l)["qid"] for l in open(devq)}
dev = [m for m in mapped if m["qid"] in dev_qids]
test = [m for m in mapped if m["qid"] not in dev_qids]

def tag_s3q(e):
    f = [("유형", e["type"]), ("특약", e["scope"]), ("조항", e["jo"])]
    role = SE.roles_of(e["text"][:1500])
    f.append(("역할", role[0] if role else ""))
    ask = bridge_terms(e)
    key = e["scope"] + "\x1f" + e["jo"]
    g = gen.get(key) or []
    ask2 = ";".join(x for x in ([ask] if ask else []) + g if x)
    f.append(("묻는말", ask2))
    return SE.serialize(f)

def rows_for(tags, items):
    reps = [(t + " ||| " + e["text"]) if t else e["text"] for t, e in zip(tags, els)]
    bm = SE.BM25(reps)
    out = {}
    for it in items:
        top10 = [els[i]["eid"] for i in bm.rank10(it["q"])]
        g = set(it["gold"])
        rk = next((k+1 for k, x in enumerate(top10) if x in g), None)
        out[it["qid"]] = {"r5": int(bool(rk and rk <= 5)), "mrr": (1/rk) if rk else 0.0}
    return out

schemas = {
    "A0": ["" for _ in els],
    "s0": [SE.build_tag(e, "tag.s0") for e in els],
    "s2q": [build_tag_v2(e, "s2q") for e in els],
    "s3q": [tag_s3q(e) for e in els],
}
tl = sorted(len(t) for t in schemas["s3q"] if t)
print(f"s3q 태그 길이 p50={tl[len(tl)//2]} max={tl[-1]} | 생성 반영 그룹={sum(1 for v in gen.values() if v)}")
res = {}
for split_name, items in (("dev", dev), ("TEST", test)):
    R = {k: rows_for(v, items) for k, v in schemas.items()}
    line = f"[{split_name}] "
    for k in schemas:
        r5 = sum(v["r5"] for v in R[k].values())/len(R[k])
        mrr = sum(v["mrr"] for v in R[k].values())/len(R[k])
        line += f"{k} R@5={r5:.4f}/MRR={mrr:.4f}  "
    print(line)
    for ref in ("A0", "s0", "s2q"):
        dm = [R["s3q"][q]["mrr"] - R[ref][q]["mrr"] for q in R["s3q"]]
        dr = [R["s3q"][q]["r5"] - R[ref][q]["r5"] for q in R["s3q"]]
        ci, cir = SE.boot_ci(dm), SE.boot_ci(dr)
        imp = sum(1 for d in dr if d > 0); reg = sum(1 for d in dr if d < 0)
        print(f"  s3q vs {ref}: ΔR@5 {sum(dr)/len(dr):+.4f} [{cir[0]:+.4f},{cir[1]:+.4f}] "
              f"ΔMRR {sum(dm)/len(dm):+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] 전환 {imp}/{reg}")
    res[split_name] = {k: {"r5": round(sum(v['r5'] for v in R[k].values())/len(R[k]), 4),
                            "mrr": round(sum(v['mrr'] for v in R[k].values())/len(R[k]), 4)}
                       for k in schemas}
Path("out/s3q_results.json").write_text(json.dumps(
    {"frozen_sha_note": "see out/qgen_frozen.json", **res}, ensure_ascii=False, indent=1))
print("saved out/s3q_results.json")
