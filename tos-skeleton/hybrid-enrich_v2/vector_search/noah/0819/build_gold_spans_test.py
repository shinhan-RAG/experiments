#!/usr/bin/env python3
"""test 149 gold(element id) → char span 채점 파일 생성 — map_lsh_gold.py 와 동일 방식(strict).

필요 입력: cowork "원문 span 매핑 파일.jsonl" (lsh element_id → 원문 발췌 텍스트).
  * hybrid-enrich/out/elements.jsonl 로는 만들 수 없다 — element_id 가 같아도 다른 우주다
    (검증: train 348 재구성 시 기존 채점파일과 span 겹침 0.5%).
채점 규칙(map_lsh_gold strict 와 동일): gold element 텍스트를 map_group 으로 앵커링 →
  1) 인접 span 병합(gap<=30)  2) 같은 텍스트(앞 160자) group 은 OR(a/a' 동치)  3) 남은 group = AND.

Usage:
    python build_gold_spans_test.py --spanmap "<원문 span 매핑 파일.jsonl>"
    python build_gold_spans_test.py --spanmap ... --validate   # train 348 재구성 → 기존 파일과 대조
"""
import argparse, json, re, sys, unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
V2OUT = HERE.parents[2] / "out"
DOC = HERE.parents[1] / "doc" / "판매약관_(간편)신한통합건강보장보험 원(ONE)(무배당, 해약환급금 미지급형)_250212.md"
sys.path.insert(0, str(FS))
from map_gold_spans import norm_map, map_group  # noqa: E402

GAP, KEY = 30, 160
DROP_LINE = re.compile(r"-{3,}|\d{1,4}|SHINHAN LIFE|_{3,}|\d{10,}")


def build(gold_path: Path, S: dict, nd, idx, raw) -> list:
    span_cache = {}

    def span_of(eid):
        if eid not in span_cache:
            t = S[eid]["text"]
            lines = [x.strip() for x in t.split("\n") if x.strip() and not DROP_LINE.fullmatch(x.strip())]
            g, _un = map_group(nd, idx, lines)
            span_cache[eid] = g
        return span_cache[eid]

    def key_of(c0, c1):
        return re.sub(r"\s", "", raw[c0:c1])[:KEY]

    rows = []
    for g in (json.loads(l) for l in open(gold_path, encoding="utf-8")):
        eids = g.get("gold") or []
        spans, fail = [], []
        for eid in eids:
            sp = span_of(eid) if eid in S else None
            if sp:
                spans.append((sp["c0"], sp["c1"], eid))
            else:
                fail.append(eid)
        spans.sort()
        merged = []
        for c0, c1, eid in spans:
            if merged and c0 - merged[-1]["c1"] <= GAP:
                merged[-1]["c1"] = max(merged[-1]["c1"], c1); merged[-1]["lsh_eids"].append(eid)
            else:
                merged.append({"c0": c0, "c1": c1, "lsh_eids": [eid]})
        groups = []
        for m in merged:
            k = key_of(m["c0"], m["c1"])
            for gr in groups:
                if gr["key"] == k:
                    gr["members"].append({"c0": m["c0"], "c1": m["c1"], "src": "gold"}); gr["lsh_eids"] += m["lsh_eids"]
                    break
            else:
                groups.append({"key": k, "members": [{"c0": m["c0"], "c1": m["c1"], "src": "gold"}], "lsh_eids": list(m["lsh_eids"])})
        status = "ok" if eids and not fail else ("empty_gold" if not eids else "partial")
        rows.append({"qid": g["qid"], "q": g["q"], "task_type": g.get("task_type", ""),
                     "core_retrieval": g.get("core_retrieval", "False"), "status": status,
                     **({"unmapped": fail} if fail else {}),
                     "groups": groups, "lsh_gold": eids,
                     "rule": f"gap<={GAP};same_text_or;no_expand"})
    return rows


def validate(rows: list, ref_path: Path) -> None:
    ref = {json.loads(l)["qid"]: json.loads(l) for l in open(ref_path, encoding="utf-8")}
    n_q = n_grp_eq = n_span_hit = n_span_tot = 0
    for r in rows:
        v = ref.get(r["qid"])
        if not v or not v.get("groups") or v.get("status", "ok") != "ok" or r["status"] != "ok":
            continue
        n_q += 1
        if len(r["groups"]) == len(v["groups"]):
            n_grp_eq += 1
        vm = [(m["c0"], m["c1"]) for gr in v["groups"] for m in (gr.get("members") or [gr])]
        for gr in r["groups"]:
            for m in gr["members"]:
                n_span_tot += 1
                if any(m["c0"] < c1 and m["c1"] > c0 for c0, c1 in vm):
                    n_span_hit += 1
    print(f"[검증] 공통 채점문항 {n_q} / group 수 일치 {n_grp_eq} ({100*n_grp_eq/max(n_q,1):.1f}%) / "
          f"span 겹침 {n_span_hit}/{n_span_tot} ({100*n_span_hit/max(n_span_tot,1):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spanmap", required=True, help='cowork "원문 span 매핑 파일.jsonl" 경로')
    ap.add_argument("--validate", action="store_true", help="train 348 재구성 → 기존 채점파일과 대조만")
    a = ap.parse_args()
    raw = unicodedata.normalize("NFC", open(DOC, encoding="utf-8").read().replace("\r\n", "\n"))
    nd, idx = norm_map(raw)
    S = {json.loads(l)["element_id"]: json.loads(l) for l in open(a.spanmap, encoding="utf-8")}
    print(f"spanmap elements: {len(S):,}")
    if a.validate:
        rows = build(V2OUT / "gold_mapped_noah_v3_348_train.jsonl", S, nd, idx, raw)
        validate(rows, FS / "out" / "gold_spans_lsh_train.jsonl")
        return
    rows = build(V2OUT / "gold_mapped_noah_v3_149_test.jsonl", S, nd, idx, raw)
    ok = sum(1 for r in rows if r["status"] == "ok")
    outp = HERE / "out" / "gold_spans_lsh_test.jsonl"
    outp.parent.mkdir(exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"test 149 → ok {ok} / partial {sum(1 for r in rows if r['status']=='partial')} / "
          f"empty {sum(1 for r in rows if r['status']=='empty_gold')} -> {outp}")


if __name__ == "__main__":
    main()
