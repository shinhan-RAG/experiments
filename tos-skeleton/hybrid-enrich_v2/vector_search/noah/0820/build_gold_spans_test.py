#!/usr/bin/env python3
"""test 149 gold(element id) → char span 채점 파일 생성 — map_lsh_gold.py 와 동일 방식(strict).

필요 입력: cowork "원문 span 매핑 파일.jsonl" (lsh element_id → 원문 발췌 텍스트).
  * hybrid-enrich/out/elements.jsonl 로는 만들 수 없다 — element_id 가 같아도 다른 우주다
    (검증: train 348 재구성 시 기존 채점파일과 span 겹침 0.5%).
채점 규칙(map_lsh_gold strict 와 동일): gold element 텍스트를 map_group 으로 앵커링 →
  1) 인접 span 병합(gap<=30)  2) 같은 텍스트(앞 160자) group 은 OR(a/a' 동치)  3) 남은 group = AND.

Usage:
    python build_gold_spans_test.py --spanmap "<원문 span 매핑 파일.jsonl>"
    python build_gold_spans_test.py --spanmap ... --validate   # train v2 재구성 → v2 span gold 대조

기본 test 출력은 ``out/gold_spans_lsh_test_v2.jsonl``이며, gold가 있는 문항이
하나라도 partial이면 실패한다. 성공 시 입력·코퍼스·index 해시와 고정 분모를 담은
manifest를 함께 기록한다. ``--allow-partial``은 감사용 초안 생성에만 사용한다.
"""
import argparse, hashlib, json, re, sys, unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
V2OUT = HERE.parents[2] / "out"
DOC = HERE.parents[1] / "doc" / "판매약관_(간편)신한통합건강보장보험 원(ONE)(무배당, 해약환급금 미지급형)_250212.md"
sys.path.insert(0, str(FS))
from map_gold_spans import norm_map, map_group  # noqa: E402

GAP, KEY = 30, 160
DROP_LINE = re.compile(r"-{3,}|\d{1,4}|SHINHAN LIFE|_{3,}|\d{10,}")
DEFAULT_TRAIN = V2OUT / "gold_mapped_noah_v3_348_train_v2.jsonl"
DEFAULT_TRAIN_REF = FS / "out" / "gold_spans_lsh_train_v2.jsonl"
DEFAULT_TEST = V2OUT / "gold_mapped_noah_v3_149_test.jsonl"
DEFAULT_OUT = HERE / "out" / "gold_spans_lsh_test_v2.jsonl"
DEFAULT_JO = FS / "out" / "elements_u2jo.jsonl"


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


def validate(rows: list, ref_path: Path) -> dict:
    ref = {json.loads(l)["qid"]: json.loads(l) for l in open(ref_path, encoding="utf-8")}
    rebuilt = {row["qid"]: row for row in rows}
    ref_scored = sum(bool(row.get("groups")) and row.get("status", "ok") == "ok" for row in ref.values())
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
    result = {
        "common_scored_q": n_q,
        "reference_scored_q": ref_scored,
        "common_scored_rate": n_q / max(ref_scored, 1),
        "qid_sets_equal": set(rebuilt) == set(ref),
        "qid_only_rebuilt": sorted(set(rebuilt) - set(ref)),
        "qid_only_reference": sorted(set(ref) - set(rebuilt)),
        "group_count_equal_q": n_grp_eq,
        "group_count_equal_rate": n_grp_eq / max(n_q, 1),
        "span_overlap_members": n_span_hit,
        "span_members": n_span_tot,
        "span_overlap_rate": n_span_hit / max(n_span_tot, 1),
    }
    print(f"[검증] QID 동일 {result['qid_sets_equal']} / 공통 채점문항 {n_q}/{ref_scored} "
          f"({100*result['common_scored_rate']:.1f}%) / group 수 일치 {n_grp_eq} ({100*result['group_count_equal_rate']:.1f}%) / "
          f"span 겹침 {n_span_hit}/{n_span_tot} ({100*result['span_overlap_rate']:.1f}%)")
    return result


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def overlaps(unit: dict, group: dict) -> bool:
    members = group.get("members") or [group]
    return any(unit["char_start"] < m["c1"] and unit["char_end"] > m["c0"] for m in members)


def reachability(rows: list, jo_path: Path, cap: int = 10) -> dict:
    """각 group의 index 도달성과 cap개 조로 모든 group을 덮을 수 있는지 검사한다."""
    units = [json.loads(l) for l in open(jo_path, encoding="utf-8")]
    unreachable = []
    over_cap = []
    min_cover = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        groups = row["groups"]
        masks = {
            sum(1 << i for i, group in enumerate(groups) if overlaps(unit, group))
            for unit in units
        }
        masks.discard(0)
        for i in range(len(groups)):
            if not any(mask & (1 << i) for mask in masks):
                unreachable.append({"qid": row["qid"], "group": i})
        full = (1 << len(groups)) - 1
        dp = {0: 0}
        for mask in masks:
            for old, count in list(dp.items()):
                merged = old | mask
                if count + 1 < dp.get(merged, cap + 1) and count + 1 <= cap:
                    dp[merged] = count + 1
        cover = dp.get(full)
        min_cover[row["qid"]] = cover
        if cover is None:
            over_cap.append(row["qid"])
    return {
        "submit_cap": cap,
        "unreachable_groups": unreachable,
        "over_cap_qids": over_cap,
        "max_min_cover": max((n for n in min_cover.values() if n is not None), default=0),
        "min_cover_by_qid": min_cover,
    }


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spanmap", type=Path, required=True, help='cowork "원문 span 매핑 파일.jsonl" 경로')
    ap.add_argument("--validate", action="store_true", help="train v2 재구성 → 기존 v2 채점파일과 대조만")
    ap.add_argument("--train-gold", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--train-ref", type=Path, default=DEFAULT_TRAIN_REF)
    ap.add_argument("--test-gold", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--jo", type=Path, default=DEFAULT_JO)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--manifest", type=Path, default=None, help="기본값: <out>.manifest.json")
    ap.add_argument("--allow-partial", action="store_true", help="미매핑 gold가 있어도 감사용 초안을 기록")
    ap.add_argument("--expected-empty", type=int, default=11, help="사전 고정한 empty-gold 문항 수")
    ap.add_argument("--submit-cap", type=int, default=10)
    a = ap.parse_args()
    for path in (a.spanmap, DOC, a.jo):
        if not path.exists():
            raise SystemExit(f"[오류] 필수 입력 없음: {path}")
    raw = unicodedata.normalize("NFC", open(DOC, encoding="utf-8").read().replace("\r\n", "\n"))
    nd, idx = norm_map(raw)
    S = {json.loads(l)["element_id"]: json.loads(l) for l in open(a.spanmap, encoding="utf-8")}
    print(f"spanmap elements: {len(S):,}")
    if a.validate:
        rows = build(a.train_gold, S, nd, idx, raw)
        result = validate(rows, a.train_ref)
        if not result["qid_sets_equal"]:
            raise SystemExit("[실패] train v2 QID 집합 불일치")
        if result["common_scored_rate"] < 0.95:
            raise SystemExit("[실패] train v2 유효 문항 재구성률이 95% 미만")
        if result["group_count_equal_rate"] < 0.95:
            raise SystemExit("[실패] train v2 group 수 일치율이 95% 미만")
        if result["span_overlap_rate"] < 0.99:
            raise SystemExit("[실패] train v2 span overlap이 99% 미만 — 다른 element 우주일 가능성")
        return
    rows = build(a.test_gold, S, nd, idx, raw)
    ok = sum(1 for r in rows if r["status"] == "ok")
    partial = [r for r in rows if r["status"] == "partial"]
    empty = [r for r in rows if r["status"] == "empty_gold"]
    if len(empty) != a.expected_empty:
        raise SystemExit(f"[실패] empty-gold 분모 변경: expected={a.expected_empty}, actual={len(empty)}")
    if partial and not a.allow_partial:
        detail = [(r["qid"], r.get("unmapped", [])) for r in partial]
        raise SystemExit(f"[실패] gold 보유 문항 미매핑 {len(partial)}개: {detail}")

    reach = reachability(rows, a.jo, a.submit_cap)
    if (reach["unreachable_groups"] or reach["over_cap_qids"]) and not a.allow_partial:
        raise SystemExit(
            f"[실패] 채점 구조 검증: unreachable={reach['unreachable_groups']} "
            f"over_cap={reach['over_cap_qids']}"
        )

    write_jsonl(a.out, rows)
    manifest_path = a.manifest or a.out.with_suffix(".manifest.json")
    manifest = {
        "frozen_denominator": ok,
        "n_source": len(rows),
        "n_ok": ok,
        "n_partial": len(partial),
        "n_empty_gold": len(empty),
        "partial": [{"qid": r["qid"], "unmapped": r.get("unmapped", [])} for r in partial],
        "empty_gold_qids": [r["qid"] for r in empty],
        "reachability": reach,
        "sources": {
            "spanmap": {"path": str(a.spanmap.resolve()), "sha256": sha256(a.spanmap)},
            "test_gold": {"path": str(a.test_gold.resolve()), "sha256": sha256(a.test_gold)},
            "document": {"path": str(DOC.resolve()), "sha256": sha256(DOC)},
            "jo_index": {"path": str(a.jo.resolve()), "sha256": sha256(a.jo)},
        },
        "output": {"path": str(a.out.resolve()), "sha256": sha256(a.out)},
        "rule": f"strict span groups; submit_cap={a.submit_cap}; empty_gold excluded",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"test {len(rows)} → ok {ok} / partial {len(partial)} / empty {len(empty)} -> {a.out}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
