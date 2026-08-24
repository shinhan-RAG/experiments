# -*- coding: utf-8 -*-
"""
gold v6 빌더 — 현업 303 확정 답안지를 원문에 정위해 채점 가능한 gold 로 만든다.

정답기준 명세 v1 (2026-08-18) 준수:
  §1 병합 gap <= 200자
  §2 required / supporting 등급 (채점은 required 만)
  §3 동일문구 출현 확장 160자 정확일치 (lenient), 특약 지정 질문은 scope 제한
  §4 주지표 sufficient@10
  §5 broken(인용 전무) 제외, 모집단 명기
  §6 test 봉인 — 신규 문항은 train 으로만

입력
  docs/0805/정답셋_503_현업303_문서보강200_v2_직접검수.csv  (확정 303행)
  vector_search/doc/판매약관_...250212.md                   (채점 코퍼스)
  filesearch/out/elements_u2jo.jsonl                        (조 단위, 커버리지 98.69%)
  out/noah/gold_v4_*.jsonl, out/gold_mapped_noah_v3_*.jsonl (qid/split 승계)

출력  out/gold_v6_{split}_{strict|lenient}.jsonl + manifest + audit
"""
import csv, json, re, ast, sys, io, hashlib, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP  = HERE.parents[2]                       # .../experiments
R    = EXP / "tos-skeleton" / "hybrid-enrich_v2"
OUT  = HERE / "out"; OUT.mkdir(exist_ok=True)

CSV_PATH = EXP / "docs" / "0805" / "정답셋_503_현업303_문서보강200_v2_직접검수.csv"
MD_PATH  = R / "vector_search" / "doc" / "판매약관_(간편)신한통합건강보장보험 원(ONE)(무배당, 해약환급금 미지급형)_250212.md"
JO_PATH  = R / "filesearch" / "out" / "elements_u2jo.jsonl"

MERGE_GAP  = 200   # 명세 §1
OCC_PREFIX = 160   # 명세 §3 동일문구 확장 기준 길이
OCC_MIN    = 40    # 40자 미만은 확장 금지
OR_CAP     = 50    # OR 멤버 상한
TITLE_MAX  = 45    # 명세 §2 자동 초벌: 45자 미만 비문장 = supporting


def nz(s):
    return "".join(c for c in s if not c.isspace())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------- 코퍼스 + 정규화 인덱스 ----------
TXT = MD_PATH.read_text(encoding="utf-8")
_ch, IMAP = [], []
for _i, _c in enumerate(TXT):
    if not _c.isspace():
        _ch.append(_c); IMAP.append(_i)
NORM = "".join(_ch)


def norm_to_orig(p, ln):
    """정규화 좌표 [p, p+ln) -> 원문 [start, end)"""
    return IMAP[p], IMAP[p + ln - 1] + 1


def find_all_norm(sub, cap):
    out, p = [], NORM.find(sub)
    while p >= 0 and len(out) < cap:
        out.append(p)
        p = NORM.find(sub, p + 1)
    return out


# ---------- 조 우주 ----------
JOS = []
for _line in JO_PATH.open(encoding="utf-8"):
    _d = json.loads(_line)
    JOS.append((int(_d["char_start"]), int(_d["char_end"]),
                _d["element_id"], _d.get("contract_scope") or ""))
JOS.sort()


def jos_of(a, b):
    return [(j, sc) for (s, e, j, sc) in JOS if s < b and e > a]


def scope_core(sc):
    s = re.sub(r"\(무배당[^)]*\)|\(간편\)|\(갱신형\)|\(해약환급금[^)]*\)", "", sc)
    return nz(s.strip())


SCOPES = sorted({sc for (_, _, _, sc) in JOS if sc}, key=len, reverse=True)
SCOPE_CORE = [(scope_core(sc), sc) for sc in SCOPES]
SCOPE_CORE = [(c, sc) for c, sc in SCOPE_CORE if len(c) >= 6]


def detect_scope(q):
    """명세 §3 — 질문에 특약명이 명시되면 그 scope 로 제한(특약 지정 질문)."""
    nq = nz(q)
    best = None
    for core, sc in SCOPE_CORE:
        if core and core in nq and (best is None or len(core) > len(best[0])):
            best = (core, sc)
    return best[1] if best else None


# ---------- 인용 정위 (명세 §3: 퍼지 금지) ----------
def locate(nq):
    if len(nq) < 8:
        return None, "short"
    p = NORM.find(nq)
    if p >= 0:
        return norm_to_orig(p, len(nq)), "full"
    for ln, tag in ((40, "head40"), (25, "head25")):
        if len(nq) >= ln:
            p = NORM.find(nq[:ln])
            if p >= 0:
                return norm_to_orig(p, ln), tag
    if len(nq) >= 40:
        p = NORM.find(nq[-40:])
        if p >= 0:
            return norm_to_orig(p, 40), "tail40"
    return None, "miss"


# ---------- qid / split 승계 ----------
def load_qid_maps():
    by_no, by_q = {}, {}
    for split, f in (("test", R / "out" / "noah" / "gold_v4_test_full.jsonl"),
                     ("train", R / "out" / "noah" / "gold_v4_train_full.jsonl")):
        for line in f.open(encoding="utf-8"):
            d = json.loads(line)
            meta = (d["qid"], split, d.get("task_type"),
                    d.get("core_retrieval"), d.get("사업구분"))
            if d.get("v4_no") is not None:
                by_no[int(d["v4_no"])] = meta
            by_q.setdefault(nz(d["q"]), meta)
    for split, f in (("test", R / "out" / "gold_mapped_noah_v3_149_test.jsonl"),
                     ("train", R / "out" / "gold_mapped_noah_v3_348_train_v2.jsonl")):
        for line in f.open(encoding="utf-8"):
            d = json.loads(line)
            by_q.setdefault(nz(d["q"]), (d["qid"], split, d.get("task_type"),
                                         d.get("core_retrieval"), d.get("사업구분")))
    return by_no, by_q


BY_NO, BY_Q = load_qid_maps()


def build():
    rows = list(csv.reader(CSV_PATH.open(encoding="utf-8-sig")))[1:]
    conf = [r for r in rows if r[4] == "확정"]
    recs, audit = [], []
    cit_stat = collections.Counter()

    for r in conf:
        no, biz, q, ans = int(r[0]), r[1], r[2], r[3]
        meta = BY_NO.get(no) or BY_Q.get(nz(q))
        if meta:
            qid, split, ttype, core, biz2 = meta
            src = "v4_no" if no in BY_NO else "question_text"
        else:
            qid, split, ttype, core, biz2 = "v6-new-%04d" % no, "train", None, None, None
            src = "new(train; test 봉인 §6.4)"

        try:
            cits = ast.literal_eval(r[5]) if r[5].strip() else []
        except Exception:
            cits = []
        quotes = [c["quote"] for c in cits if isinstance(c, dict) and c.get("quote")]

        spans, how = [], collections.Counter()
        for qt in quotes:
            sp, tag = locate(nz(qt))
            how[tag] += 1
            cit_stat[tag] += 1
            if sp:
                spans.append(sp)

        n_cit, n_loc = len(quotes), len(spans)
        if n_loc == 0:
            audit.append({"qid": qid, "no": no, "decision": "removed",
                          "reason": "인용 전무 — 모든 인용이 채점 코퍼스 밖 (명세 §5 broken)",
                          "n_cit": n_cit, "q": q})
            continue

        # 명세 §1 — gap<=200 병합
        spans.sort()
        merged = []
        for a, b in spans:
            if merged and a - merged[-1][1] <= MERGE_GAP:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])

        # 채점 단위가 조이므로 같은 조에 속한 span 은 한 그룹으로 병합한다.
        # (regold_v4_full.py 와 동일 규칙 — 분모 부풀림 방지)
        parent = list(range(len(merged)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)

        jo_owner = {}
        span_jos = []
        for i, (a, b) in enumerate(merged):
            js = {j for j, _ in jos_of(a, b)}
            span_jos.append(js)
            for j in js:
                if j in jo_owner:
                    union(jo_owner[j], i)
                else:
                    jo_owner[j] = i
        clusters = collections.OrderedDict()
        for i in range(len(merged)):
            clusters.setdefault(find(i), []).append(i)
        base_spans = merged
        merged = [[min(base_spans[i][0] for i in idxs),
                   max(base_spans[i][1] for i in idxs),
                   [tuple(base_spans[i]) for i in idxs]]
                  for idxs in clusters.values()]

        scope = detect_scope(q)
        groups_s, groups_l = [], []
        for gi, (a, b, parts) in enumerate(merged):
            body = TXT[parts[0][0]:parts[0][1]]
            nb = nz(body)
            is_title = (len(nb) < TITLE_MAX
                        and not re.search(r"[.」\]]\s*$", body.strip())
                        and bool(re.match(r"^\s*(제\s*\d+\s*조|[\[<(])", body.strip())))
            grade = "supporting" if is_title else "required"
            jl = jos_of(a, b)
            base = {"gid": gi, "grade": grade, "c0": a, "c1": b,
                    "key": nb[:120],
                    "jos": sorted({j for j, _ in jl}),
                    "scopes": sorted({s for _, s in jl if s})}

            # 등록 위치 = 실제 정위된 span 들 (봉투 [a,b] 가 아니라 조각 그대로)
            located = [{"c0": pa, "c1": pb, "src": "located"} for pa, pb in parts]
            groups_s.append(dict(base, members=list(located)))

            # 명세 §3 — lenient 동일문구 출현 확장
            mem = list(located)
            seen = {(m["c0"], m["c1"]) for m in mem}
            for pa, pb in parts:
                pn = nz(TXT[pa:pb])
                if len(pn) < OCC_MIN or len(mem) >= OR_CAP:
                    continue
                probe = pn[:OCC_PREFIX]
                for p in find_all_norm(probe, OR_CAP):
                    oa, ob = norm_to_orig(p, len(probe))
                    if (oa, ob) in seen:
                        continue
                    if scope and not any(s == scope for _, s in jos_of(oa, ob)):
                        continue
                    mem.append({"c0": oa, "c1": ob, "src": "same_text"})
                    seen.add((oa, ob))
                    if len(mem) >= OR_CAP:
                        break
            groups_l.append(dict(base, members=mem))

        common = {"qid": qid, "q": q, "no": no, "answer": ans,
                  "사업구분": biz2 or biz, "task_type": ttype, "core_retrieval": core,
                  "split": split, "qid_src": src,
                  "n_cit": n_cit, "n_cit_located": n_loc,
                  "partial": n_loc < n_cit, "locate_detail": dict(how),
                  "scope": scope, "scope_gated": bool(scope),
                  "status": "ok", "gold_src": "v6-현업303"}
        recs.append((dict(common, groups=groups_s, gold_mode="strict"),
                     dict(common, groups=groups_l, gold_mode="lenient")))
        audit.append({"qid": qid, "no": no, "decision": "kept",
                      "n_cit": n_cit, "n_located": n_loc,
                      "partial": n_loc < n_cit, "n_groups": len(merged),
                      "scope": scope, "split": split, "qid_src": src})
    return recs, audit, cit_stat


def main():
    recs, audit, cit_stat = build()
    counts = collections.Counter()
    for idx, mode in ((0, "strict"), (1, "lenient")):
        for split in ("train", "test"):
            sel = [p[idx] for p in recs if p[idx]["split"] == split]
            f = OUT / ("gold_v6_%s_%s.jsonl" % (split, mode))
            with f.open("w", encoding="utf-8") as w:
                for d in sel:
                    w.write(json.dumps(d, ensure_ascii=False) + "\n")
            counts["%s_%s" % (split, mode)] = len(sel)

    strict = [p[0] for p in recs]
    man = {
        "spec": "정답기준_명세_v1_제안_20260818.md",
        "params": {"merge_gap": MERGE_GAP, "occ_prefix": OCC_PREFIX,
                   "occ_min": OCC_MIN, "or_cap": OR_CAP, "title_max": TITLE_MAX,
                   "fuzzy": "금지 (명세 §3) — 정확일치 + head/tail 앵커만"},
        "primary_metric": "sufficient@10 (명세 §4)",
        "inputs": {"csv": {"path": str(CSV_PATH), "sha256": sha(CSV_PATH)},
                   "corpus": {"path": str(MD_PATH), "sha256": sha(MD_PATH)},
                   "jo_universe": {"path": str(JO_PATH), "sha256": sha(JO_PATH)}},
        "counts": dict({
            "csv_확정": 303,
            "kept": len(strict),
            "removed_인용전무": sum(1 for a in audit if a["decision"] == "removed"),
            "fully_located": sum(1 for d in strict if not d["partial"]),
            "partial": sum(1 for d in strict if d["partial"]),
            "scope_gated": sum(1 for d in strict if d["scope_gated"]),
        }, **dict(counts)),
        "citation_locate": dict(cit_stat),
        "groups": {
            "total": sum(len(d["groups"]) for d in strict),
            "required": sum(1 for d in strict for g in d["groups"] if g["grade"] == "required"),
            "supporting": sum(1 for d in strict for g in d["groups"] if g["grade"] == "supporting"),
        },
        "qid_src": dict(collections.Counter(d["qid_src"] for d in strict)),
        "denominator_policy": ("n = 확정303 − 인용전무(broken). 인덱스 미도달 문항은 분모에 "
                               "유지하고 strict/reachable 층위로 병기 (사용자 결정 2026-08-24)"),
    }
    (OUT / "gold_v6_manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "gold_v6_audit.jsonl").open("w", encoding="utf-8") as w:
        for a in audit:
            w.write(json.dumps(a, ensure_ascii=False) + "\n")

    print(json.dumps(man["counts"], ensure_ascii=False, indent=1))
    print("groups:", json.dumps(man["groups"], ensure_ascii=False))
    print("locate:", json.dumps(man["citation_locate"], ensure_ascii=False))
    print("qid_src:", json.dumps(man["qid_src"], ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
