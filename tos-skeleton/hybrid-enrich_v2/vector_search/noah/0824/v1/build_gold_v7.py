# -*- coding: utf-8 -*-
"""
gold v7 빌더 — R@5 목표에 맞춘 정답셋.

v6 대비 변경점 (사용자 요구 2026-08-24):
  1. required 그룹은 문항당 **최대 5개**. 초과분은 삭제하지 않고 명세 §2 의
     `supporting` 으로 강등한다 (근거는 파일에 남고 채점에서만 빠진다).
     -> 모든 문항의 R@5 산술 상한이 1.0000 이 된다.
  2. **빈 배열 금지.** 조가 안 잡히는 그룹 제거, groups/members/jos 가 빈 레코드 제거,
     값이 없는 키는 아예 쓰지 않는다.

required 선별 기준 — 인간이 쓴 `정답(종합)` 본문과의 문자 bigram 겹침.
"답을 쓰는 데 실제로 쓰인 근거"가 required 다. 동점은 인용 순서(앞선 인용이 주근거),
그다음 근거 길이로 가른다.

나머지는 v6 과 동일하게 명세 v1 (2026-08-18) 준수:
  §1 gap<=200 병합 + 같은 조 병합   §2 required/supporting
  §3 동일문구 확장 160자(lenient) · 특약 지정 질문 scope 제한 · 퍼지 금지
  §5 broken 제외                    §6.4 test 봉인
"""
import csv, json, re, ast, sys, io, hashlib, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2]
R = EXP / "tos-skeleton" / "hybrid-enrich_v2"
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)

CSV_PATH = EXP / "docs" / "0805" / "정답셋_503_현업303_문서보강200_v2_직접검수.csv"
MD_PATH = R / "vector_search" / "doc" / "판매약관_(간편)신한통합건강보장보험 원(ONE)(무배당, 해약환급금 미지급형)_250212.md"
JO_PATH = R / "filesearch" / "out" / "elements_u2jo.jsonl"

MERGE_GAP = 200
OCC_PREFIX = 160
OCC_MIN = 40
OR_CAP = 50
TITLE_MAX = 45
MAX_REQUIRED = 5          # ★ R@5 산술 상한 1.0 보장


def nz(s):
    return "".join(c for c in s if not c.isspace())


def bigrams(s):
    n = nz(s)
    return {n[i:i + 2] for i in range(len(n) - 1)}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


TXT = MD_PATH.read_text(encoding="utf-8")
_ch, IMAP = [], []
for _i, _c in enumerate(TXT):
    if not _c.isspace():
        _ch.append(_c); IMAP.append(_i)
NORM = "".join(_ch)


def norm_to_orig(p, ln):
    return IMAP[p], IMAP[p + ln - 1] + 1


def find_all_norm(sub, cap):
    out, p = [], NORM.find(sub)
    while p >= 0 and len(out) < cap:
        out.append(p); p = NORM.find(sub, p + 1)
    return out


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
SCOPE_CORE = [(c, sc) for c, sc in ((scope_core(x), x) for x in SCOPES) if len(c) >= 6]


def detect_scope(q):
    nq, best = nz(q), None
    for core, sc in SCOPE_CORE:
        if core in nq and (best is None or len(core) > len(best[0])):
            best = (core, sc)
    return best[1] if best else None


def locate_all(nq, scope=None):
    """
    인용문을 원문에 정위한다. **모든 출현**을 돌려준다.

    약관은 같은 조항이 특약마다 반복된다 — 앞 25자 앵커가 157회까지 등장한다.
    첫 출현만 찍으면 엉뚱한 특약을 정답으로 박게 되므로(2026-08-24 발견한 결함),
    동일 문구는 전부 OR 멤버로 두고, 질문이 특약을 지정하면 그 scope 로 좁힌다(명세 §3).

    반환: (spans[list[(a,b)]], tag)
    """
    if len(nq) < 8:
        return [], "short"
    probe, tag = None, None
    if NORM.find(nq) >= 0:
        probe, tag = nq, "full"
    else:
        for ln, t in ((40, "head40"), (25, "head25")):
            if len(nq) >= ln and NORM.find(nq[:ln]) >= 0:
                probe, tag = nq[:ln], t
                break
        if probe is None and len(nq) >= 40 and NORM.find(nq[-40:]) >= 0:
            probe, tag = nq[-40:], "tail40"
    if probe is None:
        return [], "miss"

    spans = [norm_to_orig(p, len(probe)) for p in find_all_norm(probe, OR_CAP * 4)]
    if scope and len(spans) > 1:
        inscope = [s for s in spans if any(sc == scope for _, sc in jos_of(*s))]
        if inscope:
            spans = inscope
            tag += "+scope"
    return spans[:OR_CAP], tag


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
    recs, audit, cit_stat = [], [], collections.Counter()
    demoted_total = 0

    for r in conf:
        no, biz, q, ans = int(r[0]), r[1], r[2], r[3]
        meta = BY_NO.get(no) or BY_Q.get(nz(q))
        if meta:
            qid, split, ttype, core, biz2 = meta
            qsrc = "v4_no" if no in BY_NO else "question_text"
        else:
            qid, split, ttype, core, biz2 = "v7-new-%04d" % no, "train", None, None, None
            qsrc = "new(train; test 봉인 §6.4)"

        try:
            cits = ast.literal_eval(r[5]) if r[5].strip() else []
        except Exception:
            cits = []
        quotes = [c["quote"] for c in cits if isinstance(c, dict) and c.get("quote")]

        scope = detect_scope(q)
        spans, alts, how = [], {}, collections.Counter()
        for ci, qt in enumerate(quotes):
            sl, tag = locate_all(nz(qt), scope)
            how[tag] += 1; cit_stat[tag] += 1
            if sl:
                spans.append((sl[0][0], sl[0][1], ci))
                alts[ci] = sl[1:]          # 동일 문구의 나머지 출현 = OR 멤버

        n_cit, n_loc = len(quotes), len(spans)
        if n_loc == 0:
            audit.append({"qid": qid, "no": no, "decision": "removed",
                          "reason": "인용 전무 — 모든 인용이 채점 코퍼스 밖 (명세 §5)",
                          "n_cit": n_cit, "q": q})
            continue

        # §1 gap<=200 병합
        spans.sort()
        merged = []
        for a, b, ci in spans:
            if merged and a - merged[-1][1] <= MERGE_GAP:
                merged[-1][1] = max(merged[-1][1], b)
                merged[-1][2] = min(merged[-1][2], ci)
                merged[-1][3].append((a, b, ci))
            else:
                merged.append([a, b, ci, [(a, b, ci)]])

        # 같은 조에 속하면 한 그룹 (채점 단위가 조)
        parent = list(range(len(merged)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x

        owner = {}
        for i, (a, b, _, _) in enumerate(merged):
            for j, _ in jos_of(a, b):
                if j in owner:
                    rx, ry = find(owner[j]), find(i)
                    if rx != ry:
                        parent[max(rx, ry)] = min(rx, ry)
                else:
                    owner[j] = i
        clus = collections.OrderedDict()
        for i in range(len(merged)):
            clus.setdefault(find(i), []).append(i)

        ans_bg = bigrams(ans or "")
        cands = []
        for idxs in clus.values():
            a = min(merged[i][0] for i in idxs)
            b = max(merged[i][1] for i in idxs)
            order = min(merged[i][2] for i in idxs)
            parts = sorted({p for i in idxs for p in merged[i][3]})
            jl = jos_of(a, b)
            jos = sorted({j for j, _ in jl})
            if not jos:                       # 빈 배열 금지 — 조 없는 그룹은 버린다
                continue
            body = TXT[parts[0][0]:parts[0][1]]
            nb = nz(body)
            gb = bigrams(body)
            # 답을 쓰는 데 실제로 쓰인 근거인가 — 정답 본문과의 bigram 겹침
            usage = len(gb & ans_bg) / max(1, len(gb)) if gb else 0.0
            cands.append({"a": a, "b": b, "parts": parts, "jos": jos,
                          "scopes": sorted({s for _, s in jl if s}),
                          "key": nb[:120], "nb": nb, "body": body,
                          "usage": round(usage, 4), "order": order})
        if not cands:
            audit.append({"qid": qid, "no": no, "decision": "removed",
                          "reason": "정위는 됐으나 조 매핑 실패 — 채점 불가",
                          "n_cit": n_cit, "q": q})
            continue

        # 제목류는 우선 supporting (명세 §2 자동 초벌)
        for c in cands:
            bs = c["body"].strip()
            c["is_title"] = (len(c["nb"]) < TITLE_MAX
                             and not re.search(r"[.」\]]\s*$", bs)
                             and bool(re.match(r"^\s*(제\s*\d+\s*조|[\[<(])", bs)))

        # ★ required <= 5 : usage 내림차순 -> 인용 순서 -> 길이 내림차순
        ranked = sorted(cands, key=lambda c: (c["is_title"], -c["usage"],
                                              c["order"], -len(c["nb"])))
        for i, c in enumerate(ranked):
            c["grade"] = "required" if i < MAX_REQUIRED else "supporting"
        if all(c["grade"] != "required" for c in ranked):
            ranked[0]["grade"] = "required"
        n_dem = sum(1 for c in ranked if c["grade"] == "supporting")
        demoted_total += n_dem

        groups_s, groups_l = [], []
        for gi, c in enumerate(ranked):
            base = {"gid": gi, "grade": c["grade"], "c0": c["a"], "c1": c["b"],
                    "key": c["key"], "jos": c["jos"], "usage": c["usage"]}
            if c["scopes"]:
                base["scopes"] = c["scopes"]
            # 등록 위치 = 정위된 span + 그 인용문의 나머지 동일 출현(OR).
            # 첫 출현만 쓰면 반복 조항에서 엉뚱한 특약을 가리키게 된다.
            located, seen = [], set()
            for pa, pb, ci in c["parts"]:
                if (pa, pb) not in seen:
                    located.append({"c0": pa, "c1": pb, "src": "located"}); seen.add((pa, pb))
                for oa, ob in alts.get(ci, []):
                    if (oa, ob) not in seen and len(located) < OR_CAP:
                        located.append({"c0": oa, "c1": ob, "src": "same_quote"}); seen.add((oa, ob))
            groups_s.append(dict(base, members=list(located)))

            # 명세 §3 — 추가로 앞 160자 접두 일치까지 확장(부지표용)
            mem = list(located)
            for pa, pb, _ci in c["parts"]:
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

        n_req = sum(1 for c in ranked if c["grade"] == "required")
        common = {"qid": qid, "q": q, "no": no, "answer": ans,
                  "사업구분": biz2 or biz, "split": split, "qid_src": qsrc,
                  "n_cit": n_cit, "n_cit_located": n_loc, "partial": n_loc < n_cit,
                  "n_required": n_req, "n_supporting": n_dem,
                  "status": "ok", "gold_src": "v7-현업303"}
        if ttype:
            common["task_type"] = ttype
        if core:
            common["core_retrieval"] = core
        if scope:
            common["scope"] = scope; common["scope_gated"] = True
        if how:
            common["locate_detail"] = dict(how)

        recs.append((dict(common, groups=groups_s, gold_mode="strict"),
                     dict(common, groups=groups_l, gold_mode="lenient")))
        audit.append({"qid": qid, "no": no, "decision": "kept", "n_cit": n_cit,
                      "n_located": n_loc, "n_groups": len(ranked),
                      "n_required": n_req, "n_demoted": n_dem,
                      "partial": n_loc < n_cit, "split": split, "qid_src": qsrc})

    # 답안지에 같은 질문이 중복 수록된 경우가 있다(no 335/342 등).
    # qid 당 1행만 남긴다 — 정위 인용이 많은 쪽, 동점이면 낮은 no.
    best = {}
    for pair in recs:
        d = pair[0]
        k = d["qid"]
        cur = best.get(k)
        if cur is None or (d["n_cit_located"], -d["no"]) > (cur[0]["n_cit_located"], -cur[0]["no"]):
            if cur is not None:
                audit.append({"qid": k, "no": cur[0]["no"], "decision": "dedup_dropped",
                              "reason": "답안지 질문 중복 — 정위 인용이 더 많은 행을 채택",
                              "kept_no": d["no"]})
            best[k] = pair
        else:
            audit.append({"qid": k, "no": d["no"], "decision": "dedup_dropped",
                          "reason": "답안지 질문 중복 — 정위 인용이 더 많은 행을 채택",
                          "kept_no": cur[0]["no"]})
    recs = [best[d[0]["qid"]] for d in recs if best.get(d[0]["qid"]) is d
            or best[d[0]["qid"]][0]["no"] == d[0]["no"]]
    seen, uniq = set(), []
    for pair in recs:
        if pair[0]["qid"] in seen:
            continue
        seen.add(pair[0]["qid"]); uniq.append(pair)
    return uniq, audit, cit_stat, demoted_total


def main():
    recs, audit, cit_stat, demoted = build()
    counts = collections.Counter()
    for idx, mode in ((0, "strict"), (1, "lenient")):
        for split in ("train", "test"):
            sel = [p[idx] for p in recs if p[idx]["split"] == split]
            f = OUT / ("gold_v7_%s_%s.jsonl" % (split, mode))
            with f.open("w", encoding="utf-8") as w:
                for d in sel:
                    w.write(json.dumps(d, ensure_ascii=False) + "\n")
            counts["%s_%s" % (split, mode)] = len(sel)

    st = [p[0] for p in recs]
    G = [d["n_required"] for d in st]
    ceil5 = sum(min(5, g) / g for g in G) / len(G)
    man = {
        "version": "v7",
        "spec": "정답기준_명세_v1_제안_20260818.md",
        "changes_vs_v6": ["required <= 5 (초과분은 supporting 강등)",
                          "빈 배열 금지 (조 없는 그룹 제거, 값 없는 키 생략)"],
        "params": {"merge_gap": MERGE_GAP, "occ_prefix": OCC_PREFIX, "occ_min": OCC_MIN,
                   "or_cap": OR_CAP, "title_max": TITLE_MAX, "max_required": MAX_REQUIRED,
                   "required_rank": "정답본문 bigram 겹침(usage) desc -> 인용순서 -> 길이 desc",
                   "fuzzy": "금지 (명세 §3)"},
        "target_metric": "R@5 (융합 meta+tag arm)",
        "inputs": {"csv": {"path": str(CSV_PATH), "sha256": sha(CSV_PATH)},
                   "corpus": {"path": str(MD_PATH), "sha256": sha(MD_PATH)},
                   "jo_universe": {"path": str(JO_PATH), "sha256": sha(JO_PATH)}},
        "counts": dict({"csv_확정": 303, "kept": len(st),
                        "removed": sum(1 for a in audit if a["decision"] == "removed"),
                        "fully_located": sum(1 for d in st if not d["partial"]),
                        "partial": sum(1 for d in st if d["partial"]),
                        "demoted_to_supporting": demoted}, **dict(counts)),
        "required_per_q": {"mean": round(sum(G) / len(G), 3), "max": max(G),
                           "hist": dict(collections.Counter(G))},
        "R@5_arithmetic_ceiling": round(ceil5, 4),
        "citation_locate": dict(cit_stat),
    }
    (OUT / "gold_v7_manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "gold_v7_audit.jsonl").open("w", encoding="utf-8") as w:
        for a in audit:
            w.write(json.dumps(a, ensure_ascii=False) + "\n")
    print(json.dumps({k: man[k] for k in
                      ("counts", "required_per_q", "R@5_arithmetic_ceiling")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
