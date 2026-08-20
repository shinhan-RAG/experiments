#!/usr/bin/env python3
"""QA셋 전체(train 348 + test 149) 감사·gold 교정 — regold_v4.py(train60)의 전체셋 확장.

입력:
  hybrid-enrich_v2/out/gold_mapped_noah_v3_348_train.jsonl  (element id gold)
  hybrid-enrich_v2/out/gold_mapped_noah_v3_149_test.jsonl
  filesearch/out/gold_spans_lsh_train.jsonl                 (train 의 v3 span gold)
  정답셋_359_최종_v4.xlsx                                    (근거 인용 원문)

문항별 판정(성적 무관·문항 내재적 기준만):
  regold    v4 확정 + 인용을 코퍼스에 위치 성공 → v4 기반 gold 재구축
            (일부 인용만 위치되면 keep 하되 c3_partial 플래그 — 분모 병기용)
  keep_v3   v4 미수록 + v3 span gold 정상 (train 만 가능 — test 는 span gold 자체가 없음)
  removed   ① v4 신뢰도='불가'(작성자 판정 답 불가)
            ② v4 확정이지만 인용 전부가 코퍼스(250212판) 밖
            ③ v4 미수록 + v3 span gold 부재/빈 gold (test 전체의 미매칭 포함)

출력: out/gold_v4_train_full.jsonl, out/gold_v4_test_full.jsonl, out/regold_full_audit.json
"""
import json, re, sys, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
V2 = HERE.parents[2]
DOC = HERE.parents[1] / "doc" / "판매약관_(간편)신한통합건강보장보험 원(ONE)(무배당, 해약환급금 미지급형)_250212.md"
XLSX = V2.parent.parent / "정답셋_359_최종_v4.xlsx"
sys.path.insert(0, str(FS))
from units import Units
UJO = Units(FS / "out/elements_u2jo.jsonl")

def norm(s):
    return re.sub(r"[^가-힣a-zA-Z0-9]", "", str(s).lower()) if s else ""

def bigrams(s):
    return {s[i:i + 2] for i in range(len(s) - 1)} or ({s} if s else set())

def sim(a, b):
    A, B = bigrams(a), bigrams(b)
    return len(A & B) / max(1, len(A | B))

doc = DOC.read_text(encoding="utf-8")
keep = [(i, ch.lower()) for i, ch in enumerate(doc) if re.match(r"[가-힣a-zA-Z0-9]", ch)]
nstream = "".join(ch for _, ch in keep)
npos = [i for i, _ in keep]

def find_all(sub):
    out, j = [], nstream.find(sub)
    while j >= 0 and len(out) < 50:
        out.append((npos[j], npos[j + len(sub) - 1] + 1))
        j = nstream.find(sub, j + 1)
    return out

def locate(cit):
    parts = [p for p in re.split(r"\.{2,}|…", str(cit)) if len(norm(p)) >= 6]
    if len(parts) > 1:
        occ = []
        for p in parts:
            o, _ = locate(p)
            if o:
                occ.extend(o)
        return (sorted(set(occ)), "ellipsis") if occ else (None, None)
    nq = norm(cit)
    if len(nq) < 6:
        return None, None
    occ = find_all(nq)
    if occ:
        return occ, "exact"
    for part, tag in ((nq[:60], "head60"), (nq[-60:], "tail60")):
        if len(part) >= 20:
            occ = find_all(part)
            if occ:
                return occ, tag
    L = min(len(nq), 300)
    qb = bigrams(nq[:L])
    best, bj = 0.0, -1
    step = max(20, L // 4)
    for j in range(0, len(nstream) - L + 1, step):
        s = len(qb & bigrams(nstream[j:j + L])) / max(1, len(qb))
        if s > best:
            best, bj = s, j
    if best >= 0.80 and bj >= 0:
        lo, hi = max(0, bj - step), min(len(nstream) - L, bj + step)
        fbest, fj = 0.0, bj
        for j in range(lo, hi + 1, 5):
            s = len(qb & bigrams(nstream[j:j + L])) / max(1, len(qb))
            if s > fbest:
                fbest, fj = s, j
        if fbest >= 0.85:
            return [(npos[fj], npos[fj + L - 1] + 1)], f"fuzzy{fbest:.2f}"
    return None, None

def build_groups(cits):
    """인용 리스트 → evidence groups (same_text_or 군집 + 전 출현 OR + 조 공유 병합)."""
    clusters = []
    for c in cits:
        nc = norm(c)
        for cl in clusters:
            if sim(nc, cl[0]) >= 0.80:
                cl[1].append(c)
                break
        else:
            clusters.append((nc, [c]))
    groups, missed = [], []
    for _, cs in clusters:
        mem = []
        for c in cs:
            occ, _m = locate(c)
            if occ:
                mem.extend(occ)
        if not mem:
            missed.append(cs[0][:80])
            continue
        mem = sorted(set(mem))
        groups.append({"c0": mem[0][0], "c1": mem[0][1],
                       "members": [{"c0": a, "c1": b, "src": "v4"} for a, b in mem],
                       "jos": sorted({(UJO.jo_of_span(a, b) or {}).get("element_id", "?") for a, b in mem})})
    dedup = []
    for gr in groups:
        js = set(gr["jos"])
        for d in dedup:
            if js & set(d["jos"]):
                pairs = sorted({(m["c0"], m["c1"]) for m in d["members"] + gr["members"]})
                d["members"] = [{"c0": a, "c1": b, "src": "v4"} for a, b in pairs]
                d["jos"] = sorted(set(d["jos"]) | js)
                break
        else:
            dedup.append(gr)
    return dedup, missed

# ── v4 로드 ──────────────────────────────────────────────────────────────────
import openpyxl
rows = list(openpyxl.load_workbook(XLSX, read_only=True)["정답셋359_v4"].iter_rows(values_only=True))[1:]
X = [(norm(r[2]), r) for r in rows if r[2]]

def match_v4(q):
    nq = norm(q)
    best = (0.0, None)
    for nx, r in X:
        if nq == nx:
            return (1.0, r)
        s = sim(nq, nx)
        if s > best[0]:
            best = (s, r)
    return best

v3_spans = {g["qid"]: g for g in (json.loads(l) for l in open(FS / "out/gold_spans_lsh_train.jsonl", encoding="utf-8"))}

def process(in_path, out_path, split):
    src = [json.loads(l) for l in open(in_path, encoding="utf-8")]
    out_gold, audit = [], []
    for r in src:
        qid, q = r["qid"], r["q"]
        ent = {"qid": qid, "split": split, "task_type": r.get("task_type"), "q": q}
        s, x = match_v4(q)
        ent["v4_sim"] = round(s, 2)
        if s >= 0.85:
            no, conf, srcs = x[0], x[4], str(x[6] or "")
            ent.update(v4_no=no, conf=conf)
            if conf == "불가":
                ent.update(status="removed", reason="v4 신뢰도=불가(작성자 판정: 답 불가)")
                audit.append(ent); continue
            cits = [c.strip() for c in re.split(r"\[L\d+\]", srcs) if norm(c)]
            groups, missed = build_groups(cits)
            if not groups:
                ent.update(status="removed", reason="v4 인용 근거 전부가 채점 코퍼스(250212판) 밖", missed=missed)
                audit.append(ent); continue
            g = {"qid": qid, "q": q, "task_type": r.get("task_type"), "core_retrieval": r.get("core_retrieval"),
                 "사업구분": r.get("사업구분"), "groups": groups, "status": "ok", "gold_src": "v4", "v4_no": no}
            if missed:
                g["c3_partial"] = True
                ent["missed"] = missed
            out_gold.append(g)
            ent.update(status="regold", n_cit=len(cits), n_groups=len(groups), c3_partial=bool(missed))
            audit.append(ent); continue
        # v4 미수록
        v3 = v3_spans.get(qid) if split == "train" else None
        if v3 and v3.get("status", "ok") == "ok" and v3.get("groups"):
            out_gold.append({**v3, "gold_src": "v3"})
            ent.update(status="keep_v3")
        else:
            why = "v4 미수록 + v3 span gold 부재" if split == "test" else \
                  ("v4 미수록 + v3 gold 비정상(status=%s, groups=%d)" % (v3.get("status") if v3 else "없음", len(v3.get("groups", [])) if v3 else 0))
            ent.update(status="removed", reason=why)
        audit.append(ent)
    with open(out_path, "w", encoding="utf-8") as f:
        for g in out_gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    return audit

audit = []
audit += process(V2 / "out/gold_mapped_noah_v3_348_train.jsonl", HERE / "out/gold_v4_train_full.jsonl", "train")
audit += process(V2 / "out/gold_mapped_noah_v3_149_test.jsonl", HERE / "out/gold_v4_test_full.jsonl", "test")
json.dump(audit, open(HERE / "out/regold_full_audit.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

for split in ("train", "test"):
    A = [a for a in audit if a["split"] == split]
    c = collections.Counter(a["status"] for a in A)
    part = sum(1 for a in A if a.get("c3_partial"))
    print("== %s (%d): %s, c3_partial=%d" % (split, len(A), dict(c), part))
    rc = collections.Counter(a["reason"].split("(")[0] for a in A if a["status"] == "removed")
    for k, v in rc.most_common():
        print("   removed:", k, v)
