#!/usr/bin/env python3
"""정답셋 v4(정답셋_359_최종_v4.xlsx) 기반 train60 gold 교정 재구축.

배경(0820): 전 arm 공통 실패 15문항의 v3 gold 를 감사한 결과 대부분이 질문과 무관한
텍스트(OCR 노이즈·목차 조각·다른 조문)를 정답 span 으로 지정한 오매핑이었다.
v4 정답셋은 문항별 근거 인용 원문([L####] 표기)을 담고 있어, 인용 텍스트를 현 코퍼스에
텍스트 매칭으로 위치시켜 gold 를 재구축한다.

원칙:
- 제외는 문항 내재적 결함만: (a) v4 신뢰도='불가'(작성자 스스로 답 불가 판정),
  (b) v4에서 문항 삭제 + v3 gold 오매핑 확인, (c) 인용 근거가 채점 코퍼스에 부재.
- "실패했으니 제외"는 금지(분모 순환). v4 매칭 문항은 통과/실패 무관하게 전부 재구축(대칭 적용).
- v4 미수록 + v3 gold 정상 문항은 v3 gold 유지.

출력: out/gold_v4_train60.jsonl(교정 gold), out/regold_audit.json(감사 로그·제외 사유)
"""
import json, re, sys, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parent.parent.parent / "filesearch"
sys.path.insert(0, str(FS))
from units import Units
UJO = Units(FS / "out/elements_u2jo.jsonl")
DOC = HERE.parent.parent / "doc" / "판매약관_(간편)신한통합건강보장보험 원(ONE)(무배당, 해약환급금 미지급형)_250212.md"
XLSX = HERE.parent.parent.parent.parent.parent / "정답셋_359_최종_v4.xlsx"

def norm(s):
    return re.sub(r"[^가-힣a-zA-Z0-9]", "", str(s).lower()) if s else ""

def bigrams(s):
    return {s[i:i + 2] for i in range(len(s) - 1)} or ({s} if s else set())

def sim(a, b):
    A, B = bigrams(a), bigrams(b)
    return len(A & B) / max(1, len(A | B))

# ── 1. 원문: 정규화 스트림 + 원 char offset 역사상 ──────────────────────────
doc = DOC.read_text(encoding="utf-8")
keep = [(i, ch.lower()) for i, ch in enumerate(doc) if re.match(r"[가-힣a-zA-Z0-9]", ch)]
nstream = "".join(ch for _, ch in keep)
npos = [i for i, _ in keep]

def find_all(sub):
    """정규화 스트림에서 sub 의 모든 출현 → [(c0,c1)] (v3 same_text_or 규칙: 동일 문구는 어느 출현이든 정답)."""
    out, j = [], nstream.find(sub)
    while j >= 0 and len(out) < 50:
        out.append((npos[j], npos[j + len(sub) - 1] + 1))
        j = nstream.find(sub, j + 1)
    return out

def locate(cit):
    """인용 정규화 텍스트의 모든 출현 [(c0,c1)] + method 반환. 실패 시 (None, None).
    v4 인용에 생략부호(…/...)가 섞인 경우 구간별로 나눠 각각 위치시키고 합집합(OR)으로 취급."""
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
    # 앞 60자 / 뒤 60자 부분 일치 (인용이 표 셀 병합 등으로 중간이 끊긴 경우)
    for part, tag in ((nq[:60], "head60"), (nq[-60:], "tail60")):
        if len(part) >= 20:
            occ = find_all(part)
            if occ:
                return occ, tag
    # 퍼지: 같은 길이 창을 성긴 보폭으로 스캔 후 최고 유사 창 정밀화
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

# ── 2. v4 xlsx 로드 ─────────────────────────────────────────────────────────
import openpyxl
rows = list(openpyxl.load_workbook(XLSX, read_only=True)[
    "정답셋359_v4"].iter_rows(values_only=True))[1:]
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

# ── 3. train60 순회 ─────────────────────────────────────────────────────────
gold = {g["qid"]: g for g in (json.loads(l) for l in open(FS / "out/gold_spans_lsh_train.jsonl", encoding="utf-8"))}
t60 = json.load(open(HERE / "out/train60_qids.json", encoding="utf-8"))

MERGE_GAP = 30  # v3 규칙과 동일: 인접 span 병합
out_gold, audit = [], []
for qid in t60:
    g = gold[qid]
    s, r = match_v4(g["q"])
    ent = {"qid": qid, "q": g["q"], "v4_sim": round(s, 2)}
    if s < 0.85:  # v4 미수록 → v3 유지 (제외는 별도 수동 감사 리스트만)
        ent.update(status="keep_v3", v4_no=None)
        out_gold.append(g)
        audit.append(ent)
        continue
    no, conf, srcs = r[0], r[4], str(r[6] or "")
    ent.update(v4_no=no, conf=conf)
    if conf == "불가":
        ent.update(status="exclude", reason="v4 신뢰도=불가(작성자 판정: 답 불가)")
        audit.append(ent)
        continue
    # 인용 파싱: [L####] 마커로 분할
    cits = [c.strip() for c in re.split(r"\[L\d+\]", srcs) if norm(c)]
    # v3 same_text_or 규칙: 문항 내 거의 동일한 인용(특약마다 반복되는 동일 조항)은 한 그룹(OR)
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
    if not groups:
        ent.update(status="exclude", reason="v4 인용 근거가 채점 코퍼스(본 약관 md)에 부재", missed=missed)
        audit.append(ent)
        continue
    # 채점 단위는 조(jo): 서로 다른 그룹이 같은 조를 공유하면 병합(그 조 하나로 둘 다 충족 → 분모 부풀림 방지)
    dedup = []
    for gr in groups:
        js = set(gr["jos"])
        for d in dedup:
            if js & set(d["jos"]):
                d["members"] = sorted({(m["c0"], m["c1"]) for m in d["members"] + gr["members"]})
                d["members"] = [{"c0": a, "c1": b, "src": "v4"} for a, b in d["members"]]
                d["jos"] = sorted(set(d["jos"]) | js)
                break
        else:
            dedup.append(gr)
    ng = {**g, "groups": dedup, "gold_src": "v4", "v4_no": no}
    groups = dedup
    out_gold.append(ng)
    ent.update(status="regold", n_cit=len(cits), n_located=len(cits) - len(missed),
               n_groups=len(groups), missed=missed)
    audit.append(ent)

with open(HERE / "out/gold_v4_train60.jsonl", "w", encoding="utf-8") as f:
    for g in out_gold:
        f.write(json.dumps(g, ensure_ascii=False) + "\n")
json.dump(audit, open(HERE / "out/regold_audit.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

cnt = collections.Counter(a["status"] for a in audit)
print("train60 처리:", dict(cnt))
for a in audit:
    if a["status"] == "exclude":
        print(" 제외:", a["qid"], a["reason"], a["q"][:40])
    elif a["status"] == "regold" and a.get("missed"):
        print(" 부분:", a["qid"], f"{a['n_located']}/{a['n_cit']} located", a["q"][:40])
