# -*- coding: utf-8 -*-
"""v8 파일럿 — build_sibling_groups.py + gen_meta_codex.py 시제품.
200청크만 돌려 게이트 G3-a~g / 지표 U1~U6 를 실측한다."""
import json, io, os, re, sys, time, subprocess, collections, unicodedata
from concurrent.futures import ThreadPoolExecutor

OUT = r"c:/Users/Atdev-pc/Desktop/wiki/experiments/tos-skeleton/hybrid-enrich/metajson-v6/meta-search-v4/out"
HERE = os.path.dirname(os.path.abspath(__file__))
CODEX = os.path.join(os.environ["APPDATA"], "npm", "codex.cmd")

# --- CLI: pilot_v8.py <chunkset> <tag> [--fix] [--model M] ---
CS    = sys.argv[1] if len(sys.argv) > 1 else "fixed600"
TAG   = sys.argv[2] if len(sys.argv) > 2 else CS
FIX   = "--fix" in sys.argv                      # C안: 수정본 적용
MODEL = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "gpt-5.4-mini"
WORK  = os.path.join(HERE, "pilot_" + TAG); os.makedirs(WORK, exist_ok=True)
SLOT, SHARD, SPLIT_TH = 20, 16, 50
WORKERS = int(os.environ.get("V8_WORKERS", "8"))

# --- C안 수정 ①: 잔재/저내용 청크 사전 필터 (생성 전에 뺀다) ---
DOCNO_RX  = re.compile(r"\b\d{15,}\b")                  # 문서관리번호
PAGEFOOT  = re.compile(r"SHINHAN LIFE|^\s*-{3,}\s*$", re.M)
ENGRUN_RX = re.compile(r"[A-Za-z][A-Za-z ,.'\"()]{40,}")  # 영어 파싱 잔재
LATEX_RX  = re.compile(r"\$\$|\\frac|\\left|\\right|[a-z]\}\s*\$\$")
MIDSENT_RX = re.compile(r"^[가-힣]{1,2}[\s]")            # 문장 중간 시작 (조사 파편)


TOC_RX = re.compile(r"약\s?관\s?목\s?차|\[특약 약관\] 제1편 일반사항")
LATEX_SPAN = re.compile(r"\$[^$]{0,400}\$|\$\$[^$]{0,400}\$\$|\\(?:text|frac|left|right|sum|square|underline)\b[^\n]*")


def clean_text(t):
    """진단(diag4) 반영: 영문·수식 잔재는 **버리지 말고 잘라낸다.**
    이 조각들의 뒤쪽에는 진짜 면책조항 본문이 붙어 있다 (실측 6/13건)."""
    t = LATEX_SPAN.sub(" ", t)
    t = ENGRUN_RX.sub(" ", t)
    t = DOCNO_RX.sub(" ", t)
    t = re.sub(r"^\s*-{3,}\s*$", " ", t, flags=re.M)
    t = re.sub(r"\bSHINHAN LIFE\b", " ", t)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", t)).strip()


def is_debris(t):
    """C안: 정제 **후에도** 색인 대상이 아닌 조각인가."""
    c = clean_text(t)
    core = re.sub(r"[\s|\-:·．.0-9]", "", c)
    if len(core) < 60: return "저내용"
    if TOC_RX.search(c) and len(re.sub(r"[\s|\-:·．.0-9]", "", TOC_RX.sub("", c))) < 150:
        return "목차잔재"
    return ""

BOILER_RX = re.compile(r"\((?:간편|무배당[^)]*|갱신형|해약환급금[^)]*|중도부가용)\)")
TITLE_RX  = re.compile(r"\(([^)]*)\)\s*$")
# G3-d2 axis 금칙어
AXIS_BAN = ("본문", "청크", "조항 내용", "주제", "내용", "상태", "성격", "종류", "구분")
AXIS_BAN_OK = ("장해분류", "질병분류", "질병 분류", "장해 분류")
CAP = {"scope": 20, "axis": 14, "mark": 16, "ask": 28, "only": 40, "twin": 10}


def norm(s):
    return unicodedata.normalize("NFC", (s or "")).strip()


def squash(s):
    return re.sub(r"\s+", "", norm(s))


# ---------- 1. 형제 묶음 ----------
SUBRIDER_RX = re.compile(r"\[([^\]\n]{2,20}형)\]|\[(간편심사형|일반심사형)\]")


def sub_rider(text):
    """C안 수정 ③: 같은 셀 안의 하위 특약/심사형 구분을 그룹 키에 반영."""
    m = SUBRIDER_RX.search(text[:400])
    return norm(m.group(1) or m.group(2)) if m else ""


def build_groups():
    ch, st = {}, {}
    for line in io.open(f"{OUT}/chunks_{CS}.jsonl", encoding="utf-8"):
        r = json.loads(line); ch[r["chunk_id"]] = r
    for line in io.open(f"{OUT}/chunk_structure_{CS}.jsonl", encoding="utf-8"):
        r = json.loads(line); st[r["chunk_id"]] = r

    dropped = collections.Counter()
    if FIX:
        for cid in list(ch):
            d = is_debris(ch[cid]["text"])
            if d: dropped[d] += 1; ch.pop(cid); st.pop(cid, None)
            else: ch[cid]["text"] = clean_text(ch[cid]["text"])   # 정제본으로 교체

    cell = collections.defaultdict(list)
    for cid, s in st.items():
        key = (norm(s.get("rider", "")), norm(s.get("article", "")))
        if FIX and "--nosub" not in sys.argv: key = key + (sub_rider(ch[cid]["text"]),)
        cell[key].append(cid)

    groups = collections.defaultdict(list)
    for k, ids in cell.items():
        if len(ids) >= 2:
            groups[("cell",) + k].extend(ids)
        else:                                    # 외톨이 → 조제목 전역으로 승격
            m = TITLE_RX.search(k[1])
            t = norm(m.group(1)) if m else (k[1] or "__none__")
            groups[("title", t)].extend(ids)
    if dropped: print(f"  [FIX] 사전 필터 제거: {dict(dropped)} (총 {sum(dropped.values())})")
    return ch, st, dict(groups)


def shard(ids, ch):
    """크기 > SLOT 인 그룹을 본문 유사도 순 정렬 후 SHARD 개씩 자른다."""
    if len(ids) <= SLOT:
        return [ids]
    ids = sorted(ids, key=lambda c: ch[c]["text"][:200])   # 유사 본문이 인접하게
    return [ids[i:i + SHARD] for i in range(0, len(ids), SHARD)]


# ---------- 2. 프롬프트 ----------
SYSTEM = io.open(os.path.join(HERE, "SYSTEM_V8.txt"), encoding="utf-8").read()
SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object",
          "properties": {"i": {"type": "integer"}, "scope": {"type": "string"}, "axis": {"type": "string"},
                         "mark": {"type": "array", "items": {"type": "string"}},
                         "ask": {"type": "string"}, "only": {"type": "string"}, "twin": {"type": "string"}},
          "required": ["i", "scope", "axis", "mark", "ask", "only", "twin"],
          "additionalProperties": False}}}, "required": ["items"], "additionalProperties": False}


def build_prompt(segs, ch, st):
    """segs: [(gkey, [chunk_id,...]), ...]  — 작은 묶음 여러 개를 한 호출에 패킹한다.
    ⚠ 비교는 **묶음 안에서만** 이뤄져야 하므로 경계를 명시한다."""
    parts, i, n = [], 0, sum(len(ids) for _, ids in segs)
    for gi, (gkey, ids) in enumerate(segs):
        s0 = st[ids[0]]
        parts.append(f"### 묶음 {chr(65+gi)} — 특약={BOILER_RX.sub('', s0.get('rider',''))} "
                     f"/ 조항={s0.get('article','')} (조각 {len(ids)}개)")
        for c in ids:
            parts.append(f"=== [{i}] ===\n" + ch[c]["text"][:600]); i += 1
    multi = ("\n\n⚠ 아래에는 **서로 다른 묶음이 여러 개** 들어 있다. `###` 로 구분된다.\n"
             "   비교와 구별은 **같은 묶음 안에서만** 해라. 다른 묶음의 조각과는 비교하지 마라.\n"
             "   axis 는 **묶음마다 따로** 정한다 (같은 묶음 안에서만 공유).\n"
             if len(segs) > 1 else "\n")
    tail = (f"\n\n--- 출력 형식 ---\n조각 {n}개 각각에 대해 6개 필드를 만들어라. "
            f"i 는 0부터 {n-1}까지 순서대로, 정확히 {n}개.")
    return SYSTEM + "\n\n--- 형제 청크 묶음 ---" + multi + "\n" + "\n\n".join(parts) + tail


def pack(groups, ch, small_lim=6):
    """작은 묶음(≤small_lim)은 SLOT까지 합쳐 1호출, 큰 묶음은 단독(필요 시 샤딩)."""
    tasks, buf, bufn = [], [], 0
    for gkey, ids in sorted(groups.items(), key=lambda x: (len(x[1]), str(x[0]))):
        if len(ids) > small_lim:
            for sh in shard(ids, ch):
                if sh: tasks.append([(gkey, sh)])
        else:
            if bufn + len(ids) > SLOT and buf:
                tasks.append(buf); buf, bufn = [], 0
            buf.append((gkey, ids)); bufn += len(ids)
    if buf: tasks.append(buf)
    return tasks


def call(k, segs, ch, st):
    p = os.path.join(WORK, f"p{k}.txt"); o = os.path.join(WORK, f"o{k}.json")
    io.open(p, "w", encoding="utf-8").write(build_prompt(segs, ch, st))
    cmd = [CODEX, "exec", "--model", MODEL, "-c", "model_reasoning_effort=low",
           "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
           "--output-schema", os.path.join(WORK, "schema.json"), "-o", o, "-"]
    t0 = time.time()
    try:
        with io.open(p, "rb") as fh:
            r = subprocess.run(cmd, stdin=fh, capture_output=True, timeout=900)
        out = r.stdout.decode("utf-8", "replace")
    except Exception as e:
        print(f"  [{k}] CALL FAIL {e}", flush=True); return k, segs, [], time.time() - t0, 0
    dt = time.time() - t0
    tok = 0
    m = re.search(r"tokens used\s*\n\s*([\d,]+)", out)
    if m: tok = int(m.group(1).replace(",", ""))
    try:
        items = json.load(io.open(o, encoding="utf-8"))["items"]
    except Exception as e:
        print(f"  [{k}] PARSE FAIL {e}", flush=True); return k, segs, [], dt, tok
    return k, segs, items, dt, tok


# ---------- 3. sanitize_v8 ----------
def sanitize(recs, bodies):
    """recs: [{chunk_id, group, scope, axis, mark, ask, only, twin}]"""
    stats = collections.Counter()
    for r in recs:                                            # S1 길이
        for f in ("scope", "axis", "ask", "only", "twin"):
            v = norm(r.get(f, ""))
            if len(v) > CAP[f]: v = ""; stats[f"cap_{f}"] += 1
            r[f] = v
        mk = [norm(m) for m in (r.get("mark") or [])][:3]
        mk = [m for m in mk if m and len(m) <= CAP["mark"]]
        r["mark"] = mk
        if r["twin"]:                                         # S2 twin → 비움
            r["mark"], r["only"] = [], ""; stats["twin_cleared"] += 1
        # S3 접지(grounding) — 진단(diag3) 결과 반영
        #   ① 자기 본문에 그대로       → 통과
        #   ③ 토큰 전부가 자기 본문에   → 통과 (띄어쓰기·괄호 차이일 뿐. 종전 검사는 과잉 엄격)
        #   ②④⑤ 형제 본문에만 / 일부만 / 전무 → 폐기 (오귀속·조작. 널 실측 .254의 원인)
        b = squash(bodies[r["chunk_id"]])
        keep = []
        for m in r["mark"]:
            if squash(m) in b: keep.append(m); continue
            toks = [t for t in re.split(r"[\s,·/()\[\]\"「」]+", m) if len(t) >= 2]
            if toks and all(squash(t) in b for t in toks):
                keep.append(m); stats["grounded_by_token"] += 1; continue
            stats["ungrounded_dropped"] += 1
        stats["mark_total"] += len(r["mark"])
        r["mark"] = keep
    by_g = collections.defaultdict(list)
    for r in recs: by_g[r["group"]].append(r)
    if FIX:                                                   # C안 수정 ②: axis 금칙어 강제 폐기
        for g, rs in by_g.items():
            ok = collections.Counter(r["axis"] for r in rs if r["axis"] and not axis_banned(r["axis"]))
            fallback = ok.most_common(1)[0][0] if ok else ""
            for r in rs:
                if r["axis"] and axis_banned(r["axis"]):
                    r["axis"] = fallback; stats["axis_replaced"] += 1
    for g, rs in by_g.items():                                # S5 형제 전원 공유값 폐기
        act = [r for r in rs if not r["twin"]]
        if len(act) < 2: continue
        cnt = collections.Counter()
        for r in act:
            for m in set(r["mark"]): cnt[squash(m)] += 1
        shared = {m for m, c in cnt.items() if c == len(act)}
        if shared:
            for r in act:
                n0 = len(r["mark"])
                r["mark"] = [m for m in r["mark"] if squash(m) not in shared]
                stats["shared_dropped"] += n0 - len(r["mark"])
    if FIX:                                                   # S6: 그룹내 중복 only/ask 폐기
        for g, rs in by_g.items():                            #   같은 값이 둘 이상이면 변별력 0
            act = [r for r in rs if not r["twin"]]
            for f in ("only", "ask"):
                c = collections.Counter(squash(r[f]) for r in act if r[f])
                dup = {v for v, n in c.items() if n > 1}
                for r in act:
                    if r[f] and squash(r[f]) in dup:
                        r[f] = ""; stats[f"dup_{f}_dropped"] += 1
    return stats


# ---------- 4. 게이트/지표 ----------
def axis_banned(a):
    a = squash(a)
    if any(ok in a for ok in (squash(x) for x in AXIS_BAN_OK)): return False
    return any(squash(b) in a for b in AXIS_BAN)


def gates(recs, bodies, stats):
    by_g = collections.defaultdict(list)
    for r in recs: by_g[r["group"]].append(r)
    n = len(recs)
    g3a = g3b = 0; align = []; ban = 0
    for g, rs in by_g.items():
        act = [r for r in rs if not r["twin"]]
        if len(act) >= 2:
            ms = [frozenset(squash(m) for m in r["mark"]) for r in act]
            if len(set(ms)) == 1 and ms[0]: g3a += 1
            seen = collections.Counter()
            for r in act:
                for f in ("ask", "only"):
                    if r[f]: seen[squash(r[f])] += 1
            g3b += sum(1 for v in seen.values() if v > 1)
        ax = collections.Counter(r["axis"] for r in rs if r["axis"])
        if ax: align.append(ax.most_common(1)[0][1] / len(rs))
    ban = sum(1 for r in recs if r["axis"] and axis_banned(r["axis"]))
    # U3 정정: 5%·1년 미만·14대질병·간편심사형·부표2-1 같은 실제 판별값을 포함하도록 확장
    SIG = re.compile(r"[A-Z]\d{2}|부표\s?\d|별표\s?\d|급여금|보험금|진단금|「|」|"
                     r"\d+\s?%|\d+\s?[년일월회세]|\d+대|심사형|개시일|분류표|제\d+-?\d*조|"
                     r"특약|보장|증명서|입원|수술|진단|해약환급금|적립액")
    act_all = [r for r in recs if not r["twin"]]              # 색인 대상만
    u3 = sum(1 for r in act_all if any(SIG.search(m) for m in r["mark"]))
    # 사후 잔존 미접지 (S3 이후 남은 것) — 구조적으로 0이어야 한다
    resid = sum(1 for r in recs for m in r["mark"] if squash(m) not in squash(bodies[r["chunk_id"]])
                and not all(squash(t) in squash(bodies[r["chunk_id"]])
                            for t in re.split(r"[\s,·/()\[\]\"「」]+", m) if len(t) >= 2))
    llm = [squash(r["axis"] + "|" + "".join(r["mark"]) + "|" + r["ask"] + "|" + r["only"]) for r in act_all]
    c = collections.Counter(llm) or collections.Counter([""])
    only_v = [squash(r["only"]) for r in recs if r["only"]]
    return {
        "n": n, "색인대상 n": len(act_all),
        "G3-a 전원동일 mark 그룹": g3a,
        "G3-b 그룹내 ask/only 중복쌍": g3b,
        "G3-c 잔존 미접지 mark(하드 ==0)": resid,
        "G3-c2 mark 빈 색인대상 비율(하드 ≤.30)": round(
            sum(1 for r in act_all if not r["mark"]) / max(1, len(act_all)), 4),
        "Q1 LLM 원출력 미접지율(기록)": round(
            stats["ungrounded_dropped"] / max(1, stats["mark_total"]), 4),
        "Q2 토큰접지로 구제(기록)": stats.get("grounded_by_token", 0),
        "G3-d 축 정렬률(평균)": round(sum(align) / max(1, len(align)), 3),
        "G3-d2 axis 금칙어 비율": round(ban / max(1, n), 4),
        # G3-f 분리: '잔재'(전처리 실패, 게이트 대상)와 '본문동일/표동일'(문서 특성, 기록)
        "G3-f 잔재 선언율(하드 ≤.05)": round(
            sum(1 for r in recs if r["twin"] == "잔재") / max(1, n), 4),
        "Q3 진짜동일 선언율(기록)": round(
            sum(1 for r in recs if r["twin"] in ("본문동일", "표동일")) / max(1, n), 4),
        "G3-f twin 선언율": round(sum(1 for r in recs if r["twin"]) / n, 4),
        "U2 mark 비어있음 비율": round(sum(1 for r in recs if not r["mark"]) / n, 4),
        "U3 판별신호 포함률": round(u3 / max(1, len(act_all)), 4),
        "U4 unique_ratio(색인대상)": round(len(c) / max(1, len(act_all)), 4),
        "U4 최대충돌": c.most_common(1)[0][1],
        "U5 only distinct": len(set(only_v)),
        "공유값 폐기수": stats["shared_dropped"],
        "중복 only 폐기": stats.get("dup_only_dropped", 0),
        "중복 ask 폐기": stats.get("dup_ask_dropped", 0),
        "길이초과 폐기": sum(v for k, v in stats.items() if k.startswith("cap_")),
    }


# ---------- main ----------
def main():
    lock = os.path.join(WORK, ".lock")           # 중복 실행 방지 (같은 파일에 두 프로세스가 쓰면 산출물이 오염된다)
    if os.path.exists(lock):
        sys.exit(f"[중단] 이미 실행 중으로 보인다: {lock}  (아니면 이 파일을 지워라)")
    io.open(lock, "w").write(str(os.getpid()))
    import atexit; atexit.register(lambda: os.path.exists(lock) and os.remove(lock))
    io.open(os.path.join(WORK, "schema.json"), "w", encoding="utf-8").write(
        json.dumps(SCHEMA, ensure_ascii=False))
    ch, st, groups = build_groups()
    sizes = collections.Counter(len(v) for v in groups.values())
    print(f"그룹 {len(groups)}개 / 외톨이 {sizes[1]} / 청크 {sum(len(v) for v in groups.values())}")

    if "--all" in sys.argv:                      # 전량 모드
        picked = sorted(groups.items(), key=lambda x: str(x[0]))
        total = sum(len(v) for _, v in picked)
        print(f"전량 그룹 {len(picked)}개 / 청크 {total}")
    else:
      # 층화 표집: 작은 그룹~큰 그룹을 고루 (샤딩 경로도 태운다)
      strata = [(2, 3, 4), (4, 9, 5), (10, 19, 3), (20, 49, 2), (50, 10**9, 1)]
      picked, total = [], 0
      for lo, hi, want in strata:
          cand = sorted([(k, v) for k, v in groups.items() if lo <= len(v) <= hi],
                        key=lambda x: (len(x[1]), str(x[0])))
          step = max(1, len(cand) // max(1, want))
          for k, v in cand[::step][:want]:
              v = v[:45]
              picked.append((k, v)); total += len(v)
      print(f"파일럿 그룹 {len(picked)}개 / 청크 {total}")

    done = set()
    outp = os.path.join(WORK, "meta_v8.jsonl")
    if os.path.exists(outp):
        for l in io.open(outp, encoding="utf-8"):
            try: done.add(json.loads(l)["chunk_id"])
            except: pass
        print(f"  재개: 이미 완료 {len(done)}청크")
    remain = {k: [c for c in ids if c not in done] for k, ids in picked}
    remain = {k: v for k, v in remain.items() if v}
    tasks = pack(remain, ch)                       # 작은 묶음 패킹 + 큰 묶음 샤딩
    nseg = sum(len(t) for t in tasks)
    print(f"호출 {len(tasks)}회 (묶음 {nseg}개 · 패킹+샤딩 후) / 청크 "
          f"{sum(len(i) for t in tasks for _, i in t)}", flush=True)

    def to_recs(segs, items):
        bi = {it["i"]: it for it in items}
        out, i = [], 0
        for gk, ids in segs:
            for cid in ids:
                it = bi.get(i); i += 1
                if not it: continue
                out.append({"chunk_id": cid, "group": str(gk), "scope": it["scope"], "axis": it["axis"],
                            "mark": it["mark"], "ask": it["ask"], "only": it["only"], "twin": it["twin"]})
        return out

    t0 = time.time(); recs, fails, toks, ndone = [], 0, 0, 0
    fh_out = io.open(outp, "a", encoding="utf-8")             # 완료분 즉시 append → 재개 가능
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(call, i, segs, ch, st) for i, segs in enumerate(tasks)]
        for f in futs:
            k, segs, items, dt, tok = f.result()
            toks += tok; ndone += 1
            if len(items) != sum(len(i) for _, i in segs): fails += 1
            rs = to_recs(segs, items)
            recs.extend(rs)
            for r in rs: fh_out.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh_out.flush()
            if ndone % 20 == 0:
                el = time.time() - t0
                print(f"  {ndone}/{len(tasks)} 호출 · {el/60:.1f}분 · 잔여 약 "
                      f"{el/ndone*(len(tasks)-ndone)/60:.0f}분 · 실패 {fails}", flush=True)
    fh_out.close()
    wall = time.time() - t0
    # 재개분(이전 실행에서 저장된 것)까지 합쳐서 게이트 계산
    if True:
        seen = set()
        recs = [r for r in recs if not (r["chunk_id"] in seen or seen.add(r["chunk_id"]))]
        for l in io.open(outp, encoding="utf-8"):
            try: r = json.loads(l)
            except: continue
            if r["chunk_id"] not in seen: seen.add(r["chunk_id"]); recs.append(r)
    # C안 수정 ④: 환각 검사를 '잘려나간 이웃'까지 확장
    #   fixed600은 overlap 100 으로 문장 중간을 자른다. 신호가 옆 청크에 남아 있을 수 있다.
    order = sorted(ch, key=lambda c: (ch[c].get("doc_id", ""), ch[c].get("char_start", 0)))
    pos = {c: i for i, c in enumerate(order)}
    bodies = {}
    for c in ch:
        i = pos[c]; t = ch[c]["text"]
        if FIX:
            if i > 0:   t = ch[order[i - 1]]["text"][-300:] + t
            if i + 1 < len(order): t = t + ch[order[i + 1]]["text"][:300]
        bodies[c] = t

    stats = sanitize(recs, bodies)
    rep = gates(recs, bodies, stats)
    rep.update({"청크셋": CS, "모델": MODEL, "수정본(FIX)": FIX,
                "호출수": len(tasks), "개수불일치 배치": fails, "총토큰": toks,
                "axis 대체": stats.get("axis_replaced", 0),
                "벽시계초": round(wall, 1)})
    io.open(os.path.join(WORK, "pilot_report.json"), "w", encoding="utf-8").write(
        json.dumps(rep, ensure_ascii=False, indent=2))
    io.open(os.path.join(WORK, "pilot_recs.jsonl"), "w", encoding="utf-8").write(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in recs))
    print(f"### {TAG} ###")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
