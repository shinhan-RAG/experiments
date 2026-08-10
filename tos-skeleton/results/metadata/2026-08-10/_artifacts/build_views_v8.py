# -*- coding: utf-8 -*-
"""v8 뷰 조립 — build_views.py 를 **건드리지 않고** 같은 포맷으로 직접 쓴다.

건드리지 않는 이유: 08-07/08-09 등록본이 그 파일의 arm 정의로 산출됐다.
출력: out/view_fixed600_{ARM}.jsonl  = {chunk_id, doc_id, text}
      BUDGET=240, prefix 배치 — 08-07/08-09와 동일 조건이라 수치를 이어 읽을 수 있다.
"""
import json, io, os, re, sys, random, hashlib, collections

OUT = r"c:/Users/Atdev-pc/Desktop/wiki/experiments/tos-skeleton/hybrid-enrich/metajson-v6/meta-search-v4/out"
HERE = os.path.dirname(os.path.abspath(__file__))
META = os.path.join(HERE, "pilot_FULL", "meta_v8.jsonl")
CS = "fixed600"
BUDGET = 240
BOILER_RX = re.compile(r"\((?:간편|무배당[^)]*|갱신형|해약환급금[^)]*|중도부가용)\)")


def clean_rider(s):
    return BOILER_RX.sub("", s or "").strip()


def _join(parts, budget):
    """build_views._join 과 동일 — 앞에서부터 예산까지 채우고 절단 여부를 돌려준다."""
    out, used = [], 0
    for p in parts:
        if not p: continue
        if budget is not None and used + len(p) + 1 > budget:
            return " ".join(out), True
        out.append(p); used += len(p) + 1
    return " ".join(out), False


# ---------------- arm 정의 (설계안 §6-1) ----------------
def v8_block(m, with_scope=True):
    p = []
    if with_scope and m.get("scope"): p.append(f"[소속] {m['scope'][:20]}")
    if m.get("axis"):                 p.append(f"[축] {m['axis'][:14]}")
    mk = [x for x in (m.get("mark") or []) if x][:3]
    if mk:                            p.append("[표지] " + " · ".join(x[:16] for x in mk))
    if m.get("ask"):                  p.append(f"[질문] {m['ask'][:28]}")
    if m.get("only"):                 p.append(f"[차이] {m['only'][:40]}")
    return p


def v8_contrast(c, st, m): return v8_block(m, True)


def v8_plus(c, st, m):
    p = []
    r = clean_rider(st.get("rider", ""))
    if r:                    p.append(f"[특약] {r[:40]}")
    if st.get("article"):    p.append(f"[조항] {st['article'][:40]}")
    return p + v8_block(m, False)          # [소속] 제외 — 정규식 주소가 그 자리를 대신한다


def v8_markonly(c, st, m):
    p = []
    if m.get("scope"): p.append(f"[소속] {m['scope'][:20]}")
    mk = [x for x in (m.get("mark") or []) if x][:3]
    if mk:             p.append("[표지] " + " · ".join(x[:16] for x in mk))
    return p


ARMS = {"V8_CONTRAST": v8_contrast, "V8_PLUS": v8_plus, "V8_MARKONLY": v8_markonly}


def main():
    ch, st = {}, {}
    for l in io.open(f"{OUT}/chunks_{CS}.jsonl", encoding="utf-8"):
        r = json.loads(l); ch[r["chunk_id"]] = r
    for l in io.open(f"{OUT}/chunk_structure_{CS}.jsonl", encoding="utf-8"):
        r = json.loads(l); st[r["chunk_id"]] = r
    meta, seen = {}, set()
    for l in io.open(META, encoding="utf-8"):
        r = json.loads(l)
        if r["chunk_id"] in seen: continue
        seen.add(r["chunk_id"]); meta[r["chunk_id"]] = r
    print(f"청크 {len(ch)} / 구조 {len(st)} / v8 메타 {len(meta)}")

    order = sorted(ch)
    stats = {}
    strings = {}
    for arm, fn in ARMS.items():
        rows, trunc, lens = [], 0, []
        for cid in order:
            m = meta.get(cid, {})
            parts = fn(ch[cid], st.get(cid, {}), m) if m else []
            s, cut = _join(parts, BUDGET)
            trunc += cut; lens.append(len(s))
            if arm == "V8_CONTRAST": strings[cid] = s
            rows.append({"chunk_id": cid, "doc_id": ch[cid]["doc_id"],
                         "text": (s + "\n" + ch[cid]["text"]) if s else ch[cid]["text"]})
        p = f"{OUT}/view_{CS}_{arm}.jsonl"
        io.open(p, "w", encoding="utf-8").write(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        stats[arm] = {"n": len(rows), "메타있음": sum(1 for c in order if meta.get(c)),
                      "G1 절단률": round(trunc / len(rows), 4),
                      "최대길이": max(lens), "평균길이": round(sum(lens) / len(lens), 1)}
        print(f"  {arm:14s} {stats[arm]}")

    # 널 arm — V8_CONTRAST 문자열 기준. ⚠ v7 널 파일을 덮어쓰지 않는 새 이름
    for name, mk in (("N1_DUMMY_V8", "dummy"),):
        rows = [{"chunk_id": c, "doc_id": ch[c]["doc_id"],
                 "text": ("〇" * len(strings[c]) + "\n" + ch[c]["text"]) if strings[c] else ch[c]["text"]}
                for c in order]
        io.open(f"{OUT}/view_{CS}_{name}.jsonl", "w", encoding="utf-8").write(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        print(f"  {name:14s} 길이일치 널")
    for seed in (20260812, 20260813):
        rnd = random.Random(seed)
        src = list(order); shuf = src[:]; rnd.shuffle(shuf)
        rows = [{"chunk_id": c, "doc_id": ch[c]["doc_id"],
                 "text": (strings[s] + "\n" + ch[c]["text"]) if strings[s] else ch[c]["text"]}
                for c, s in zip(order, shuf)]
        nm = f"N2_SHUF_V8{seed}"
        io.open(f"{OUT}/view_{CS}_{nm}.jsonl", "w", encoding="utf-8").write(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        print(f"  {nm:14s} 주소 오배정 널")

    io.open(f"{OUT}/view_stats_v8.json", "w", encoding="utf-8").write(
        json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
