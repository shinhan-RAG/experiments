#!/usr/bin/env python3
"""Semantic tag 구조·필드 순차 실험 — 전 구간 결정적 (PLAN.md 사전 등록).

usage:
  python3 semtag_experiment.py --doc <260507.md> --qa-xlsx <정답셋.xlsx> \
      --dev-qids <qa100_gold.jsonl or qid txt> --out out/
"""
import argparse
import hashlib
import json
import math
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from random import Random
from xml.etree import ElementTree as ET

M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
nfc = lambda s: unicodedata.normalize("NFC", s)
ns = lambda s: re.sub(r"\s+", "", s)

# ── BM25 (문자 bigram, 검증된 공식과 동일) ─────────────────────────────

def bigrams(s):
    s = ns(s)
    return [s[i:i + 2] for i in range(len(s) - 1)]


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.tf, self.dl, self.df = [], [], Counter()
        for d in docs:
            c = Counter(bigrams(d))
            self.tf.append(c)
            self.dl.append(sum(c.values()))
            for t in c:
                self.df[t] += 1
        self.avgdl = sum(self.dl) / max(1, self.N)

    def rank10(self, query):
        q = list(Counter(bigrams(query)))
        scores = []
        for i in range(self.N):
            s = 0.0
            tfi = self.tf[i]
            for t in q:
                f = tfi.get(t, 0)
                if not f:
                    continue
                idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
                B = 1 - self.b + self.b * (self.dl[i] / self.avgdl)
                s += idf * f * (self.k1 + 1) / (f + self.k1 * B)
            scores.append(s)
        order = sorted(range(self.N), key=lambda i: -scores[i])
        return order[:10]


# ── 1. Element 분할 (결정적) ───────────────────────────────────────────

JO = re.compile(r"^#{0,4}\s*제\s?\d+(?:-\d+)?\s?조(?:의\s?\d+)?\s*[\(（【]")
HEAD = re.compile(r"^#{1,4}\s+\S")
FORMULA = re.compile(r"\$\$|\\frac|\\sum|\\left|\\begin")
MAX_LINES = 120


def split_elements(lines):
    """경계: 헤딩/조 제목/표 블록 전환. 유형: table/formula/heading/text."""
    els, cur, start = [], [], 1

    def flush(end):
        nonlocal cur, start
        if not cur or not any(l.strip() for l in cur):
            cur = []
            return
        text = "\n".join(cur)
        n_table = sum(1 for l in cur if l.lstrip().startswith("|"))
        if n_table / max(1, len(cur)) > 0.5:
            et = "table"
        elif FORMULA.search(text):
            et = "formula"
        elif len(cur) <= 2 and HEAD.match(cur[0]):
            et = "heading"
        else:
            et = "text"
        for off in range(0, len(cur), MAX_LINES):
            part = cur[off:off + MAX_LINES]
            els.append({"lines": part, "start": start + off,
                        "end": start + off + len(part) - 1, "type": et})
        cur = []

    in_table = False
    for i, line in enumerate(lines, 1):
        is_tbl = line.lstrip().startswith("|")
        boundary = JO.match(line) or HEAD.match(line) or (is_tbl != in_table)
        if boundary and cur:
            flush(i - 1)
            start = i
        cur.append(line)
        in_table = is_tbl
    flush(len(lines))
    for j, e in enumerate(els):
        e["eid"] = f"e{j:05d}"
        e["text"] = "\n".join(e.pop("lines"))
    return els


# ── 2. 구조 컨텍스트 (특약/편·조 breadcrumb — 원문 스캔) ────────────────

RIDER = re.compile(r"^#{0,4}\s*(?:\[?\d*\]?\s*)?((?:\(무\)|\(간편\)|\[?[0-9]{0,3}\]?)?[가-힣A-Za-z0-9（）()\[\]·%\s]{2,45}특약)\s*(?:약관)?\s*$")
PYEON = re.compile(r"^#{0,4}\s*(제\s?\d+\s?편[^\n]{0,30})")
JO_T = re.compile(r"^#{0,4}\s*(제\s?\d+(?:-\d+)?\s?조(?:의\s?\d+)?)\s*[\(（【]\s*([^\)）】]{1,40})")


def annotate_context(els):
    scope, pyeon, jo = "주계약", "", ""
    for e in els:
        first = e["text"].split("\n", 1)[0]
        m = RIDER.match(first)
        if m and "특약" in m.group(1):
            scope = re.sub(r"\s+", " ", m.group(1)).strip()[:30]
        m = PYEON.match(first)
        if m:
            pyeon = ns(m.group(1))[:14]
        m = JO_T.match(first)
        if m:
            jo = (ns(m.group(1)) + "(" + m.group(2).strip()[:20] + ")")
        e["scope"], e["pyeon"], e["jo"] = scope, pyeon, jo
    return els


# ── 3. 태그 빌더 (schema_version별, 규칙만·원문만) ──────────────────────

ROLE_VOCAB = ["지급사유", "정의", "납입면제사유", "보장개시일", "면책", "부담보",
              "감액", "무효", "진단확정기준", "제출서류", "청구절차", "갱신",
              "해지환급금", "해지", "보험금변경", "특약구성"]
ROLE_HINT = {"지급사유": ["지급사유", "지급합니다", "지급률"],
             "정의": ["정의", "이라 함은", "란 함은", "말합니다"],
             "납입면제사유": ["납입면제", "납입을 면제"],
             "보장개시일": ["보장개시일"], "면책": ["면책"], "부담보": ["부담보"],
             "감액": ["감액"], "무효": ["무효"],
             "진단확정기준": ["진단확정"], "제출서류": ["구비서류", "제출", "증명서"],
             "청구절차": ["청구"], "갱신": ["갱신"],
             "해지환급금": ["해지환급금", "해약환급금"], "해지": ["해지"],
             "보험금변경": ["보험금 받는 방법"], "특약구성": ["부가할 수", "준용"]}
CODE = re.compile(r"(?<![A-Za-z0-9])([A-Z]\d{2}(?:\.\d{1,2})?(?:\s?[-~]\s?[A-Z]?\d{2}(?:\.\d{1,2})?)?)(?![A-Za-z])")
BYULPYO = re.compile(r"(별표\s?\d+|부표\s?\d+|분류표)")
DEFTERM = re.compile(r"[「\"']([가-힣A-Za-z0-9·\s]{2,16})[」\"']\s*(?:이?라\s*함은|이란)")
JONUM = re.compile(r"(제\s?\d+(?:-\d+)?\s?조(?:의\s?\d+)?)")


def roles_of(text):
    found = []
    for r in ROLE_VOCAB:
        if any(h in text for h in ROLE_HINT[r]):
            found.append(r)
        if len(found) == 3:
            break
    return found


def exact_keys_of(e):
    keys = []
    if e["jo"]:
        keys.append(e["jo"].split("(")[0])
    head = e["text"][:2000]
    keys += list(dict.fromkeys(BYULPYO.findall(head)))[:2]
    codes = list(dict.fromkeys(CODE.findall(head)))
    keys += codes[:2]
    return keys[:5]


def defterms_of(text):
    return list(dict.fromkeys(m.strip() for m in DEFTERM.findall(text[:3000])))[:3]


def table_extras(e):
    if e["type"] != "table":
        return ""
    codes = list(dict.fromkeys(CODE.findall(e["text"])))
    rng = f"{codes[0]}~{codes[-1]}" if len(codes) > 2 else ";".join(codes)
    first = next((l for l in e["text"].split("\n") if l.strip()), "")[:30]
    return ";".join(x for x in (rng, ns(first)[:20]) if x)


def serialize(fields, cap=200):
    s = " ".join(f"[{k}]{v}" for k, v in fields if v)
    return s[:cap]


def build_tag(e, schema):
    f = []
    if schema == "A0":
        return ""
    f.append(("유형", e["type"]))
    f.append(("특약", e["scope"]))
    if schema == "tag.s0":
        f.append(("조항", e["jo"]))
        role = roles_of(e["text"][:1500])
        f.append(("역할", role[0] if role else ""))
        return serialize(f)
    f.append(("경로", ">".join(x for x in (e["pyeon"], e["jo"]) if x)))
    if schema >= "tag.s1k1":
        f.append(("역할", ";".join(roles_of(e["text"][:1500]))))
    if schema >= "tag.s1k2":
        f.append(("키", ";".join(exact_keys_of(e))))
    if schema >= "tag.s1k3":
        f.append(("정의어", ";".join(defterms_of(e["text"]))))
    if schema >= "tag.s1k4":
        f.append(("표", table_extras(e)))
    if schema.endswith("r2"):
        f = [(k, v + (" " + ns(v) if k in ("키", "경로") and ns(v) != v else ""))
             for k, v in f]
    return serialize(f)


# ── 4. QA 파싱 + gold 매핑 (quote-first) ───────────────────────────────

def parse_xlsx(path):
    z = zipfile.ZipFile(path)
    sh = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for r in sh.findall(".//" + M + "row"):
        c = {}
        for cell in r.findall(M + "c"):
            col = "".join(ch for ch in cell.get("r") if ch.isalpha())
            if cell.get("t") == "inlineStr":
                c[col] = "".join(x.text or "" for x in cell.iter(M + "t"))
            else:
                v = cell.find(M + "v")
                c[col] = "" if v is None else v.text
        rows.append(c)
    out = []
    for c in rows[1:]:
        if not c.get("A"):
            continue
        out.append({"no": c["A"], "biz": c.get("B", ""), "q": nfc(c.get("C", "")),
                    "conf": c.get("E", ""), "src": nfc(c.get("G", ""))})
    return out


SRC_SPLIT = re.compile(r"\[L(\d+)\]")


def map_gold(qa_rows, doc_norm, pos_map, els):
    """doc_norm: 공백 제거 전문. pos_map[i] = 원문 라인 번호."""
    line_of = pos_map
    el_by_line = []
    for e in els:
        el_by_line.append((e["start"], e["end"], e["eid"]))

    def overlap_elements(l0, l1):
        return [eid for s, en, eid in el_by_line if s <= l1 and en >= l0]

    mapped = []
    for row in qa_rows:
        if row["conf"] != "확정":
            continue
        parts = SRC_SPLIT.split(row["src"])
        golds, spans = set(), []
        for i in range(1, len(parts), 2):
            lineno = int(parts[i]); quote = parts[i + 1].strip()
            qn = ns(quote)[:300]
            if len(qn) < 10:
                continue
            hits = []
            st = 0
            while True:
                j = doc_norm.find(qn, st)
                if j < 0:
                    break
                hits.append(j); st = j + 1
                if len(hits) > 20:
                    break
            if not hits and len(qn) > 80:
                qn2 = qn[:80]
                st = 0
                while True:
                    j = doc_norm.find(qn2, st)
                    if j < 0:
                        break
                    hits.append(j); st = j + 1
                    if len(hits) > 20:
                        break
                qn = qn2
            if not hits:
                continue
            best = min(hits, key=lambda j: abs(line_of[j] - lineno))
            l0, l1 = line_of[best], line_of[min(best + len(qn) - 1,
                                                len(line_of) - 1)]
            for eid in overlap_elements(l0, l1):
                golds.add(eid)
            spans.append([l0, l1])
        if golds:
            mapped.append({"qid": row["no"], "q": row["q"], "biz": row["biz"],
                           "gold": sorted(golds), "spans": spans})
    return mapped


# ── 5. paired bootstrap ────────────────────────────────────────────────

def boot_ci(deltas, seed=20260805, n=10000):
    rng = Random(seed)
    N = len(deltas)
    ms = sorted(sum(deltas[rng.randrange(N)] for _ in range(N)) / N
                for _ in range(n))
    return [ms[int(0.025 * n)], ms[int(0.975 * n) - 1]]


# ── main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--qa-xlsx", required=True)
    ap.add_argument("--dev-qids", required=True,
                    help="qa100_gold.jsonl (qid 필드) — dev 분할 정의")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    raw = Path(args.doc).read_text(encoding="utf-8", errors="ignore")
    doc_sha = hashlib.sha256(raw.encode()).hexdigest()
    lines = nfc(raw).splitlines()
    print(f"doc lines={len(lines):,} sha={doc_sha[:12]}", flush=True)

    els = annotate_context(split_elements(lines))
    print(f"elements={len(els):,} types={Counter(e['type'] for e in els)}",
          flush=True)

    # 정규화 전문 + 위치→라인 맵
    buf, lmap = [], []
    for li, line in enumerate(lines, 1):
        t = ns(line)
        buf.append(t)
        lmap.extend([li] * len(t))
    doc_norm = "".join(buf)

    qa = parse_xlsx(args.qa_xlsx)
    print(f"xlsx rows={len(qa)} 확정={sum(1 for r in qa if r['conf']=='확정')}",
          flush=True)
    mapped = map_gold(qa, doc_norm, lmap, els)
    print(f"mapped={len(mapped)}", flush=True)

    dev_qids = {json.loads(l)["qid"] for l in open(args.dev_qids)}
    dev = [m for m in mapped if m["qid"] in dev_qids]
    test = [m for m in mapped if m["qid"] not in dev_qids]
    print(f"dev={len(dev)} test={len(test)}", flush=True)
    (out / "gold_mapped.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in mapped),
        encoding="utf-8")

    schemas = ["A0", "tag.s0", "tag.s1", "tag.s1k1", "tag.s1k2",
               "tag.s1k3", "tag.s1k4", "tag.s1k4r2"]

    # ── verify 채널: 원문 BM25 순위(스키마 불변) + 태그 멤버십 분할 ──
    TSTOP = {"무엇인가요", "알려줘", "가능한가요", "가능해", "되나요", "인가요",
             "해주세요", "해줘", "설명해줘", "정리해줘"}
    def qtokens(q):
        out = []
        for t in re.split(r"[\s,?.!·()\[\]]+", q):
            t = t.strip()
            if 2 <= len(t) <= 14 and re.search(r"[가-힣A-Z0-9]", t) and t not in TSTOP:
                out.append(t)
        return out
    JONUM_Q = re.compile(r"제\s?\d+(?:-\d+)?\s?조(?:의\s?\d+)?")
    CODE_Q = re.compile(r"[A-Z]\d{2}(?:\.\d{1,2})?")
    def qtokens_sel(q):
        out = [t for t in qtokens(q)
               if t.endswith("특약") or "특약" in t or JONUM_Q.fullmatch(t)
               or CODE_Q.fullmatch(t) or t in ROLE_VOCAB]
        out += JONUM_Q.findall(q) + CODE_Q.findall(q)
        return list(dict.fromkeys(out))
    def verify_rank(base_top, tags_of, toks, depth=20):
        head = base_top[:depth]
        passed = [i for i in head if any(t in tags_of[i] for t in toks)]
        if not passed:
            return base_top
        failed = [i for i in head if i not in set(passed)]
        return passed + failed + base_top[depth:]
    results = {}
    prev_rows = None
    champion, champ_rows, champ_mrr = None, None, -1
    bm_a0 = BM25([e["text"] for e in els])
    a0_cache = {}
    def a0_top(q, n=200):
        if q not in a0_cache:
            qb = list(Counter(bigrams(q)))
            scores = []
            for i in range(bm_a0.N):
                sc = 0.0
                tfi = bm_a0.tf[i]
                for t in qb:
                    fq = tfi.get(t, 0)
                    if not fq:
                        continue
                    idf = math.log(1 + (bm_a0.N - bm_a0.df[t] + 0.5) / (bm_a0.df[t] + 0.5))
                    B = 1 - bm_a0.b + bm_a0.b * (bm_a0.dl[i] / bm_a0.avgdl)
                    sc += idf * fq * (bm_a0.k1 + 1) / (fq + bm_a0.k1 * B)
                scores.append(sc)
            a0_cache[q] = sorted(range(bm_a0.N), key=lambda i: -scores[i])[:n]
        return a0_cache[q]
    for schema in schemas:
        tags = [build_tag(e, schema) for e in els]
        reps = [(t + " ||| " + e["text"]) if t else e["text"]
                for t, e in zip(tags, els)]
        tlen = [len(t) for t in tags if t]
        bm = BM25(reps)
        def run(items):
            rows = []
            for it in items:
                top10 = [els[i]["eid"] for i in bm.rank10(it["q"])]
                g = set(it["gold"])
                r = next((k + 1 for k, x in enumerate(top10) if x in g), None)
                rows.append({"qid": it["qid"],
                             "r5": int(bool(r and r <= 5)),
                             "r10": int(bool(r)),
                             "mrr": (1 / r) if r else 0.0})
            return rows
        rows = run(dev)
        mrr = sum(r["mrr"] for r in rows) / len(rows)
        r5 = sum(r["r5"] for r in rows) / len(rows)
        entry = {"dev_mrr10": round(mrr, 4), "dev_r5": round(r5, 4),
                 "dev_r10": round(sum(r['r10'] for r in rows) / len(rows), 4),
                 "tag_len_p50": sorted(tlen)[len(tlen) // 2] if tlen else 0,
                 "tag_len_max": max(tlen) if tlen else 0}
        if prev_rows is not None:
            pm = {r["qid"]: r for r in prev_rows}
            deltas = [r["mrr"] - pm[r["qid"]]["mrr"] for r in rows]
            entry["dev_dmrr_vs_prev"] = round(sum(deltas) / len(deltas), 4)
            entry["dev_dmrr_ci"] = [round(x, 4) for x in boot_ci(deltas)]
        # verify 채널 (concat 아님 — 순위 보존 + 태그 확인 분할)
        vrows = []
        for it in dev:
            toks = qtokens(it["q"])
            order = verify_rank(a0_top(it["q"]), tags, toks) if schema != "A0" else a0_top(it["q"])
            top10 = [els[i]["eid"] for i in order[:10]]
            g = set(it["gold"])
            rk = next((k + 1 for k, x in enumerate(top10) if x in g), None)
            vrows.append({"qid": it["qid"], "r5": int(bool(rk and rk <= 5)),
                          "r10": int(bool(rk)), "mrr": (1 / rk) if rk else 0.0})
        entry["verify_dev_mrr10"] = round(sum(r["mrr"] for r in vrows) / len(vrows), 4)
        entry["verify_dev_r5"] = round(sum(r["r5"] for r in vrows) / len(vrows), 4)
        svrows = []
        for it in dev:
            toks = qtokens_sel(it["q"])
            order = verify_rank(a0_top(it["q"]), tags, toks) if (schema != "A0" and toks) else a0_top(it["q"])
            top10 = [els[i]["eid"] for i in order[:10]]
            g = set(it["gold"])
            rk = next((k + 1 for k, x in enumerate(top10) if x in g), None)
            svrows.append({"qid": it["qid"], "r5": int(bool(rk and rk <= 5)),
                           "mrr": (1 / rk) if rk else 0.0})
        entry["sel_verify_dev_mrr10"] = round(sum(r["mrr"] for r in svrows) / len(svrows), 4)
        entry["sel_verify_dev_r5"] = round(sum(r["r5"] for r in svrows) / len(svrows), 4)
        print(f"    verify: MRR={entry['verify_dev_mrr10']:.4f} R@5={entry['verify_dev_r5']:.4f}  |  sel_verify: MRR={entry['sel_verify_dev_mrr10']:.4f} R@5={entry['sel_verify_dev_r5']:.4f}", flush=True)
        results[schema] = entry
        prev_rows = rows
        if mrr > champ_mrr:
            champion, champ_rows, champ_mrr = schema, rows, mrr
        print(f"{schema:12s} dev MRR={mrr:.4f} R@5={r5:.4f} "
              f"len(p50/max)={entry['tag_len_p50']}/{entry['tag_len_max']}",
              flush=True)

    # 동결 test 1회: champion + 대조(A0, s0)
    for schema in {"A0", "tag.s0", champion}:
        tags = [build_tag(e, schema) for e in els]
        reps = [(t + " ||| " + e["text"]) if t else e["text"]
                for t, e in zip(tags, els)]
        bm = BM25(reps)
        rows = []
        for it in test:
            top10 = [els[i]["eid"] for i in bm.rank10(it["q"])]
            g = set(it["gold"])
            r = next((k + 1 for k, x in enumerate(top10) if x in g), None)
            rows.append({"qid": it["qid"], "r5": int(bool(r and r <= 5)),
                         "r10": int(bool(r)), "mrr": (1 / r) if r else 0.0})
        results.setdefault("test", {})[schema] = {
            "mrr10": round(sum(r["mrr"] for r in rows) / len(rows), 4),
            "r5": round(sum(r["r5"] for r in rows) / len(rows), 4),
            "r10": round(sum(r["r10"] for r in rows) / len(rows), 4),
            "rows": rows}
    tA = results["test"]["A0"]; tC = results["test"][champion]
    pa = {r["qid"]: r for r in tA["rows"]}
    deltas = [r["mrr"] - pa[r["qid"]]["mrr"] for r in tC["rows"]]
    results["test"]["champion"] = champion
    results["test"]["champ_vs_A0_dmrr"] = round(sum(deltas) / len(deltas), 4)
    results["test"]["champ_vs_A0_ci"] = [round(x, 4) for x in boot_ci(deltas)]
    for k in {"A0", "tag.s0", champion}:
        results["test"][k].pop("rows", None)

    payload = {"doc_sha256": doc_sha, "n_elements": len(els),
               "n_mapped": len(mapped), "n_dev": len(dev), "n_test": len(test),
               "schemas": results}
    (out / "semtag_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results.get("test"), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
