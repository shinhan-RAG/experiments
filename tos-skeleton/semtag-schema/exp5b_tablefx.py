#!/usr/bin/env python3
"""실험 5b — 표·수식 element 직렬화 정규화 (검색 뷰 텍스트 변환만, 분할·eid 불변).

- A0: 원문 그대로 (기준선 게이트: dev 96 R@5=.3125, MRR@10≈.2647)
- V1: table element 마크다운 표 구문 제거(파이프·구분행 제거, 셀 텍스트 공백 연결)
      + formula element LaTeX 구문 제거($$·제어 시퀀스·중괄호 제거, 한글·숫자·연산 보존)
- V2: V1 + 표 헤더 전파(각 데이터 행 앞에 그 표의 헤더 행 셀들을 prefix)

전 과정 결정적 — LLM·API 호출 없음. 페이지 헤더("# SHINHAN LIFE" 블록)는 건드리지 않음(실험 5a 변수).
dev만 평가. test 채점·보고 금지.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from semtag_experiment import nfc, split_elements, BM25, boot_ci  # noqa: E402

DOC = ("/Users/donggyu/Documents/논문/신한라이프/신한RAG_QA셋_원본문서_20260710/"
       "원본문서/md/판매약관_신한(간편가입)통합건강보험 원(ONE)"
       "(무배당, 해약환급금 미지급형)_260507.md")
GOLD = HERE / "out" / "gold_mapped.jsonl"
DEVQ = HERE.parent / "hybrid-enrich" / "out" / "qa100_gold.jsonl"
OUT = HERE / "out" / "exp5b"

GATE_R5, GATE_MRR = 0.3125, 0.2647  # A0 dev 96 기준선 (재현 필수)

# ── 표 구문 정규화 ─────────────────────────────────────────────────────

def is_table_row(line):
    return line.lstrip().startswith("|")


def is_sep_row(line):
    s = line.strip()
    if not s.startswith("|"):
        return False
    body = s.strip("|").strip()
    return bool(body) and set(body) <= set("-:| ") and "-" in body


def row_cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def table_lines(text, header=None, skip_first_row_prefix=False):
    """마크다운 표 구문 제거. header가 주어지면 데이터 행마다 헤더 셀을 prefix."""
    out, seen_rows = [], 0
    for line in text.split("\n"):
        if not is_table_row(line):
            out.append(line)
            continue
        if is_sep_row(line):
            continue
        cells = [c for c in row_cells(line) if c]
        if not cells:
            continue
        seen_rows += 1
        joined = " ".join(cells)
        if header and not (skip_first_row_prefix and seen_rows == 1):
            joined = " ".join(header) + " " + joined
        out.append(joined)
    return "\n".join(out)


# ── LaTeX 구문 정규화 ──────────────────────────────────────────────────

OP_MAP = [("times", "×"), ("div", "÷"), ("cdot", "·"), ("pm", "±"),
          ("leq", "≤"), ("le", "≤"), ("geq", "≥"), ("ge", "≥"), ("neq", "≠"),
          ("sim", "~"), ("therefore", "∴"), ("Delta", "Δ"), ("pi", "π"),
          ("alpha", "α"), ("angle", "∠"), ("circ", "°"), ("infty", "∞")]


def clean_formula_text(text):
    t = text
    # phantom(비가시 공백)·begin/end(환경 이름 인자)는 인자까지 제거
    t = re.sub(r"\\phantom\s*\{[^{}]*\}", " ", t)
    t = re.sub(r"\\begin\s*\{[^{}]*\}(\s*\{[^{}]*\})*", " ", t)
    t = re.sub(r"\\end\s*\{[^{}]*\}", " ", t)
    # 단순 \frac{a}{b} → a/b (연산 보존; 중첩 중괄호는 아래 일반 규칙으로 처리)
    t = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", t)
    # 연산·기호 명령은 기호로 치환(보존), 그 외 제어 시퀀스는 이름만 제거(인자 내용 보존)
    for name, sym in OP_MAP:
        t = re.sub(r"\\" + name + r"(?![a-zA-Z])", sym, t)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    # 잔여 백슬래시(\\, \{ 등)·달러·중괄호 제거
    t = t.replace("\\", " ").replace("$", " ")
    t = t.replace("{", " ").replace("}", " ")
    t = re.sub(r"[ \t]+", " ", t)
    return t


# ── 변형 적용 (검색 뷰만 — els 자체는 불변) ────────────────────────────

def table_runs(els):
    """MAX_LINES 분할로 쪼개진 연속 table element를 하나의 표로 묶는다."""
    runs, i, n = [], 0, len(els)
    while i < n:
        if els[i]["type"] != "table":
            i += 1
            continue
        j = i
        while (j + 1 < n and els[j + 1]["type"] == "table"
               and els[j + 1]["start"] == els[j]["end"] + 1):
            j += 1
        runs.append((i, j))
        i = j + 1
    return runs


def run_header(els, i, j):
    """run의 첫 표 행이 헤더인지(markdown: 둘째 표 행이 구분행) 판정."""
    rows = []
    for k in range(i, j + 1):
        rows += [l for l in els[k]["text"].split("\n") if is_table_row(l)]
        if len(rows) >= 2:
            break
    if len(rows) >= 2 and not is_sep_row(rows[0]) and is_sep_row(rows[1]):
        cells = [c for c in row_cells(rows[0]) if c]
        return cells or None
    return None


def build_view(els, variant):
    """variant in {A0, V1, V2} → 검색용 텍스트 리스트(eid 순서 = els 순서)."""
    if variant == "A0":
        return [e["text"] for e in els]
    header_of = {}
    first_of_run = set()
    if variant == "V2":
        for i, j in table_runs(els):
            h = run_header(els, i, j)
            for k in range(i, j + 1):
                header_of[k] = h
            first_of_run.add(i)
    views = []
    for idx, e in enumerate(els):
        if e["type"] == "table":
            views.append(table_lines(
                e["text"], header=header_of.get(idx),
                skip_first_row_prefix=(idx in first_of_run)))
        elif e["type"] == "formula":
            views.append(clean_formula_text(e["text"]))
        else:
            views.append(e["text"])
    return views


# ── 평가 ───────────────────────────────────────────────────────────────

def evaluate(views, dev, els):
    bm = BM25(views)
    rows = []
    for it in dev:
        top10 = [els[i]["eid"] for i in bm.rank10(it["q"])]
        g = set(it["gold"])
        r = next((k + 1 for k, x in enumerate(top10) if x in g), None)
        rows.append({"qid": it["qid"], "rank": r,
                     "r5": int(bool(r and r <= 5)), "r10": int(bool(r)),
                     "mrr": (1 / r) if r else 0.0})
    return rows


def agg(rows):
    n = len(rows)
    return {"n": n,
            "r5": round(sum(r["r5"] for r in rows) / n, 4),
            "r10": round(sum(r["r10"] for r in rows) / n, 4),
            "mrr10": round(sum(r["mrr"] for r in rows) / n, 4)}


def paired(base_rows, var_rows):
    b = {r["qid"]: r for r in base_rows}
    dmrr = [r["mrr"] - b[r["qid"]]["mrr"] for r in var_rows]
    dr5 = [r["r5"] - b[r["qid"]]["r5"] for r in var_rows]
    imp = [r["qid"] for r in var_rows if r["r5"] == 1 and b[r["qid"]]["r5"] == 0]
    reg = [r["qid"] for r in var_rows if r["r5"] == 0 and b[r["qid"]]["r5"] == 1]
    return {"dmrr": round(sum(dmrr) / len(dmrr), 4),
            "dmrr_ci95": [round(x, 4) for x in boot_ci(dmrr)],
            "dr5": round(sum(dr5) / len(dr5), 4),
            "dr5_ci95": [round(x, 4) for x in boot_ci(dr5)],
            "r5_improved": imp, "r5_regressed": reg}


def by_group(rows, groups):
    out = {}
    for gname, qids in groups.items():
        sub = [r for r in rows if r["qid"] in qids]
        out[gname] = agg(sub)
    return out


def group_transitions(base_rows, var_rows, groups):
    b = {r["qid"]: r for r in base_rows}
    out = {}
    for gname, qids in groups.items():
        sub = [r for r in var_rows if r["qid"] in qids]
        imp = [r["qid"] for r in sub if r["r5"] == 1 and b[r["qid"]]["r5"] == 0]
        reg = [r["qid"] for r in sub if r["r5"] == 0 and b[r["qid"]]["r5"] == 1]
        dmrr = [r["mrr"] - b[r["qid"]]["mrr"] for r in sub]
        out[gname] = {"n": len(sub), "r5_improved": imp, "r5_regressed": reg,
                      "dmrr": round(sum(dmrr) / len(dmrr), 4)}
    return out


# ── main ───────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = nfc(Path(DOC).read_text(encoding="utf-8", errors="ignore")).splitlines()
    els = split_elements(lines)
    by_eid = {e["eid"]: e for e in els}
    print(f"elements={len(els)} types={Counter(e['type'] for e in els)}", flush=True)

    mapped = [json.loads(l) for l in open(GOLD, encoding="utf-8")]
    dev_qids = {json.loads(l)["qid"] for l in open(DEVQ, encoding="utf-8")}
    dev = [m for m in mapped if m["qid"] in dev_qids]
    print(f"dev={len(dev)} (test는 채점하지 않음)", flush=True)
    assert len(dev) == 96, f"dev must be 96, got {len(dev)}"

    # gold 유형 분해: 첫 gold element(eid 정렬)의 type — 17/35/44 재현 확인
    def gclass(m):
        return by_eid[sorted(m["gold"])[0]]["type"]
    groups = {"table_gold": {m["qid"] for m in dev if gclass(m) == "table"},
              "formula_gold": {m["qid"] for m in dev if gclass(m) == "formula"},
              "text_gold": {m["qid"] for m in dev if gclass(m) == "text"}}
    gsizes = {k: len(v) for k, v in groups.items()}
    print(f"gold groups={gsizes}", flush=True)
    assert gsizes == {"table_gold": 17, "formula_gold": 35, "text_gold": 44}, gsizes

    def full_pass():
        out = {}
        for v in ("A0", "V1", "V2"):
            out[v] = evaluate(build_view(els, v), dev, els)
        return out

    pass1 = full_pass()
    pass2 = full_pass()
    determinism = (json.dumps(pass1, sort_keys=True)
                   == json.dumps(pass2, sort_keys=True))
    print(f"determinism(2-pass identical)={determinism}", flush=True)

    rows = pass1
    a0 = agg(rows["A0"])
    print(f"A0 dev: {a0}", flush=True)
    gate_ok = (a0["r5"] == GATE_R5 and a0["mrr10"] == GATE_MRR)
    if not gate_ok:
        payload = {"gate": {"ok": False, "expected": {"r5": GATE_R5, "mrr10": GATE_MRR},
                            "got": a0}}
        (OUT / "results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("BASELINE GATE FAILED — 중단", flush=True)
        sys.exit(1)
    print("baseline gate OK", flush=True)

    # 뷰 길이 통계 (문자 수, 공백 제거 전)
    def len_stats(views, etype):
        ls = sorted(len(v) for v, e in zip(views, els) if e["type"] == etype)
        return {"n": len(ls), "p50": ls[len(ls) // 2], "total": sum(ls)}
    vstats = {}
    for v in ("A0", "V1", "V2"):
        views = build_view(els, v)
        vstats[v] = {"table": len_stats(views, "table"),
                     "formula": len_stats(views, "formula")}

    results = {
        "doc": DOC, "n_elements": len(els), "n_dev": len(dev),
        "note": "dev only — test 미채점. 분할·eid 불변, 검색 뷰 텍스트 변환만. "
                "페이지 헤더 블록 미변경(5a 변수 분리). 결정적(LLM 없음).",
        "gate": {"ok": True, "expected": {"r5": GATE_R5, "mrr10": GATE_MRR}, "got": a0},
        "gold_groups": {k: sorted(v, key=lambda q: (len(q), q))
                        for k, v in groups.items()},
        "gold_group_sizes": gsizes,
        "determinism_2pass_identical": determinism,
        "view_len_chars": vstats,
        "metrics": {v: agg(rows[v]) for v in ("A0", "V1", "V2")},
        "paired_vs_A0": {v: paired(rows["A0"], rows[v]) for v in ("V1", "V2")},
        "group_metrics": {v: by_group(rows[v], groups) for v in ("A0", "V1", "V2")},
        "group_transitions_vs_A0": {
            v: group_transitions(rows["A0"], rows[v], groups) for v in ("V1", "V2")},
        "per_query": {v: rows[v] for v in ("A0", "V1", "V2")},
    }
    (OUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    for v in ("V1", "V2"):
        m, p = results["metrics"][v], results["paired_vs_A0"][v]
        print(f"{v}: R@5={m['r5']} R@10={m['r10']} MRR@10={m['mrr10']} "
              f"dMRR={p['dmrr']} CI={p['dmrr_ci95']} "
              f"+{len(p['r5_improved'])}/-{len(p['r5_regressed'])}", flush=True)
    print("group metrics:", json.dumps(results["group_metrics"], ensure_ascii=False),
          flush=True)


if __name__ == "__main__":
    main()
