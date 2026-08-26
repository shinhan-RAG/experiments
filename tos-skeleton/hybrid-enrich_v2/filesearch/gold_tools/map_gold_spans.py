#!/usr/bin/env python3
"""noah v3 QA(csv)의 `출처` 인용문 → 원문 char span (검색 독립 Gold).

hybrid-enrich/build_gold_noah_v3.py 의 규칙을 따른다: 공백 제거·NFC 정규화 후 exact substring,
실패 시 앞 60% 재시도. 인용문은 줄 단위로 분리해 각 줄 = evidence group 후보로 두되,
같은 문항의 연속 줄이 원문에서도 인접(간격 ≤ 400자)하면 하나의 group 으로 병합한다.

출력: qid, q, task_type, core_retrieval, groups=[{c0,c1,...}], unmapped=[...]
train/test 어느 split에도 동일하게 적용하며, 선택한 입력 경로·내용 hash와 생성기 hash를
manifest에 고정한다. 검색 결과·Gold JO·실험 산출물은 읽지 않는다.
"""
import argparse, csv, hashlib, json, os, re, sys, unicodedata
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm_map(raw):
    """공백 제거 정규화 문자열과 (정규화 idx → 원문 idx) 배열."""
    out, idx = [], []
    for i, ch in enumerate(raw):
        if ch.isspace():
            continue
        out.append(ch)
        idx.append(i)
    return "".join(out), idx


def all_occ(nd, cand, cap=400):
    out, p = [], nd.find(cand)
    while p >= 0 and len(out) < cap:
        out.append(p)
        p = nd.find(cand, p + 1)
    return out


def norm_q(quote):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", quote))


def map_group(nd, idx, lines):
    """한 evidence group(--- 로 구분된 발췌 블록)의 줄들을 원문에 앵커링.
    가장 매치 수가 적은 줄을 앵커로 삼고, 나머지 줄은 앵커에 가장 가까운 매치를 택한다."""
    cands = []
    for ln in lines:
        q = norm_q(ln)
        if len(q) < 4:
            continue
        occ = all_occ(nd, q)
        partial = False
        if not occ and len(q) >= 20:
            occ = all_occ(nd, q[: int(len(q) * 0.6)]); partial = True
            q = q[: int(len(q) * 0.6)]
        cands.append((ln, q, occ, partial))
    mapped = [c for c in cands if c[2]]
    if not mapped:
        return None, [c[0][:120] for c in cands]
    anchor = min(mapped, key=lambda c: (len(c[2]), -len(c[1])))
    a_pos = anchor[2][0]
    picks = []
    for ln, q, occ, partial in mapped:
        p = min(occ, key=lambda x: abs(x - a_pos))
        picks.append((p, p + len(q), partial, len(occ)))
    c0 = min(p[0] for p in picks); c1 = max(p[1] for p in picks)
    # 앵커에서 지나치게 먼 줄(> 3000 정규화 자)은 오매핑으로 보고 제외
    near = [p for p in picks if abs(p[0] - a_pos) <= 3000] or picks
    c0 = min(p[0] for p in near); c1 = max(p[1] for p in near)
    return {"c0": idx[c0], "c1": idx[c1 - 1] + 1, "n_lines": len(lines), "n_mapped": len(near),
            "anchor_multi": len(anchor[2]), "quotes": [l[:80] for l in lines[:3]]}, \
           [c[0][:120] for c in cands if not c[2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--qa", required=True, help="official QA CSV for the selected split")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "gold_spans_train.jsonl"))
    a = ap.parse_args()
    raw = unicodedata.normalize("NFC", open(a.doc, encoding="utf-8").read().replace("\r\n", "\n"))
    nd, idx = norm_map(raw)
    rows = list(csv.DictReader(open(a.qa, encoding="utf-8-sig")))
    n_ok = n_partial = n_un = 0
    output = Path(a.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for r in rows:
            src = (r["출처"] or "").replace("\r", "")
            blocks = [b for b in re.split(r"\n-{3,}\n|^-{3,}\n|\n-{3,}$", src) if b.strip()]
            groups, unmapped = [], []
            for b in blocks:
                lines = [x.strip() for x in b.split("\n") if x.strip() and not re.fullmatch(r"-{3,}|\d{10,}|\d{1,4}", x.strip())]
                if not lines:
                    continue
                g, un = map_group(nd, idx, lines)
                unmapped += un
                if g:
                    groups.append(g)
            groups.sort(key=lambda g: g["c0"])
            n_blocks = len([b for b in blocks if b.strip()])
            status = "ok" if groups and len(groups) == n_blocks and not unmapped else ("partial" if groups else "unmapped")
            n_ok += status == "ok"; n_partial += status == "partial"; n_un += status == "unmapped"
            f.write(json.dumps({
                "qid": r["qid"], "q": r["질문"], "task_type": r["task_type"], "core_retrieval": r["core_retrieval"],
                "신뢰도": r["신뢰도"], "groups": groups, "unmapped": unmapped, "n_blocks": n_blocks, "status": status,
            }, ensure_ascii=False) + "\n")
    counts = {"n": len(rows), "ok": n_ok, "partial": n_partial, "unmapped": n_un}
    manifest = {
        "policy": "retrieval-blind official source quote to raw character span",
        "retrieval_blind": True,
        "inputs": {
            "document": {"path": str(Path(a.doc).resolve()), "sha256": sha256(a.doc)},
            "official_csv": {"path": str(Path(a.qa).resolve()), "sha256": sha256(a.qa)},
        },
        "generator": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "python": sys.version,
        "counts": counts,
        "output": {"path": str(output.resolve()), "sha256": sha256(output)},
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts | {"output_sha256": sha256(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
