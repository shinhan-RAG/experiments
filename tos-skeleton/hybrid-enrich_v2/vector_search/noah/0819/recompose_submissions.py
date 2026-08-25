#!/usr/bin/env python3
"""호스트측 결정론 제출 재조합(R10+R11).

R11(--dedup-family-thr): 같은 특약 계열(판본 접미·대괄호 수식어 제거 후 동일)의
근사중복 조(공백 제거 bigram Jaccard >= thr)를 뒤 항목 제거·후미 강등한다.
gold 명세 §3 이 내용 동등 판본을 OR 로 취급하므로 같은 계열 판본 중복 제거는 채점상 안전하며,
질의에 판본 변별 표면(계약명 차집합 토큰)이 있으면 제거하지 않는다(판본 열거형 질문 보호).
교차 런 검증: r11v/r1v 852셀에서 thr 0.75~0.85 손실 0 (thr 0.70 은 양 런 손실 — 채택 0.75).

R10 — 에이전트 제출 상위 5 보존, 첫 검색 top-N 을 6위부터 보충.

에이전트 재실행 없이 기존 run 의 host_trace 에서 첫 search 반환 순서를 읽어
submitted 리스트를 재조합해 재채점한다. 상위 5는 절대 재배열하지 않으므로(append-only)
R@5 는 top5 미달(<5 gold 커버) 문항에서만 변할 수 있다 — 제로섬 강등이 구조적으로 불가능.

사용:
  python3 recompose_submissions.py --run out/host_agent/<run>/ \
      --gold ../../../filesearch/out/gold_....jsonl [--anchor-n 2] \
      --out results_recomposed.jsonl
"""
import argparse
import json
import re
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
sys.path.insert(0, str(FS))
from scoring import score  # noqa: E402
from units import Units  # noqa: E402


def first_search_ids(trace):
    """host_trace 에서 최초 search 응답의 노출 id 순서."""
    for turn in trace:
        resp = turn.get("response") or {}
        results = resp.get("results")
        if results:
            order = []
            for item in results:
                i = item.get("id")
                if i and i not in order:
                    order.append(i)
            return order
    return []


def family_dedup(submitted, question, jo_units, thr):
    """같은 특약 계열의 근사중복 판본 조를 뒤 항목 제거·후미 강등 (R11)."""
    from textmatch import contract_core
    bg_cache = {}

    def core(jid):
        cs = jo_units.get(jid, {}).get("contract_scope", "")
        return contract_core(re.sub(r"\[.*?\]", "", cs))

    def bigrams(jid):
        if jid not in bg_cache:
            s = re.sub(r"\s+", "", jo_units.get(jid, {}).get("text", ""))
            bg_cache[jid] = {s[i:i + 2] for i in range(len(s) - 1)}
        return bg_cache[jid]

    def jaccard(a, b):
        A, B = bigrams(a), bigrams(b)
        return len(A & B) / len(A | B) if A and B else 0.0

    def disc_tokens(a, b):
        ca = set(re.findall(r"[가-힣A-Za-z0-9]+", jo_units.get(a, {}).get("contract_scope", "")))
        cb = set(re.findall(r"[가-힣A-Za-z0-9]+", jo_units.get(b, {}).get("contract_scope", "")))
        return {t for t in (ca - cb) | (cb - ca) if len(t) >= 2}

    kept, removed = [], []
    for x in submitted:
        dup = None
        if x.startswith("j") and x in jo_units:
            cx = core(x)
            for k in kept:
                if (k.startswith("j") and k in jo_units and cx
                        and core(k) == cx and jaccard(x, k) >= thr):
                    dup = k
                    break
        if dup is not None and not any(t in question for t in disc_tokens(x, dup)):
            removed.append(x)
        else:
            kept.append(x)
    return kept + removed


def recompose(submitted, fs_ids, anchor_n):
    head = submitted[:5]
    anchors = [x for x in fs_ids[:anchor_n] if x not in head]
    tail = [x for x in submitted[5:] if x not in anchors]
    return (head + anchors + tail)[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--jo", default=str(FS / "out/elements_u3jo.jsonl"))
    ap.add_argument("--anchor-n", type=int, default=2)
    ap.add_argument("--dedup-family-thr", type=float, default=0.0,
                    help="R11: 같은 계열 근사중복 판본 축약 임계(0=비활성, 채택값 0.75)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run = Path(a.run)
    gold = {json.loads(l)["qid"]: json.loads(l) for l in open(a.gold, encoding="utf-8")}
    units = Units(a.jo)
    jo_units = ({json.loads(l)["element_id"]: json.loads(l)
                 for l in open(a.jo, encoding="utf-8")}
                if a.dedup_family_thr else {})
    rows = []
    for line in open(run / "results.jsonl", encoding="utf-8"):
        r = json.loads(line)
        sessions = sorted(run.glob(f"sessions/{r['qid']}_*_r{r['rep']}"))
        trace_path = next((s / "host_trace.json" for s in sessions
                           if (s / "host_trace.json").exists()), None)
        fs_ids = (first_search_ids(json.loads(trace_path.read_text(encoding="utf-8")))
                  if trace_path else [])
        sub = list(r["submitted"])
        if a.dedup_family_thr:
            sub = family_dedup(sub, gold[r["qid"]]["q"], jo_units, a.dedup_family_thr)
        sub2 = recompose(sub, fs_ids, a.anchor_n)
        sc = score(units.resolve(sub2), gold[r["qid"]]["groups"], ks=(1, 5, 10, 20))
        rows.append({**r, "submitted": sub2, "recomposed": sub2 != list(r["submitted"])[:10],
                     **{k: sc[k] for k in ("R@1", "R@5", "R@10", "suff@5")}})
    out = Path(a.out)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                   encoding="utf-8")
    byq = {}
    for r in rows:
        byq.setdefault(r["qid"], []).append(r["R@5"])
    macro = sum(sum(v) / len(v) for v in byq.values()) / len(byq)
    print(json.dumps({"rows": len(rows), "qids": len(byq), "anchor_n": a.anchor_n,
                      "R@5_macro": round(macro, 4),
                      "recomposed_rows": sum(r["recomposed"] for r in rows)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
