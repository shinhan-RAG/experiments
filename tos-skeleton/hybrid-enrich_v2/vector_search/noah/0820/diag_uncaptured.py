#!/usr/bin/env python3
"""미포착 그룹 원인 분해 — ceiling ①(인덱스 한계) 문항의 gold 조가 왜 안 잡히는가.

그룹(=gold 근거 1곳) 단위로 4가지를 판정한다:
  A. tag_missing   gold element 에 태그 레코드가 없거나 실질 필드가 비어 있음 → 태깅 보강 대상
  B. lex_mismatch  질문 내용어와 gold 조 텍스트+태그의 어휘 겹침 0 → alias/패러프레이즈 대상
  C. chunk_missing gold 조 span 을 덮는 V9 청크가 없음 → 메타 뷰 커버리지 구멍
  D. rank_only     둘 다 있고 어휘도 겹치는데 순위 밖 → 가중치/랭킹 문제(인덱스 보강 아님)

입력: out/ceiling_train.json, filesearch/out/{gold_spans_lsh_train,tags_u2_rules,elements_u2jo}.jsonl,
      vector_search/out/chunks.jsonl
출력: out/diag_uncaptured.json + 요약표
"""
import json, re, sys, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
VS = HERE.parents[1]
sys.path.insert(0, str(FS))

TOK = re.compile(r"[가-힣a-zA-Z0-9]{2,}")
STOP = {"알려줘", "해줘", "뭐야", "무엇", "어떤", "어떻게", "대해", "관련", "경우", "있나요", "인가요",
        "되나요", "합니까", "주세요", "정리", "설명", "특약", "보험", "약관", "조건", "기준", "종류"}


def toks(s):
    return {t for t in TOK.findall(s or "") if t not in STOP}


def main():
    ceil = json.load(open(HERE / "out/ceiling_train.json", encoding="utf-8"))
    gold = {g["qid"]: g for g in (json.loads(l) for l in open(FS / "out/gold_spans_lsh_train.jsonl", encoding="utf-8"))}
    tags = {}
    for l in open(FS / "out/tags_u2_rules.jsonl", encoding="utf-8"):
        t = json.loads(l); tags[t["element_id"]] = t
    E = {}
    for l in open(FS / "out/elements_u2.jsonl", encoding="utf-8"):
        e = json.loads(l); E[e["element_id"]] = e
    J = [json.loads(l) for l in open(FS / "out/elements_u2jo.jsonl", encoding="utf-8")]
    chunks = [json.loads(l) for l in open(VS / "out/chunks.jsonl", encoding="utf-8")]

    def jo_of(c0, c1):
        best, bo = None, 0
        for u in J:
            o = min(c1, u["char_end"]) - max(c0, u["char_start"])
            if o > bo:
                best, bo = u, o
        return best

    def tag_substance(t):
        """태그 레코드의 실질 필드 수(빈 태깅 감지)."""
        n = 0
        for k in ("role", "subject", "qualifier"):
            n += len(t.get(k) or [])
        n += 1 if t.get("schema_tag") else 0
        n += 1 if (t.get("locator") or {}).get("article_title") else 0
        return n

    rows_out, cause = [], collections.Counter()
    targets = [r for r in ceil["rows"] if r["class"].startswith("①")]
    for r in targets:
        g = gold[r["qid"]]
        qtok = toks(g["q"])
        for gi, (gr, st) in enumerate(zip(g["groups"], r["groups"])):
            if st["union_rank"] is not None:
                continue  # 이 그룹은 잡혔음 (문항 내 다른 그룹이 미포착)
            eids = gr.get("lsh_eids") or []
            # A. 태그 존재/실질성
            trecs = [tags.get(e) for e in eids]
            tag_ok = any(t and tag_substance(t) >= 2 for t in trecs)
            # 조 텍스트 + 태그 문자열
            jo = jo_of(gr["c0"], gr["c1"])
            jtext = (jo or {}).get("text", "")
            tagstr = " ".join(json.dumps(t, ensure_ascii=False) for t in trecs if t)
            # B. 어휘 겹침
            overlap = qtok & toks(jtext + " " + tagstr + " " + (jo or {}).get("contract_scope", ""))
            # C. 청크 커버리지
            cov = any(c["char_start"] < gr["c1"] and c["char_end"] > gr["c0"] for c in chunks)
            if not tag_ok:
                c = "A.tag_missing"
            elif not overlap:
                c = "B.lex_mismatch"
            elif not cov:
                c = "C.chunk_missing"
            else:
                c = "D.rank_only"
            cause[c] += 1
            rows_out.append({"qid": r["qid"], "group": gi, "eids": eids, "cause": c,
                             "core": r["core"], "task_type": r["task_type"],
                             "jo": (jo or {}).get("element_id"), "contract": (jo or {}).get("contract_scope", "")[:40],
                             "q": g["q"][:60], "overlap": sorted(overlap)[:8],
                             "tag_substance": [tag_substance(t) if t else 0 for t in trecs]})

    n = len(rows_out)
    print(f"미포착 그룹 {n}개 원인 분해 (문항 {len(targets)}개):")
    for c, v in sorted(cause.items()):
        print(f"  {c}: {v} ({100*v/n:.0f}%)")
    # 특약(contract)별 상위 — 태깅 보강 대상 목록
    cc = collections.Counter(x["contract"] for x in rows_out if x["cause"] in ("A.tag_missing", "B.lex_mismatch"))
    print("\nA/B 원인 특약 상위:")
    for k, v in cc.most_common(12):
        print(f"  {v:2d}  {k}")
    json.dump({"cause": dict(cause), "rows": rows_out}, open(HERE / "out/diag_uncaptured.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("-> out/diag_uncaptured.json")


if __name__ == "__main__":
    main()
