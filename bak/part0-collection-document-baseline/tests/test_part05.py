"""Unit tests for Part 0.5 improvement conditions.

Run: python3 tests/test_part05.py --repo-root <experiments checkout>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import improve  # noqa: E402

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name)


def test_bm25f_rank_equivalence(repo_root):
    """Single field, w=1, b=0.75 must rank identically to gate2.BM25
    (gate2's (k1+1) numerator is a rank-preserving constant)."""
    sys.path.insert(0, str(repo_root / "tos-skeleton" / "frontmatter"))
    import gate2
    docs = ["보험금 지급 사유와 절차", "해약환급금 산정 기준 및 방법",
            "보험료 납입 면제 조건", "지급 절차 안내 문서",
            "특약 보험금 지급 기준"]
    base = gate2.BM25(docs)
    f = improve.BM25F([{"base": d} for d in docs], gate2.bigrams, ("base",))
    w, b = {"base": 1.0}, {"base": 0.75}
    for q in ("보험금 지급", "해약환급금 기준", "납입 면제"):
        r1 = base.rank(q, list(range(len(docs))))
        r2 = f.rank(q, list(range(len(docs))), w, b)
        check(f"bm25f rank-equiv '{q}'", r1 == r2)


def test_bm25f_weights():
    """Raising a field's weight must be able to flip the ranking toward it."""
    import re
    bigr = lambda s: [s2[i:i+2] for s2 in [re.sub(r"\s+", "", s)]
                      for i in range(len(s2) - 1)]
    docs = [{"base": "가나다라", "fm": "타파하자"},
            {"base": "마바사아", "fm": "가나다라"}]
    f = improve.BM25F(docs, bigr, ("base", "fm"))
    lo = f.rank("가나다라", [0, 1], {"base": 1.0, "fm": 0.01}, {"base": 0.75, "fm": 0.75})
    hi = f.rank("가나다라", [0, 1], {"base": 0.01, "fm": 5.0}, {"base": 0.75, "fm": 0.75})
    check("bm25f weight steers to base-field doc", lo[0] == 0)
    check("bm25f weight steers to fm-field doc", hi[0] == 1)


def test_rrf():
    fused = improve.rrf_fuse([[1, 2, 3], [3, 2, 1]], k=60)
    # symmetric: doc2 (rank2+rank2) beats doc1/doc3 (rank1+rank3)
    s1 = 1 / 61 + 1 / 63
    s2 = 1 / 62 + 1 / 62
    check("rrf symmetric middle wins", (fused[0] == 2) == (s2 > s1))
    fused2 = improve.rrf_fuse([[5, 6], [5, 6]], k=60)
    check("rrf agreement preserved", fused2 == [5, 6])


def test_cascade():
    class FakeFm:
        def rank(self, q, cands):
            return sorted(cands, reverse=True)
    out = improve.cascade_rank([10, 20, 30, 40], FakeFm(), "q", 2)
    check("cascade reranks head only", out == [20, 10, 30, 40])
    out2 = improve.cascade_rank([10, 20, 30, 40], FakeFm(), "q", 10)
    check("cascade K beyond length ok", out2 == [40, 30, 20, 10])


def test_router():
    cases = [
        ("참좋은치아보험 판매약관 보여줘", "identity"),
        ("신한건강보험 2024년 사업방법서", "identity"),
        ("테스트보험 최신 공시약관", "identity"),
        ("특별약정해지환급금 산정 내용이 포함된 문서를 찾아줘", "content"),
        ("보험금 지급 조건에 관한 내용이 있는 문서는?", "content"),
        ("해약환급금 산정 내용이 있는 최신 판매약관", "mixed"),
        ("고지의무 위반 내용이 있는 2023년 사업방법서", "mixed"),
    ]
    for q, want in cases:
        got = improve.route(q)
        check(f"router {want}: {q[:24]}…", got == want)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    args = ap.parse_args()
    test_bm25f_rank_equivalence(Path(args.repo_root).resolve())
    test_bm25f_weights()
    test_rrf()
    test_cascade()
    test_router()
    test_verify_partition()
    test_content_tokens()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)




def test_verify_partition():
    fm = ["", "가나 해약환급금은 조항", "", "해약환급금은 내용"]
    r = [0, 1, 2, 3]
    out = improve.verify_partition(r, fm, ["해약환급금은"], depth=3)
    check("verify stable partition", out == [1, 0, 2, 3])
    out2 = improve.verify_partition(r, fm, ["없는토큰같은것"], depth=3)
    check("verify empty-pass fallback", out2 == r)
    out3 = improve.verify_partition(r, fm, [], depth=3)
    check("verify no-tokens fallback", out3 == r)
    out4 = improve.verify_partition(r, fm, ["해약환급금은"], depth=4)
    check("verify depth covers tail", out4 == [1, 3, 0, 2])


def test_content_tokens():
    t = improve.content_tokens("특별약정해지환급금은 계약해지시점의 내용이 있는 최신 판매약관")
    check("content tokens keep phrase", t == ["특별약정해지환급금은", "계약해지시점의"])
    t2 = improve.content_tokens("참좋은치아보험 2024년 사업방법서 보여줘")
    check("content tokens drop identity words", "사업방법서" not in t2 and "2024년" not in t2)


if __name__ == "__main__":
    main()
