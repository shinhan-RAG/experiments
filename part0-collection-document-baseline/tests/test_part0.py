"""RED/GREEN tests for the Part 0 harness on a synthetic corpus.

Run:  python3 -m tests.test_part0 --repo-root <experiments checkout>
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import Config, Quotas  # noqa: E402
from harness.source_facts import guard, parse_date, parse_doc_type  # noqa: E402
from harness.score import metrics_for, verify_freeze  # noqa: E402
from harness.stats import mcnemar_exact  # noqa: E402
from harness import runner, validate_qa  # noqa: E402
from harness.source_facts import scan_universe  # noqa: E402

PASS = []
FAIL = []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name)


def expect_exit(name, fn):
    try:
        fn()
        check(name, False)
    except SystemExit:
        check(name, True)
    except ValueError:
        check(name, True)


# ---------- unit: parsers ----------

def test_parsers():
    check("doc_type strict single", parse_doc_type("사업방법서_x_240101.md") == "사업방법서")
    check("doc_type ambiguous none",
          parse_doc_type("판매약관_사업방법서.md") is None)
    check("doc_type absent none", parse_doc_type("NPFS060401MDB_1.md") is None)
    check("date 8digit", parse_date("x_20240101(3).md") == "20240101")
    check("date 6digit", parse_date("x_230404(2).md") == "20230404")
    check("date conflict none", parse_date("x_20240101_230404.md") is None)
    check("date 7digit ignored", parse_date("x_1701011.md") is None)


# ---------- unit: metrics (hand-computed) ----------

def test_metrics():
    m = metrics_for([3, 7, 9], {7})
    check("hit1 miss", m["hit1"] == 0)
    check("hit5 hit", m["hit5"] == 1)
    check("mrr rank2", abs(m["mrr10"] - 0.5) < 1e-12)
    check("ndcg single rank2",
          abs(m["ndcg10"] - (1 / math.log2(3))) < 1e-12)
    check("recall5 full", m["recall5"] == 1.0)
    m2 = metrics_for([2, 5, 1], {1, 2})
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    check("ndcg multi", abs(m2["ndcg10"] - dcg / idcg) < 1e-12)
    check("recall5 multi", m2["recall5"] == 1.0)
    m3 = metrics_for([], set([1]))
    check("empty ranking zero", m3["hit5"] == 0 and m3["mrr10"] == 0.0)


def test_mcnemar():
    check("mcnemar 1v3", abs(mcnemar_exact(1, 3) - 0.625) < 1e-12)
    check("mcnemar 0v8", abs(mcnemar_exact(0, 8) - 2 / 256) < 1e-12)
    check("mcnemar zero", mcnemar_exact(0, 0) == 1.0)


# ---------- RED: forbidden QA input ----------

def test_guard_red():
    expect_exit("RED guard rejects arm text",
                lambda: guard("… 최신판 정본 현행 …"))
    guard("평범한 원문 텍스트")
    check("guard accepts source text", True)


# ---------- synthetic corpus ----------

B_BODY = [f"제{i}항 일반 조건에 대한 서술 내용입니다 항목{i}" for i in range(1, 15)]


def _mk_doc(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_synthetic(root: Path) -> int:
    p1 = root / "테스트건강보험(무배당)"
    p2 = root / "샘플암보험플러스"
    shared = "특별약정해지환급금은 계약해지시점의 적립금액을 기준으로 산정한다"
    uniq24 = "고액암진단특약보험금은 진단확정일로부터 지급된다"
    uniq23 = "재해골절치료특약자금은 사고발생일 기준으로 계산한다"
    uniqA = "수술보장특약금액은 수술등급분류표에 따라 차등지급된다"
    head = ["# 문서", "개요 설명입니다"]
    filler = [f"부가 설명 문장 {i}번째 줄입니다" for i in range(20)]
    _mk_doc(p1 / "사업방법서_테스트건강보험_20240101.md",
            head + B_BODY + [shared, uniq24] + filler)
    _mk_doc(p1 / "사업방법서_테스트건강보험_20230101.md",
            head + B_BODY + [shared, uniq23] + filler)
    _mk_doc(p1 / "판매약관_테스트건강보험_20240101.md",
            head + B_BODY + [uniqA] + filler)
    _mk_doc(p2 / "사업방법서_샘플암보험_20220505.md",
            head + B_BODY + ["가입연령한도조건은 만십오세부터 적용한다"] + filler)
    tbl = ["| 구분항목 | 지급금액기준 |", "|---|---|"] + \
          [f"| 항목분류{i}종 | 기준금액{i}만원 지급조건 |" for i in range(1, 15)]
    _mk_doc(p2 / "공시약관_샘플암보험_20220505.md",
            head + tbl + [f"| 추가항목{i} | 추가내용{i} |" for i in range(20)])
    return 5


def synth_config(tmp: Path, repo_root: Path, seed=7) -> Config:
    quotas = Quotas(identity=2, content=3, mixed=1,
                    identity_types=(0, 2, 0), content_types=(0, 2, 1),
                    mixed_types=(0, 1, 0), max_per_doc=2, max_per_product=6,
                    gold_max_content=5)
    return Config(source_root=tmp / "src" / "parsed_md",
                  repo_root=repo_root,
                  work_dir=tmp / "work" / f"s{seed}",
                  results_dir=tmp / "results",
                  seed=seed, expected_docs=5, source_zip_sha256="synthetic",
                  quotas=quotas)


def test_synthetic(repo_root: Path):
    tmp = Path(tempfile.mkdtemp(prefix="part0_synth_"))
    try:
        n = build_synthetic(tmp / "src" / "parsed_md")
        cfg = synth_config(tmp, repo_root)
        out = runner.run(cfg)
        check("GREEN synthetic publishes", out["status"] == "published")
        target = Path(out["target"])
        res = json.loads((target / "results.json").read_text(encoding="utf-8"))
        qa = [json.loads(l) for l in open(target / "qa_500.jsonl",
                                          encoding="utf-8")]
        check("GREEN qa count", len(qa) == 6)
        check("GREEN suites",
              res["qa_report"]["suite_counts"] ==
              {"identity": 2, "content": 3, "mixed": 1})
        rows = [json.loads(l) for l in open(target / "per_query_rows.jsonl",
                                            encoding="utf-8")]
        check("GREEN per-query rows", len(rows) == len(qa) * 4)
        led = [json.loads(l) for l in open(target / "review_ledger.jsonl",
                                           encoding="utf-8")]
        check("GREEN ledger all accepted",
              len(led) == 6 and all(l["decision"] == "accepted" for l in led))
        idn_b00 = [r for r in rows if r["arm"] == "B00" and r["suite"] == "identity"]
        check("GREEN identity B00 finds by filename",
              all(r["hit5"] == 1 for r in idn_b00))

        # rerun same identity → reuse, bytes/mtime untouched
        before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size)
                  for p in target.iterdir()}
        out2 = runner.run(cfg)
        after = {p.name: (p.stat().st_mtime_ns, p.stat().st_size)
                 for p in target.iterdir()}
        check("GREEN rerun reuses", out2["status"] == "reused")
        check("GREEN rerun byte/mtime unchanged", before == after)

        # RED: different identity at same target
        cfg2 = synth_config(tmp, repo_root, seed=8)
        bad = Config(**{**cfg2.__dict__, "results_dir": cfg.results_dir})
        renamed = cfg.results_dir / f"part0_{bad.identity_sha()[:16]}"
        os.rename(target, renamed)
        try:
            expect_exit("RED different identity at same target",
                        lambda: runner.check_existing(bad, renamed))
        finally:
            os.rename(renamed, target)

        # RED: tamper after freeze
        qa_path = cfg.work_dir / "qa" / "qa_500.jsonl"
        orig = qa_path.read_bytes()
        qa_path.write_bytes(orig + b"\n")
        expect_exit("RED freeze tamper refused",
                    lambda: verify_freeze(cfg))
        qa_path.write_bytes(orig)

        # RED: corrupted gold caught by validator
        items = [json.loads(l) for l in open(qa_path, encoding="utf-8")]
        items[0]["gold_document_ids"] = ["doc-0000000000000000"]
        bad_qa = cfg.work_dir / "qa" / "qa_bad.jsonl"
        with open(bad_qa, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        universe = scan_universe(cfg.source_root)
        v = validate_qa.validate(cfg, universe, bad_qa)
        check("RED validator rejects corrupted gold",
              any(l["decision"] == "rejected" for l in v["ledger"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    args = ap.parse_args()
    test_parsers()
    test_metrics()
    test_mcnemar()
    test_guard_red()
    test_synthetic(Path(args.repo_root).resolve())
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
