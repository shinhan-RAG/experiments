"""Mechanical validation of every QA item + machine review ledger.

Re-derives gold from source alone and compares with the stored gold.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .config import Config
from .qa_build import RgGold, norm_q
from .source_facts import DocFacts

REVIEWER = "part0-harness mechanical validator + Claude (Fable 5), session 2026-08-04"


def validate(cfg: Config, universe: list[DocFacts], qa_path: Path) -> dict:
    by_id = {d.doc_id: d for d in universe}
    by_rel = {d.rel_path: d for d in universe}
    family = defaultdict(list)
    for d in universe:
        if d.doc_type and len(d.clean_name) >= 4:
            family[(d.clean_name, d.doc_type)].append(d)
    rg = RgGold(cfg.source_root)

    items = [json.loads(l) for l in open(qa_path, encoding="utf-8")]
    ledger = []
    failures = []
    seen_q = set()
    seen_id = set()
    per_doc = Counter()
    per_prod = Counter()
    evidence_groups = defaultdict(list)

    for item in items:
        checks = {}
        qa_id = item["qa_id"]
        checks["unique_qa_id"] = qa_id not in seen_id
        seen_id.add(qa_id)
        nq = norm_q(item["query"])
        checks["unique_query"] = nq not in seen_q
        seen_q.add(nq)

        gold = [by_id.get(g) for g in item["gold_document_ids"]]
        checks["gold_in_universe"] = all(gold)
        checks["gold_paths_match"] = (
            all(gold) and sorted(d.rel_path for d in gold) ==
            sorted(item["gold_source_paths"]))
        checks["target_in_gold"] = item["target_document_id"] in item["gold_document_ids"]
        checks["type_scored"] = item["document_structure_type"] in ("A", "B", "C")

        per_doc[item["target_document_id"]] += 1
        per_prod[item["product_id"]] += 1

        if item["suite"] == "identity":
            c = item["identity_constraints"]
            fam = family.get((c["product_name"], c["doc_type"]), [])
            if c["kind"] == "type":
                expect = fam
            elif c["kind"] == "year_type":
                expect = [d for d in fam if d.date and d.date[:4] == c["year"]]
            else:
                dated = [d for d in fam if d.date]
                latest = max((d.date for d in dated), default=None)
                expect = [d for d in dated if d.date == latest]
            checks["identity_gold_rederived"] = (
                sorted(d.doc_id for d in expect) == sorted(item["gold_document_ids"]))
            checks["type_keyword_in_gold_filenames"] = all(
                d and c["doc_type"] in d.filename for d in gold)
        else:
            toks = item["identity_constraints"]["evidence_tokens"]
            checks["no_product_name_leak"] = all(
                t not in (by_id[item["target_document_id"]].product_dir)
                for t in toks)
            files = set(rg.files_with(toks))
            if item["suite"] == "content":
                expect_paths = files
            else:
                c = item["identity_constraints"]
                cands = [by_rel[f] for f in files
                         if f in by_rel and by_rel[f].doc_type == c["doc_type"]]
                if c["kind"] == "year":
                    cands = [d for d in cands if d.date and d.date[:4] == c["year"]]
                elif c["kind"] == "latest":
                    grp = defaultdict(list)
                    for d in cands:
                        if d.date:
                            grp[d.product_id].append(d)
                    cands = []
                    for ds in grp.values():
                        m = max(d.date for d in ds)
                        cands.extend(d for d in ds if d.date == m)
                expect_paths = {d.rel_path for d in cands}
            checks["gold_rederived_from_source"] = (
                expect_paths == set(item["gold_source_paths"]))
            span_ok = True
            for ev in item["evidence"]:
                body = (cfg.source_root / ev["source_path"]).read_text(
                    encoding="utf-8", errors="ignore")
                ns_body = re.sub(r"\s+", "", body)
                ns_span = re.sub(r"\s+", "", ev["verbatim_span"])
                if ns_span not in ns_body:
                    span_ok = False
            checks["evidence_verbatim_in_gold"] = span_ok
            checks["evidence_covers_all_gold"] = (
                {e["document_id"] for e in item["evidence"]} ==
                set(item["gold_document_ids"]))
            evidence_groups[" ".join(toks)].append(qa_id)

        ok = all(checks.values())
        if not ok:
            failures.append({"qa_id": qa_id,
                             "failed": [k for k, v in checks.items() if not v]})
        ledger.append({"qa_id": qa_id, "reviewer": REVIEWER,
                       "checks": checks, "decision": "accepted" if ok else "rejected"})

    cap_fail = {
        "doc_cap": [d for d, c in per_doc.items() if c > cfg.quotas.max_per_doc],
        "product_cap": [p for p, c in per_prod.items()
                        if c > cfg.quotas.max_per_product],
    }
    competition = {k: v for k, v in evidence_groups.items() if len(v) > 1}
    summary = {
        "n_items": len(items),
        "n_accepted": sum(1 for l in ledger if l["decision"] == "accepted"),
        "failures": failures,
        "cap_violations": cap_fail,
        "evidence_competition_groups": competition,
        "suite_counts": dict(Counter(i["suite"] for i in items)),
        "type_counts": dict(Counter(i["document_structure_type"] for i in items)),
    }
    return {"ledger": ledger, "summary": summary}


def write_validation(cfg: Config, result: dict) -> dict:
    qa_dir = cfg.work_dir / "qa"
    ledger_path = qa_dir / "review_ledger.jsonl"
    with open(ledger_path, "w", encoding="utf-8") as f:
        for row in result["ledger"]:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary_path = qa_dir / "validation_summary.json"
    summary_path.write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8")
    s = result["summary"]
    if s["failures"] or s["cap_violations"]["doc_cap"] or s["cap_violations"]["product_cap"]:
        raise SystemExit(f"QA validation failed: {json.dumps(s['failures'][:5])}")
    if s["n_accepted"] != s["n_items"]:
        raise SystemExit("QA validation: not all items accepted")
    return {"ledger_path": str(ledger_path), "summary_path": str(summary_path)}
