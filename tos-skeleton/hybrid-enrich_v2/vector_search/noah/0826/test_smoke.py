#!/usr/bin/env python3
"""0826 실험 스모크 테스트.

필수 데이터 존재 여부, 모듈 임포트, 검색 엔진 초기화, 기준선 검색을 검증.
LLM 태그 생성은 테스트하지 않는다(외부 의존).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FS = ROOT / "filesearch"

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def main():
    print("=== 0826 Smoke Test ===\n")

    print("[1] 필수 데이터")
    elements = FS / "out" / "elements_u3.jsonl"
    tags_rules = FS / "out" / "tags_u4_fact_rules.jsonl"
    jo = FS / "out" / "elements_u3jo.jsonl"
    gold = ROOT / "vector_search" / "noah" / "0824" / "v2" / "gold_v21_train213.jsonl"
    for p in (elements, tags_rules, jo, gold):
        check(p.name, p.exists(), f"not found: {p}")

    print("\n[2] 모듈 임포트")
    try:
        sys.path.insert(0, str(FS))
        from scoring import score
        check("scoring.score", True)
    except Exception as e:
        check("scoring.score", False, str(e))
    try:
        from tag_hybrid import TagHybridSearch, build_index, default_paths
        check("tag_hybrid.*", True)
    except Exception as e:
        check("tag_hybrid.*", False, str(e))
    try:
        from build_tags_llm_v3 import build_prompt, validate_tag, build_search_text
        check("build_tags_llm_v3.*", True)
    except Exception as e:
        check("build_tags_llm_v3.*", False, str(e))

    print("\n[3] arms.json")
    arms_path = HERE / "arms.json"
    check("arms.json exists", arms_path.exists())
    if arms_path.exists():
        arms = json.loads(arms_path.read_text(encoding="utf-8"))
        check("arms count >= 3", len(arms) >= 3, f"got {len(arms)}")
        check("baseline_rules in arms", "baseline_rules" in arms)
        check("llm_v3 in arms", "llm_v3" in arms)

    print("\n[4] 기준선 검색 엔진 초기화")
    if elements.exists() and tags_rules.exists() and jo.exists():
        try:
            from tag_hybrid import build_index as bi
            idx_dir = HERE / "out" / "idx_baseline_rules"
            if not (idx_dir / "tag_index.pkl").exists():
                print("  Building baseline index...", flush=True)
                bi(elements, tags_rules, jo, idx_dir)
            engine = TagHybridSearch(elements, tags_rules, jo, idx_dir)
            check("TagHybridSearch init", True)
        except Exception as e:
            check("TagHybridSearch init", False, str(e))
            engine = None
    else:
        engine = None
        check("TagHybridSearch init", False, "필수 데이터 없음")

    print("\n[5] 기준선 검색 실행")
    if engine and gold.exists():
        g = json.loads(gold.open(encoding="utf-8").readline())
        try:
            rows, slots, toks = engine.search(g["q"], weights={"clm": 1.0, "sparse": 1.0},
                                               limit=20, collapse_jo=True)
            check(f"search({g['qid'][:20]}...)", len(rows) > 0, f"got {len(rows)} rows")
            from scoring import score as sc
            ranked = []
            seen = set()
            for row in rows:
                jo_id = row["jo"]
                if jo_id and jo_id not in seen and jo_id in engine.jid:
                    seen.add(jo_id)
                    ranked.append(engine.jid[jo_id])
            metrics = sc(ranked, g["groups"], ks=(1, 5, 10))
            check(f"R@5={metrics['R@5']:.4f}", metrics["R@5"] >= 0, f"metrics={metrics}")
        except Exception as e:
            check("search", False, str(e))

    print("\n[6] LLM 태그 프롬프트 생성")
    try:
        from build_tags_llm_v3 import build_prompt
        contracts = sorted({json.loads(l)["contract_key"]
                           for l in open(tags_rules, encoding="utf-8")
                           if json.loads(l).get("contract_key")} - {""})
        prompt = build_prompt(contracts)
        check("prompt length > 500", len(prompt) > 500, f"len={len(prompt)}")
        check("prompt has [유효목록]", "[유효목록]" in prompt or "유효목록" in prompt)
        check(f"contracts count={len(contracts)}", len(contracts) > 10)
    except Exception as e:
        check("prompt build", False, str(e))

    print("\n[7] validate_tag 검증")
    try:
        from build_tags_llm_v3 import validate_tag
        valid_set = set(contracts) if 'contracts' in dir() else set()
        llm_out = {
            "i": 0, "contract_key": "없는특약", "subject_key": ["테스트"],
            "role": ["payment_trigger", "invalid_role"],
            "locator": {"article": "제5조", "article_title": "테스트", "section": "잘못됨"},
            "schema_tag": "invalid",
        }
        rule_tag = {"contract_key": "규칙계약", "schema_tag": "paragraph",
                    "locator": {"article": "제1조", "article_title": "", "section": ""}}
        result = validate_tag(llm_out, rule_tag, {"element_type": "paragraph"}, valid_set)
        check("contract_key fallback", result["contract_key"] == "규칙계약")
        check("invalid role filtered", "invalid_role" not in result["role"])
        check("schema_tag fallback", result["schema_tag"] == "paragraph")
        check("section fallback", result["locator"]["section"] == "")
    except Exception as e:
        check("validate_tag", False, str(e))

    print(f"\n=== Results: {PASS} passed, {FAIL} failed ===")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
