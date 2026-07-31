"""질의 → taxonomy L1 라우팅 맵 생성 (Part 5 taxonomy_routed/boosted backend용).

- 질의 텍스트를 Qwen3-8B(:8100)로 L1 분류 (guided_json enum — 스키마 밖 값 원천 차단)
- 진단: gold 청크의 taxonomy L1과 대조한 라우팅 정확도를 함께 저장한다.
  recall 변화를 "라우팅이 틀려서"와 "라우팅은 맞았는데 못 찾아서"로 분해하는 근거.

출력: data/routing/<dataset>_<size>k.json
  {"dataset", "size_key", "l1_values", "routing": {qid: {"L1": ...}},
   "diagnostics": {"routing_accuracy", "routed_distribution", ...}}

  python scripts/build_query_routing.py aihub-full --size=20000
  python scripts/build_query_routing.py paper-ha --size=5000
"""
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils import dataset_dir, load_jsonl, parse_sizes, run_batch_llm, parse_llm_content  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data"
SCHEMA_DIR = Path(__file__).parent.parent / "config" / "taxonomy_schemas"
OUTPUT_DIR = DATA_DIR / "routing"


def classify_queries(queries: list, l1_list: list[str], l2_map: dict) -> dict:
    system = f"""You are a search query router for a document retrieval system.
Classify each Korean query into exactly one L1 category.

L1 categories: {', '.join(l1_list)}
Category hints (L2 topics per L1): {json.dumps(l2_map, ensure_ascii=False)}

Respond in JSON: {{"L1": "..."}}"""
    guided = {"type": "object",
              "properties": {"L1": {"type": "string", "enum": l1_list}},
              "required": ["L1"]}
    prompts = [{"id": str(q["_id"]), "text": f"Query: {q.get('title') or q['text']}"}
               for q in queries]
    raw = run_batch_llm(prompts, system, max_tokens=50, guided_json=guided)

    routing = {}
    fallback = 0
    for q, r in zip(queries, raw):
        l1 = None
        if r:
            try:
                l1 = json.loads(parse_llm_content(r)).get("L1")
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
        if l1 not in l1_list:
            l1 = "Other" if "Other" in l1_list else l1_list[0]
            fallback += 1
        routing[str(q["_id"])] = {"L1": l1}
    if fallback:
        print(f"  [!] 분류 실패 폴백 {fallback}건")
    return routing


def diagnose(routing: dict, qrels: list, taxonomy: dict) -> dict:
    """라우팅 정확도: routed L1 == gold 청크의 taxonomy L1 인 질의 비율."""
    gold_by_q = {}
    for e in qrels:
        if e.get("score", 0) >= 1:
            gold_by_q.setdefault(str(e["query-id"]), set()).add(str(e["corpus-id"]))
    n_eval, n_correct, missing_tax = 0, 0, 0
    for qid, route in routing.items():
        gold_l1s = set()
        for gid in gold_by_q.get(qid, ()):
            tax = taxonomy.get(gid)
            if isinstance(tax, dict) and tax.get("L1"):
                gold_l1s.add(tax["L1"])
        if not gold_l1s:
            missing_tax += 1
            continue
        n_eval += 1
        if route["L1"] in gold_l1s:
            n_correct += 1
    return {
        "routing_accuracy": round(n_correct / n_eval, 4) if n_eval else None,
        "evaluated_queries": n_eval,
        "queries_without_gold_taxonomy": missing_tax,
        "routed_distribution": dict(Counter(r["L1"] for r in routing.values())),
    }


def build(dataset: str, subset_size: int) -> None:
    size_key = f"{subset_size // 1000}k"
    print(f"=== query routing: {dataset} ({size_key}) ===")

    schema_path = SCHEMA_DIR / f"{dataset}.yaml"
    with open(schema_path, encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    l1_list = schema["L1"]

    ds_dir = dataset_dir(DATA_DIR, dataset)
    queries = load_jsonl(ds_dir / "queries.jsonl")
    qrels = load_jsonl(ds_dir / "qrels.jsonl")

    routing = classify_queries(queries, l1_list, schema.get("L2", {}))

    tax_path = DATA_DIR / "taxonomy" / f"{dataset}_{size_key}.json"
    diagnostics = {}
    if tax_path.exists():
        with open(tax_path, encoding="utf-8") as f:
            taxonomy = json.load(f)
        diagnostics = diagnose(routing, qrels, taxonomy)
    else:
        print(f"  [!] taxonomy 아티팩트 없음({tax_path}) — 라우팅 정확도 진단 생략")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{dataset}_{size_key}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset,
            "size_key": size_key,
            "l1_values": l1_list,
            "routing": routing,
            "diagnostics": diagnostics,
        }, f, ensure_ascii=False, indent=2)

    print(f"  queries: {len(routing)} -> {out_path}")
    if diagnostics:
        print(f"  routing_accuracy: {diagnostics['routing_accuracy']} "
              f"(n={diagnostics['evaluated_queries']})")
        print(f"  routed 분포: {diagnostics['routed_distribution']}")
        acc = diagnostics["routing_accuracy"]
        if acc is not None and acc < 0.8:
            print("  [!] 라우팅 정확도 0.8 미만 — taxonomy_routed의 recall 상한이 "
                  "이 값에 묶인다. 결과 해석 시 반드시 병기할 것.")


if __name__ == "__main__":
    argv = sys.argv[1:]
    args = [a for a in argv if not a.startswith("-")]
    if not args:
        raise SystemExit("usage: python scripts/build_query_routing.py <dataset> --size=N")
    for size in parse_sizes(argv):
        build(args[0], size)
