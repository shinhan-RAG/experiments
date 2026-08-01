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
    """질의별 top-2 L1 + confidence 분류.

    paper-mixed 실험(8/2)에서 라우팅 오류 26~28%가 routed recall 손실 전액이었다.
    주 오류원이 인접 카테고리 혼동(인문학↔예술체육학)이므로 2순위 후보와
    확신도를 함께 저장해, 소비측이 top-2 필터·신뢰도 폴백을 선택할 수 있게 한다.
    """
    system = f"""You are a search query router for a document retrieval system.
Classify each Korean query into L1 categories.

L1 categories: {', '.join(l1_list)}
Category hints (L2 topics per L1): {json.dumps(l2_map, ensure_ascii=False)}

Respond in JSON:
{{"L1": "<most likely category>",
 "L1_second": "<second most likely category (repeat L1 if none plausible)>",
 "confidence": <0.0-1.0 probability that L1 is correct>}}"""
    guided = {"type": "object",
              "properties": {
                  "L1": {"type": "string", "enum": l1_list},
                  "L1_second": {"type": "string", "enum": l1_list},
                  "confidence": {"type": "number", "minimum": 0, "maximum": 1},
              },
              "required": ["L1", "L1_second", "confidence"]}
    prompts = [{"id": str(q["_id"]), "text": f"Query: {q.get('title') or q['text']}"}
               for q in queries]
    raw = run_batch_llm(prompts, system, max_tokens=80, guided_json=guided)

    routing = {}
    fallback = 0
    for q, r in zip(queries, raw):
        l1, l1_second, conf = None, None, None
        if r:
            try:
                parsed = json.loads(parse_llm_content(r))
                l1 = parsed.get("L1")
                l1_second = parsed.get("L1_second")
                conf = parsed.get("confidence")
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
        if l1 not in l1_list:
            l1 = "Other" if "Other" in l1_list else l1_list[0]
            fallback += 1
        if l1_second not in l1_list:
            l1_second = l1
        if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
            conf = 1.0  # confidence 미출력 시 폴백이 발동하지 않도록 보수적으로
        top2 = [l1] if l1_second == l1 else [l1, l1_second]
        routing[str(q["_id"])] = {"L1": l1, "L1_top2": top2,
                                  "confidence": round(float(conf), 3)}
    if fallback:
        print(f"  [!] 분류 실패 폴백 {fallback}건")
    return routing


def diagnose(routing: dict, qrels: list, taxonomy: dict,
             parent_map: dict = None) -> dict:
    """라우팅 정확도: routed L1 == gold 문서의 taxonomy L1 인 질의 비율.

    aihub 계열은 qrels corpus-id가 parent 문서 ID이고 taxonomy는 청크 키라서,
    parent→L1 집합(parent_l1)을 경유해 대조한다."""
    parent_l1: dict[str, set] = {}
    if parent_map:
        for chunk_id, tax in taxonomy.items():
            if isinstance(tax, dict) and tax.get("L1"):
                pid = parent_map.get(chunk_id)
                if pid:
                    parent_l1.setdefault(pid, set()).add(tax["L1"])

    gold_by_q = {}
    for e in qrels:
        if e.get("score", 0) >= 1:
            gold_by_q.setdefault(str(e["query-id"]), set()).add(str(e["corpus-id"]))
    n_eval, n_correct, n_correct_top2, missing_tax = 0, 0, 0, 0
    for qid, route in routing.items():
        gold_l1s = set()
        for gid in gold_by_q.get(qid, ()):
            tax = taxonomy.get(gid)
            if isinstance(tax, dict) and tax.get("L1"):
                gold_l1s.add(tax["L1"])
            gold_l1s |= parent_l1.get(gid, set())
        if not gold_l1s:
            missing_tax += 1
            continue
        n_eval += 1
        if route["L1"] in gold_l1s:
            n_correct += 1
        top2 = route.get("L1_top2") or [route["L1"]]
        if any(l1 in gold_l1s for l1 in top2):
            n_correct_top2 += 1
    return {
        "routing_accuracy": round(n_correct / n_eval, 4) if n_eval else None,
        "routing_accuracy_top2": round(n_correct_top2 / n_eval, 4) if n_eval else None,
        "evaluated_queries": n_eval,
        "queries_without_gold_taxonomy": missing_tax,
        "routed_distribution": dict(Counter(r["L1"] for r in routing.values())),
        "confidence_mean": round(
            sum(r.get("confidence", 1.0) for r in routing.values()) / len(routing), 3
        ) if routing else None,
    }


def build(dataset: str, subset_size: int) -> None:
    size_key = f"{subset_size // 1000}k"
    print(f"=== query routing: {dataset} ({size_key}) ===")

    schema_path = SCHEMA_DIR / f"{dataset}.yaml"
    with open(schema_path, encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    l1_list = schema["L1"]

    ds_dir = dataset_dir(DATA_DIR, dataset)
    # run_experiment.load_queries 와 동일한 우선순위: 층화 표본이 있으면 그것이 질의 집합
    query_file = ds_dir / "agent_queries_50.jsonl"
    if query_file.exists():
        print(f"  질의: 층화 표본 {query_file.name}")
    else:
        query_file = ds_dir / "queries.jsonl"
    queries = load_jsonl(query_file)
    qrels = load_jsonl(ds_dir / "qrels.jsonl")

    routing = classify_queries(queries, l1_list, schema.get("L2", {}))

    tax_path = DATA_DIR / "taxonomy" / f"{dataset}_{size_key}.json"
    diagnostics = {}
    if tax_path.exists():
        with open(tax_path, encoding="utf-8") as f:
            taxonomy = json.load(f)
        # 청크형 corpus면 chunk→parent 맵으로 parent 수준 gold와 대조
        parent_map = {}
        corpus_path = ds_dir / "corpus.jsonl"
        if corpus_path.exists():
            with open(corpus_path, encoding="utf-8") as f:
                for line in f:
                    doc = json.loads(line)
                    pid = doc.get("parent_id")
                    if pid and pid != doc["_id"]:
                        parent_map[str(doc["_id"])] = str(pid)
        diagnostics = diagnose(routing, qrels, taxonomy, parent_map)
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
              f"(top2: {diagnostics['routing_accuracy_top2']}, "
              f"conf_mean: {diagnostics['confidence_mean']}, "
              f"n={diagnostics['evaluated_queries']})")
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
