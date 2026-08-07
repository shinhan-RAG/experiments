"""
7단계: Metadata 생성 (Schema-driven, 병렬 처리)
- 데이터셋별 YAML 스키마에 따라 구조화된 메타데이터 추출
- controlled vocabulary로 일관성 보장
- vLLM guided_json으로 출력 형식 강제
- async 32 concurrent
"""

import json
import re
import yaml
from pathlib import Path
from utils import run_batch_llm, parse_llm_content, load_corpus_subset

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "metadata"
SCHEMA_DIR = Path(__file__).parent.parent / "config" / "metadata_schemas"


def load_schema(dataset: str) -> dict:
    """데이터셋별 metadata 스키마 로드"""
    schema_path = SCHEMA_DIR / f"{dataset}.yaml"
    with open(schema_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_system_prompt(schema: dict) -> str:
    """스키마에서 시스템 프롬프트 생성"""
    domain = schema["domain"]
    desc = schema["description"]
    fields = schema["fields"]

    field_descriptions = []
    for field_name, field_def in fields.items():
        if field_name == "entities":
            categories = field_def["item_schema"]["category"]["values"]
            max_items = field_def.get("max_items", 5)
            field_descriptions.append(
                f'  "{field_name}": [list of objects {{"name": "...", "category": "..."}}]  '
                f'// max {max_items}, category must be one of: {categories}'
            )
        elif field_def["type"] == "enum":
            values = field_def["values"]
            nullable = field_def.get("nullable", False)
            null_note = " or null" if nullable else ""
            field_descriptions.append(
                f'  "{field_name}": "..."  // one of: {values}{null_note}'
            )
        elif field_def["type"] == "integer":
            nullable = field_def.get("nullable", False)
            range_ = field_def.get("range", [])
            null_note = " or null" if nullable else ""
            range_note = f" (range: {range_[0]}-{range_[1]})" if range_ else ""
            field_descriptions.append(
                f'  "{field_name}": integer{range_note}{null_note}'
            )

    fields_str = "\n".join(field_descriptions)

    return f"""Extract structured metadata from the given {domain} document.
Domain: {desc}

Return JSON with exactly these fields:
{{
{fields_str}
}}

IMPORTANT:
- Use ONLY the allowed values listed above for enum fields.
- Return ONLY valid JSON, no explanation."""


def build_json_schema(schema: dict) -> dict:
    """vLLM guided_json용 JSON Schema 생성"""
    fields = schema["fields"]
    properties = {}
    required = []

    for field_name, field_def in fields.items():
        if field_name == "entities":
            categories = field_def["item_schema"]["category"]["values"]
            properties[field_name] = {
                "type": "array",
                "maxItems": field_def.get("max_items", 5),
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "category": {"type": "string", "enum": categories},
                    },
                    "required": ["name", "category"],
                },
            }
            required.append(field_name)
        elif field_def["type"] == "enum":
            nullable = field_def.get("nullable", False)
            if nullable:
                properties[field_name] = {
                    "anyOf": [
                        {"type": "string", "enum": field_def["values"]},
                        {"type": "null"},
                    ]
                }
            else:
                properties[field_name] = {
                    "type": "string",
                    "enum": field_def["values"],
                }
                required.append(field_name)
        elif field_def["type"] == "integer":
            nullable = field_def.get("nullable", False)
            int_schema = {"type": "integer"}
            range_ = field_def.get("range")
            if range_:
                int_schema["minimum"] = range_[0]
                int_schema["maximum"] = range_[1]
            if nullable:
                properties[field_name] = {
                    "anyOf": [int_schema, {"type": "null"}]
                }
            else:
                properties[field_name] = int_schema
                required.append(field_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def build_default_result(schema: dict) -> dict:
    """스키마 기반 기본값 생성"""
    fields = schema["fields"]
    result = {}
    for field_name, field_def in fields.items():
        if field_name == "entities":
            result[field_name] = []
        elif field_def.get("nullable", False):
            result[field_name] = None
        elif field_def["type"] == "enum":
            result[field_name] = field_def["values"][-1]  # 'other'
        elif field_def["type"] == "integer":
            result[field_name] = None
    return result


def build_metadata(dataset: str = "trec-covid", subset_size: int = 10_000):
    print(f"=== Building Metadata for {dataset} ({subset_size // 1000}K) ===")

    # 이미 존재하면 스킵
    out_path = OUTPUT_DIR / f"{dataset}_{subset_size // 1000}k.json"
    if out_path.exists():
        print(f"  Already exists: {out_path}, skipping.")
        return

    # 스키마 로드
    schema = load_schema(dataset)
    system_prompt = build_system_prompt(schema)
    json_schema = build_json_schema(schema)
    print(f"  Domain: {schema['domain']}")
    print(f"  Fields: {list(schema['fields'].keys())}")

    # 서브셋 corpus 로드 (aihub=parent-id 서브셋, BEIR=doc-id 서브셋)
    corpus_subset = load_corpus_subset(DATA_DIR, dataset, subset_size)
    print(f"  Docs to process: {len(corpus_subset)}")

    # 프롬프트 준비
    prompts = []
    for doc in corpus_subset:
        title = doc.get("title", "")
        text = doc.get("text", "")[:400]
        prompts.append({
            "id": doc["_id"],
            "text": f"Title: {title}\nText: {text}",
        })

    # 배치 처리
    BATCH_SIZE = 500
    results = {}
    failed = 0
    for batch_start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[batch_start:batch_start + BATCH_SIZE]
        raw_results = run_batch_llm(
            batch, system_prompt, max_tokens=300
        )

        for prompt_item, raw in zip(batch, raw_results):
            if raw is None:
                results[prompt_item["id"]] = build_default_result(schema)
                failed += 1
            else:
                try:
                    parsed = json.loads(parse_llm_content(raw))
                    results[prompt_item["id"]] = parsed
                except json.JSONDecodeError:
                    results[prompt_item["id"]] = build_default_result(schema)
                    failed += 1

        done = min(batch_start + BATCH_SIZE, len(prompts))
        print(f"  Progress: {done}/{len(prompts)} (failed: {failed})")

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 통계
    print(f"\n  Total: {len(results)} docs (failed: {failed})")

    for field_name, field_def in schema["fields"].items():
        if field_def["type"] == "enum":
            counts = {}
            for meta in results.values():
                if meta is None:
                    continue
                val = meta.get(field_name)
                if val is not None:
                    counts[val] = counts.get(val, 0) + 1
            if counts:
                print(f"\n  {field_name} distribution:")
                for val, count in sorted(counts.items(), key=lambda x: -x[1])[:8]:
                    print(f"    {val}: {count}")

    if "entities" in schema["fields"]:
        cat_counts = {}
        for meta in results.values():
            if meta is None:
                continue
            for ent in meta.get("entities", []):
                cat = ent.get("category", "other")
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        if cat_counts:
            print(f"\n  entity category distribution:")
            for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
                print(f"    {cat}: {count}")

    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    dataset = args[0] if args else "trec-covid"
    sizes = [20_000, 50_000, 110_000] if "--all" in sys.argv else [20_000]
    for size in sizes:
        build_metadata(dataset, size)
