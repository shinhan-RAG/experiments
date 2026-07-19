"""
7단계: Metadata 생성 (Schema-driven)
- 데이터셋별 YAML 스키마에 따라 구조화된 메타데이터 추출
- controlled vocabulary로 일관성 보장
- vLLM guided_json으로 출력 형식 강제
- Pull pre-filter에 사용
"""

import json
import re
import requests
import yaml
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
SUBSET_DIR = DATA_DIR / "subsets"
OUTPUT_DIR = DATA_DIR / "metadata"
SCHEMA_DIR = Path(__file__).parent.parent / "config" / "metadata_schemas"

VLLM_URL = "http://localhost:8100/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-8B"


def load_schema(dataset: str) -> dict:
    """데이터셋별 metadata 스키마 로드"""
    schema_path = SCHEMA_DIR / f"{dataset}.yaml"
    with open(schema_path) as f:
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
                f'  "{field_name}": [list of objects {{\"name\": \"...\", \"category\": \"...\"}}]  '
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


def load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def call_llm(prompt: str, system: str, json_schema: dict, max_retries: int = 3) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
        "guided_json": json_schema,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(VLLM_URL, json=payload, timeout=30)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if "<think>" in content:
                content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except (json.JSONDecodeError, Exception):
            if attempt < max_retries - 1:
                import time
                time.sleep(1)
            else:
                return None


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

    # 서브셋 doc IDs
    subset_path = SUBSET_DIR / dataset / f"{subset_size // 1000}k.json"
    with open(subset_path) as f:
        subset_info = json.load(f)
    doc_ids = set(subset_info["doc_ids"])

    # corpus 로드
    corpus = load_jsonl(RAW_DIR / dataset / "corpus.jsonl")
    corpus_subset = [doc for doc in corpus if doc["_id"] in doc_ids]
    print(f"  Docs to process: {len(corpus_subset)}")

    results = {}
    failed = 0
    for i, doc in enumerate(corpus_subset):
        title = doc.get("title", "")
        text = doc.get("text", "")[:400]

        prompt = f"Title: {title}\nText: {text}"
        metadata = call_llm(prompt, system_prompt, json_schema)

        if metadata is None:
            metadata = build_default_result(schema)
            failed += 1

        results[doc["_id"]] = metadata

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(corpus_subset)} (failed: {failed})")

    # 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{dataset}_{subset_size // 1000}k.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 통계
    print(f"\n  Total: {len(results)} docs (failed: {failed})")

    # enum 필드 분포 출력
    for field_name, field_def in schema["fields"].items():
        if field_def["type"] == "enum":
            counts = {}
            for meta in results.values():
                val = meta.get(field_name)
                if val is not None:
                    counts[val] = counts.get(val, 0) + 1
            if counts:
                print(f"\n  {field_name} distribution:")
                for val, count in sorted(counts.items(), key=lambda x: -x[1])[:8]:
                    print(f"    {val}: {count}")

    # entity category 분포
    if "entities" in schema["fields"]:
        cat_counts = {}
        for meta in results.values():
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
    sizes = [20_000, 50_000, 110_000] if "--all" in sys.argv else [20_000]
    dataset = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "trec-covid"
    for size in sizes:
        build_metadata(dataset, size)
