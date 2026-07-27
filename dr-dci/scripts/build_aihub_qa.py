"""Generate an explicitly non-confirmatory AIHub text-chunk smoke QA fixture."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request


BASE = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = BASE / "data" / "aihub" / "qa"
DEFAULT_PARENTS = BASE / "data" / "aihub" / "parents.jsonl"
DEFAULT_META = BASE / "data" / "aihub" / "smoke" / "parent_meta.jsonl"
PROMPT = """당신은 법률·의료 문서 검색 평가셋을 만드는 전문가입니다.
아래는 한 문서(일부)입니다. 이 내용에 근거해 실무자가 물어볼 구체적 질문 1개를 만들고,
정답과 정답 근거가 되는 원문 문장을 그대로 인용하세요.

규칙:
- 질문은 이 문서를 봐야만 답할 수 있을 만큼 구체적으로 작성합니다.
- evidence는 원문에 그대로 존재하는 8자 이상의 문자열만 허용합니다.
- 정보성이 없으면 usable=false로 답합니다.
- 아래 JSON 객체만 출력합니다.
{"usable":true,"question":"...","answer":"...","evidence":["원문 인용"]}

[도메인] {domain}
[분야] {category}
[문서]
{text}
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: dict) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_response(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("generator response must be an object")
    if type(value.get("usable")) is not bool:
        raise ValueError("generator response usable must be boolean")
    if not value["usable"]:
        return value
    if not isinstance(value.get("question"), str) or not value["question"].strip():
        raise ValueError("usable response requires question")
    if not isinstance(value.get("answer"), str):
        raise ValueError("usable response requires answer string")
    evidence = value.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("usable response requires evidence list")
    if any(not isinstance(item, str) or len(item.strip()) < 8 for item in evidence):
        raise ValueError("every evidence item must be a string of at least eight chars")
    return value


def call_llm(
    text: str,
    domain: str,
    category: str,
    *,
    api_key: str,
    model: str,
    timeout_seconds: int,
    max_retries: int,
) -> tuple[dict | None, dict]:
    prompt = (
        PROMPT.replace("{domain}", domain)
        .replace("{category}", category[:80])
        .replace("{text}", text[:4000])
    )
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 500,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}],
    }
    request_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=request_bytes,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    receipt = {
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "attempts": 0,
        "status": "failed",
    }
    for attempt in range(max_retries + 1):
        receipt["attempts"] = attempt + 1
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                response_bytes = response.read()
            receipt["response_sha256"] = hashlib.sha256(response_bytes).hexdigest()
            payload = json.loads(response_bytes)
            content = payload["choices"][0]["message"]["content"]
            result = _validate_response(json.loads(content))
            receipt["status"] = "complete"
            receipt["system_fingerprint"] = payload.get("system_fingerprint")
            return result, receipt
        except Exception as error:
            receipt["error_type"] = type(error).__name__
            if attempt >= max_retries:
                return None, receipt
            time.sleep(2 * (attempt + 1))
    return None, receipt


def _without_whitespace(raw: str) -> tuple[str, list[int]]:
    characters = []
    source_indices = []
    for index, character in enumerate(raw):
        if not character.isspace():
            characters.append(character)
            source_indices.append(index)
    return "".join(characters), source_indices


def locate_all(raw: str, evidence: str) -> list[tuple[int, int]]:
    evidence = (evidence or "").strip()
    if len(evidence) < 8:
        return []
    occurrences = []
    offset = 0
    while True:
        found = raw.find(evidence, offset)
        if found < 0:
            break
        occurrences.append((found, found + len(evidence)))
        offset = found + 1
    if occurrences:
        return occurrences

    normalized_raw, source_indices = _without_whitespace(raw)
    normalized_evidence = re.sub(r"\s+", "", evidence)
    if len(normalized_evidence) < 8:
        return []
    offset = 0
    while True:
        found = normalized_raw.find(normalized_evidence, offset)
        if found < 0:
            break
        occurrences.append(
            (
                source_indices[found],
                source_indices[found + len(normalized_evidence) - 1] + 1,
            )
        )
        offset = found + 1
    return occurrences


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _allocation(meta: list[dict], count: int, seed: int) -> dict[tuple[str, str], int]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in meta:
        grouped[(str(row["domain"]), str(row["category"]))].append(str(row["parent_id"]))
    if count < 1 or count > sum(len(values) for values in grouped.values()):
        raise ValueError("query count must fit the available parent population")
    ordered_keys = sorted(
        grouped,
        key=lambda key: (
            -len(grouped[key]),
            hashlib.sha256(f"{seed}\0{key}".encode()).hexdigest(),
        ),
    )
    allocation = {key: 0 for key in ordered_keys}
    for key in ordered_keys[: min(count, len(ordered_keys))]:
        allocation[key] = 1
    remaining = count - sum(allocation.values())
    while remaining:
        candidates = [
            key for key in ordered_keys
            if allocation[key] < len(grouped[key])
        ]
        key = max(
            candidates,
            key=lambda item: (
                len(grouped[item]) / (allocation[item] + 1),
                item,
            ),
        )
        allocation[key] += 1
        remaining -= 1
    return allocation


def build(
    *,
    parents_path: Path,
    meta_path: Path,
    output_dir: Path,
    model: str,
    generator_revision: str,
    api_key: str,
    seed: int,
    query_count: int,
    timeout_seconds: int,
    max_retries: int,
) -> dict:
    parents = _load_jsonl(parents_path)
    metadata = _load_jsonl(meta_path)
    fulltext = {str(row["parent_id"]): str(row["text"]) for row in parents}
    if len(fulltext) != len(parents):
        raise ValueError("duplicate parent ID in parents source")
    meta_by_parent = {str(row["parent_id"]): row for row in metadata}
    if len(meta_by_parent) != len(metadata) or set(meta_by_parent) != set(fulltext):
        raise ValueError("parent metadata and parent text IDs differ")

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for parent, row in meta_by_parent.items():
        grouped[(str(row["domain"]), str(row["category"]))].append(parent)
    allocation = _allocation(metadata, query_count, seed)
    for key in grouped:
        grouped[key].sort(
            key=lambda parent: hashlib.sha256(
                f"{seed}\0{parent}".encode("utf-8")
            ).hexdigest()
        )

    queries = []
    qa_meta = []
    receipts = []
    for key in sorted(allocation, key=lambda item: (-len(grouped[item]), item)):
        needed = allocation[key]
        made = 0
        for parent in grouped[key]:
            if made >= needed:
                break
            row = meta_by_parent[parent]
            raw = fulltext[parent]
            response, receipt = call_llm(
                raw,
                str(row["domain"]),
                str(row["category"]),
                api_key=api_key,
                model=model,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
            receipt["parent_id_sha256"] = hashlib.sha256(parent.encode()).hexdigest()
            receipts.append(receipt)
            if not response or not response.get("usable"):
                continue
            spans = []
            for evidence in response["evidence"]:
                for start, end in locate_all(raw, evidence):
                    spans.append(
                        {
                            "text": evidence.strip(),
                            "char_start": start,
                            "char_end": end,
                            "text_sha256": hashlib.sha256(
                                raw[start:end].encode("utf-8")
                            ).hexdigest(),
                        }
                    )
            if not spans:
                continue
            qid = str(len(queries) + 1)
            question = response["question"].strip()
            queries.append({"_id": qid, "text": question})
            qa_meta.append(
                {
                    "qid": qid,
                    "question": question,
                    "answer": response["answer"].strip(),
                    "parent_id": parent,
                    "domain": row["domain"],
                    "category": row["category"],
                    "supporting_spans": spans,
                    "review_status": "generated_unreviewed",
                }
            )
            made += 1
    if len(queries) != query_count:
        raise RuntimeError(
            f"generated {len(queries)} usable queries, expected {query_count}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("queries.jsonl", queries),
        ("qa_meta.jsonl", qa_meta),
        ("generation_receipts.jsonl", receipts),
    ):
        with (output_dir / filename).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "contract": "aihub-generated-text-qa-v2",
        "evaluation_track": "s0_exploratory_chunk_smoke",
        "confirmatory_gate_status": "not_advanced",
        "review_status": "generated_unreviewed",
        "model": model,
        "generator_revision": generator_revision,
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "seed": seed,
        "query_count": query_count,
        "request_controls": {
            "temperature": 0,
            "max_tokens": 500,
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
        },
        "inputs": {
            "parents_sha256": sha256_file(parents_path),
            "parent_meta_sha256": sha256_file(meta_path),
        },
    }
    manifest["contract_sha256"] = canonical_json_sha256(manifest)
    (output_dir / "generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parents", type=Path, default=DEFAULT_PARENTS)
    parser.add_argument("--parent-meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", required=True)
    parser.add_argument("--generator-revision", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-count", type=int, default=50)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise SystemExit(f"missing API key environment variable: {args.api_key_env}")
    manifest = build(
        parents_path=args.parents,
        meta_path=args.parent_meta,
        output_dir=args.output_dir,
        model=args.model,
        generator_revision=args.generator_revision,
        api_key=api_key,
        seed=args.seed,
        query_count=args.query_count,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    print(
        f"queries={manifest['query_count']} "
        f"contract_sha256={manifest['contract_sha256']}"
    )


if __name__ == "__main__":
    main()
