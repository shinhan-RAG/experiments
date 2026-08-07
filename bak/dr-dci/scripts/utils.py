"""공통 유틸리티"""

import re
import json
import asyncio
import aiohttp
from pathlib import Path
from typing import Any

VLLM_URL = "http://localhost:8100/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-8B"
MAX_CONCURRENT = 32


# aihub 계열(법률)은 BEIR 표준 위치(data/raw/<dataset>)가 아닌
# data/aihub/<variant>에 빌드된다. run_experiment.dataset_dir 와 동일하게 유지.
AIHUB_DATASET_DIRS = {
    "aihub-full": ("aihub", "full"),
    "aihub-smoke20k": ("aihub", "smoke20k"),
}


def dataset_dir(data_dir: Path, dataset: str) -> Path:
    """dataset 이름 → 데이터 디렉토리 (run_experiment.dataset_dir 미러)."""
    if dataset in AIHUB_DATASET_DIRS:
        group, variant = AIHUB_DATASET_DIRS[dataset]
        return data_dir / group / variant
    return data_dir / "raw" / dataset


def load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_corpus_subset(data_dir: Path, dataset: str, subset_size: int | None) -> list:
    """서브셋 corpus 로드 (run_experiment.load_corpus 미러).

    청크형(법률) corpus는 서브셋을 parent 문서 ID 목록({size}_parent_ids.json)으로,
    BEIR corpus는 doc-id 목록(subsets/<dataset>/<size>.json)으로 정의한다.
    """
    ddir = dataset_dir(data_dir, dataset)
    corpus = load_jsonl(ddir / "corpus.jsonl")
    if not subset_size:
        return corpus
    size_key = f"{subset_size // 1000}k"
    parent_subset = ddir / f"{size_key}_parent_ids.json"
    if parent_subset.exists():
        with open(parent_subset, encoding="utf-8") as f:
            parent_ids = set(json.load(f))
        return [doc for doc in corpus
                if doc.get("parent_id", doc["_id"]) in parent_ids]
    subset_path = data_dir / "subsets" / dataset / f"{size_key}.json"
    with open(subset_path, encoding="utf-8") as f:
        doc_ids = set(json.load(f)["doc_ids"])
    return [doc for doc in corpus if doc["_id"] in doc_ids]


def strip_thinking(content: str) -> str:
    """Qwen3 <think>...</think> 블록 제거"""
    if "<think>" in content:
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
    return content.strip()


def parse_llm_content(content: str) -> str:
    """LLM 출력에서 JSON 추출 전처리"""
    content = strip_thinking(content)
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return content.strip()


async def _call_single(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    payload: dict,
    max_retries: int = 3,
) -> str | None:
    """단일 비동기 LLM 호출"""
    for attempt in range(max_retries):
        try:
            async with semaphore:
                async with session.post(
                    VLLM_URL, json=payload, timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status != 200:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1)
                            continue
                        return None
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"].strip()
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
            else:
                return None
    return None


async def batch_llm_calls(
    prompts: list[dict],
    system: str,
    max_tokens: int = 100,
    guided_json: dict | None = None,
    concurrency: int = MAX_CONCURRENT,
) -> list[str | None]:
    """배치 비동기 LLM 호출

    Args:
        prompts: [{"id": ..., "text": ...}, ...]
        system: 시스템 프롬프트
        max_tokens: 최대 출력 토큰
        guided_json: vLLM guided_json 스키마 (optional)
        concurrency: 동시 요청 수

    Returns:
        list of raw content strings (or None for failures)
    """
    semaphore = asyncio.Semaphore(concurrency)

    payloads = []
    for p in prompts:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": p["text"]},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if guided_json:
            payload["guided_json"] = guided_json
        payloads.append(payload)

    async with aiohttp.ClientSession() as session:
        tasks = [
            _call_single(session, semaphore, payload)
            for payload in payloads
        ]
        results = await asyncio.gather(*tasks)

    return results


def run_batch_llm(
    prompts: list[dict],
    system: str,
    max_tokens: int = 100,
    guided_json: dict | None = None,
    concurrency: int = MAX_CONCURRENT,
) -> list[str | None]:
    """동기 래퍼 for batch_llm_calls"""
    return asyncio.run(
        batch_llm_calls(prompts, system, max_tokens, guided_json, concurrency)
    )
