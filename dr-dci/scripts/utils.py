"""공통 유틸리티"""

import re
import json
import asyncio
import aiohttp
from typing import Any

VLLM_URL = "http://localhost:8100/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-8B"
MAX_CONCURRENT = 32


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
