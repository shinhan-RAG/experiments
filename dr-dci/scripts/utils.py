"""공통 유틸리티"""

import re


def strip_thinking(content: str) -> str:
    """Qwen3 <think>...</think> 블록 제거"""
    if "<think>" in content:
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
    return content.strip()
