"""claude CLI 호출 공용 헬퍼 (shinhan-tos QA 생성용)."""
import json
import re
import shutil
import subprocess

MODEL = "claude-haiku-4-5-20251001"
CLAUDE = shutil.which("claude") or "claude"
JSON_RE = re.compile(r"\{.*\}", re.S)


def call_claude_json(prompt: str, timeout: int = 180):
    """claude -p 호출 → 응답에서 첫 JSON 오브젝트 파싱. 실패 시 None."""
    try:
        p = subprocess.run(
            [CLAUDE, "-p", "--model", MODEL],
            input=prompt, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    m = JSON_RE.search(p.stdout or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def locate_spans(chunk_text: str, evidences):
    """evidence 문자열을 청크 원문에서 찾아 char offset 부여. 못 찾으면 제외.
    (build_shinhan_qa_v2.py의 검증 로직 이식)"""
    spans = []
    for ev in evidences or []:
        ev = (ev or "").strip()
        if len(ev) < 8:
            continue
        idx = chunk_text.find(ev)
        if idx != -1:
            spans.append({"text": ev, "char_start": idx, "char_end": idx + len(ev),
                          "verbatim": True})
            continue
        # 공백 정규화 후 재시도 (verbatim이지만 개행·공백 차이만 있는 경우)
        norm = " ".join(ev.split())
        comp = " ".join(chunk_text.split())
        if norm in comp:
            spans.append({"text": ev, "char_start": None, "char_end": None,
                          "verbatim": True})
    return spans
