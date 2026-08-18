#!/usr/bin/env python3
"""에이전틱 하이브리드 RRF 검색 실험 러너. claude -p 서브프로세스 + tools_rrf.py CLI 도구.

데이터는 전부 이 폴더(hybrid-enrich/out) 안에서만 읽는다. 예전 구현이 참조하던
bak/metajson-v6/meta-search-v4 는 이 머신에 존재하지 않아 인덱스가 0개였다.

골드셋: out/gold_train.jsonl (173) / out/gold_test.jsonl (67)
  - build_unified_gold.py --split {train,test} 가 생성
  - gold_spans 는 coord_space="raw", 즉 원본 MD 문자 오프셋이다.
    elements/chunks 의 char_start/char_end 와 같은 좌표계여야 채점이 성립한다.
    (예전 골드는 공백 정규화 좌표라 최대 56,667자 어긋나 있었고, 그 상태의
     recall 수치는 전부 무의미하다.)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 코드는 noah/ 에, 데이터는 그 상위 hybrid-enrich/out 에 있다.
HERE = Path(__file__).resolve().parent   # noah/ — 코드 위치
BASE = HERE.parent                       # hybrid-enrich/ — 데이터 루트
OUT = BASE / "out"

MODEL = "claude-sonnet-5"
# 에이전트가 어떤 인터프리터 이름으로 호출하든 허용한다. Windows에서 python3 는
# Microsoft Store 스텁으로 연결돼 실행되지 않으므로 프롬프트는 python 을 쓰게
# 지시하지만, 패턴 자체는 넓게 열어 두어 권한 거부로 조용히 죽는 일을 막는다.
ALLOWED_TOOLS = ",".join([
    "Bash(python tools_rrf.py:*)",
    "Bash(py tools_rrf.py:*)",
    "Bash(python.exe tools_rrf.py:*)",
])
DISALLOWED_TOOLS = "Read,Write,Edit,Grep,Glob,WebFetch,WebSearch,Task"
def _resolve_claude() -> str:
    """claude 실행 파일을 찾는다. .CMD 셈이 아니라 claude.exe 를 직접 쓴다.

    shutil.which("claude") 는 PATHEXT 때문에 claude.CMD 를 돌려주는데, 이 셈은
    내부에서 cmd.exe 를 한 겹 더 띄우고 `"%dp0%\\node_modules\\...\\claude.exe"` 를
    호출한다. 실험 도중 claude-code 가 npm 으로 업데이트되면서 exe 가 교체되는
    동안 이 호출이 깨져 692세션 중 56세션(28%)이 '실행 파일을 찾을 수 없음'으로
    실패했다. exe 를 직접 지목하면 셈 계층과 그 실패 모드가 사라지고, 프로세스도
    하나 덜 뜬다."""
    override = os.environ.get("CLAUDE_BIN")
    if override and Path(override).exists():
        return override
    cmd_path = shutil.which("claude")
    if cmd_path:
        exe = (Path(cmd_path).parent / "node_modules" / "@anthropic-ai"
               / "claude-code" / "bin" / "claude.exe")
        if exe.exists():
            return str(exe)
    for cand in (shutil.which("claude.exe"), cmd_path, "claude"):
        if cand:
            return cand
    return "claude"


CLAUDE_BIN = _resolve_claude()

PROMPT_PATH = HERE / "retrieval_prompt_rrf.md"
PROMPT_TEMPLATE = (PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.exists()
                   else "너는 보험 약관 검색 에이전트다.\n{{TOOLS}}\n{{STEP3}}")

_TOOL_LINES = {
    "hybrid": '- `python tools_rrf.py hybrid "query"` — **hybrid search**. Internally fuses\n'
              '  sparse (BM25) and dense (bge-m3 embedding) channels with RRF. Returns chunk\n'
              '  IDs (`c*`) with per-channel provenance. This is your primary tool.',
    "slot":   '- `python tools_rrf.py slot "query"` — element tag file search over structured\n'
              '  slots (계약/주체/역할/한정/스키마). Returns element IDs (`e*`).',
    "slotand": '- `python tools_rrf.py slotand \'{"contract":"..."}\'` — AND-slot element search.\n'
               '  Returns element IDs (`e*`).',
    "grep":   '- `python tools_rrf.py grep "pattern"` — regex search over the raw text.',
    "read":   '- `python tools_rrf.py read "id"` — read the full text of one chunk (`c*`)\n'
              '  or element (`e*`).',
}
_STEP3 = {
    "slot": "3. **Cross-check with `slot`** when the question is about a specific 특약, 지급사유,\n"
            "   면제조건, 지급금액, or 기간 — the element index is finer-grained than chunks and\n"
            "   often pinpoints the exact clause.",
    "slotand": "3. **Cross-check with `slotand`** when the question names a specific 특약 or\n"
               "   조항 — supply the slot filters explicitly.",
    None: "3. **Cross-check with a different `grep` pattern** — try the 특약 name, a 조 title,\n"
          "   or a distinctive 금액/기간 string to catch what ranked search missed.",
}


def build_system_prompt(arm: str) -> str:
    """arm 이 실제로 쓸 수 있는 도구만 프롬프트에 적는다.

    도구 노출은 두 곳에서 막아야 한다. 여기(문서)와 tools_rrf.py CLI(집행).
    프롬프트에만 적고 CLI 를 안 막으면 에이전트가 문서에 없는 명령을 추론해
    호출할 수 있고, CLI 만 막으면 거부 응답으로 호출 예산을 낭비한다."""
    from tools_rrf import ARMS
    cfg = ARMS.get(arm, ARMS["V6V9"])
    names = ["hybrid"]
    if cfg["clm"]:
        names.append("slot")
    if cfg.get("slot_and", False):
        names.append("slotand")
    names += ["grep", "read"]
    tools = "\n".join(_TOOL_LINES[n] for n in names)
    step3 = _STEP3["slot"] if cfg["clm"] else (
        _STEP3["slotand"] if cfg.get("slot_and", False) else _STEP3[None])
    return PROMPT_TEMPLATE.replace("{{TOOLS}}", tools).replace("{{STEP3}}", step3)


def prompt_hash(arm: str) -> str:
    return hashlib.sha256(build_system_prompt(arm).encode("utf-8")).hexdigest()[:12]

SESSION_ROOT = OUT / "sessions_rrf"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_final(text: str) -> dict:
    text = text or ""
    for m in reversed(list(re.finditer(r'\{[^{}]*"ranked_chunk_ids"\s*:\s*\[.*?\][^{}]*\}', text, re.S))):
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue
    for m in reversed(list(re.finditer(r'\{[^{}]*"ranked_chunk_ids"[^{}]*\}', text))):
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            continue
    # 최후 수단: 본문에 등장한 c*/e* ID를 등장 순서대로 추출한다.
    # repaired 엘리먼트 49,929건 중 23,808건이 e00024__r000 형태의 접미사 ID다.
    # 접미사를 옵션으로 받지 않으면 절반가량을 조용히 버리게 된다.
    ids = re.findall(r"\b[ce]\d{5}(?:__r\d{3})?\b", text)
    if ids:
        seen: list[str] = []
        for uid in ids:
            if uid not in seen:
                seen.append(uid)
        return {"status": "ranked", "ranked_chunk_ids": seen[:10],
                "final_reason": "extracted from agent text (no JSON)"}
    return {"status": "error", "ranked_chunk_ids": [], "final_reason": "final JSON parse failed"}


def _kill_tree(proc: subprocess.Popen) -> None:
    """Windows에서 timeout 시 자식만 죽으면 claude.CMD가 띄운 node와 진행 중인
    tools_rrf.py 손자 프로세스가 고아로 남는다. taskkill /T 로 트리째 정리한다."""
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, timeout=30)
        else:
            proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass


def _count_calls(calls_f: Path) -> int:
    if not calls_f.exists():
        return 0
    return sum(1 for l in calls_f.read_text(encoding="utf-8").splitlines() if l.strip())


def _invoke(question: str, sdir: Path, env: dict, timeout: int, insist: bool,
            arm: str = "V6V9") -> dict:
    """claude -p 한 번 호출. insist=True 면 첫 검색 명령을 문자 그대로 지시한다."""
    q_escaped = question[:80].replace('"', '\\"')
    if insist:
        user_msg = (
            f"먼저 이 Bash 명령을 그대로 실행하라: python tools_rrf.py hybrid \"{q_escaped}\"\n"
            f"그 결과를 보고 추가 검색과 read 로 검증한 뒤 랭킹하라 "
            f"(시스템 프롬프트에 적힌 명령만 쓸 것).\n"
            f"검색 없이 답을 지어내면 실패로 처리된다. 마지막 메시지는 JSON 하나만.\n\n"
            f"질문: {question}"
        )
    else:
        user_msg = (
            f"질문: {question}\n\n"
            f"위 질문의 근거가 되는 약관 청크/엘리먼트를 찾아라. "
            f"시스템 프롬프트의 4단계 검색 전략을 따르고, 도구를 최소 3회 이상 사용하라. "
            f"마지막 메시지는 JSON 하나만 출력한다."
        )

    cmd = [
        CLAUDE_BIN, "-p", user_msg,
        "--model", MODEL,
        # --append-system-prompt 를 쓰면 Claude Code 기본 페르소나 위에 얹히는
        # 형태가 되어, 에이전트가 검색을 하지 않고 질문에 직접 답해 버린다
        # (스모크에서 4세션 전부 도구 0회 호출로 실패했다). 반드시 교체해야 한다.
        "--system-prompt", build_system_prompt(arm),
        "--allowedTools", ALLOWED_TOOLS,
        "--disallowedTools", DISALLOWED_TOOLS,
        "--output-format", "text",
        "--max-turns", "12",
    ]
    suffix = "_retry" if insist else ""
    try:
        # Popen 자체가 실패할 수 있다. claude.exe 는 304MB 이미지라 동시 실행이
        # 많으면 Windows 커밋 한도를 넘겨 즉시 예외가 난다(16워커에서 실측).
        # 일시적 자원 고갈이므로 지수 백오프로 몇 번 기다려 준다.
        proc = None
        for attempt in range(5):
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding="utf-8", errors="replace",
                    cwd=str(HERE), env=env,
                )
                break
            except OSError as exc:
                if attempt == 4:
                    return {"status": "error", "ranked_chunk_ids": [],
                            "final_reason": f"spawn_failed({type(exc).__name__}): {exc}"[:300]}
                time.sleep(2 ** attempt + random.random() * 2)
        try:
            raw, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            (sdir / f"raw_stderr{suffix}.txt").write_text("timeout", encoding="utf-8")
            return {"status": "error", "ranked_chunk_ids": [], "final_reason": "timeout"}
        (sdir / f"raw_stdout{suffix}.txt").write_text(raw or "", encoding="utf-8")
        if err:
            (sdir / f"raw_stderr{suffix}.txt").write_text(err, encoding="utf-8")
        # 실행 자체가 실패한 경우(런처 교체 중 exe 부재 등)를 "모델이 검색 없이
        # 답했다"와 구분한다. 둘 다 도구 0회로 끝나지만 원인과 대처가 다르다.
        if proc.returncode != 0 and not (raw or "").strip():
            return {"status": "error", "ranked_chunk_ids": [],
                    "final_reason": f"launch_failed rc={proc.returncode}: {(err or '')[:160]}"}
        # --output-format text 이므로 stdout 은 평문이다. 예전 구현은 여기서
        # json.loads 를 시도해 놓고 text 포맷을 요청하는 모순이 있었다.
        return parse_final(raw)
    except Exception as exc:
        return {"status": "error", "ranked_chunk_ids": [], "final_reason": repr(exc)[:300]}


def run_one(qid: str, question: str, arm: str, timeout: int, rep: int = 1) -> dict:
    # 반복(rep)마다 세션 디렉터리를 분리한다. 같은 경로를 쓰면 두 번째 반복이
    # 첫 번째 결과를 그대로 재사용해 버려 분산을 전혀 못 재게 된다.
    sdir = SESSION_ROOT / arm / f"rep{rep}" / qid
    sdir.mkdir(parents=True, exist_ok=True)
    done_f = sdir / "result.json"
    if done_f.exists():
        return json.loads(done_f.read_text(encoding="utf-8"))

    calls_f = sdir / "calls.jsonl"
    if calls_f.exists():
        calls_f.unlink()

    env = dict(os.environ, ARM=arm, SESSION_DIR=str(sdir), PYTHONIOENCODING="utf-8")
    t0 = time.time()
    final = _invoke(question, sdir, env, timeout, insist=False, arm=arm)

    calls_n = _count_calls(calls_f)
    # 도구를 한 번도 안 쓴 세션은 모델이 사전지식으로 답을 지어낸 것이다. 검색
    # 실험에서 이런 응답은 결과가 아니라 오염이므로, 첫 명령을 문자 그대로
    # 박아 넣어 한 번 더 시도한다.
    if calls_n == 0:
        final = _invoke(question, sdir, env, timeout, insist=True, arm=arm)
        calls_n = _count_calls(calls_f)
        if calls_n == 0:
            # 진짜 원인(spawn_failed/launch_failed/timeout)이 있으면 그것을 남긴다.
            # 이걸 뭉뚱그려 덮어쓰는 바람에 1,189건의 자원 고갈이 "모델이 검색 없이
            # 답했다"로 기록돼 원인 파악이 늦어졌다.
            reason = final.get("final_reason", "")
            if not any(k in reason for k in ("spawn_failed", "launch_failed", "timeout")):
                reason = "no tool calls after retry (모델이 검색 없이 답변)"
            final = {"status": "error", "ranked_chunk_ids": [], "final_reason": reason}

    calls: list[dict] = []
    if calls_f.exists():
        calls = [json.loads(l) for l in calls_f.read_text(encoding="utf-8").splitlines() if l.strip()]

    result = {
        "qid": qid, "question": question, "arm": arm, "rep": rep,
        "status": final.get("status", "error"),
        "ranked_chunk_ids": (final.get("ranked_chunk_ids") or [])[:10],
        "final_reason": final.get("final_reason", ""),
        "n_tool_calls": len(calls), "tool_calls": calls,
        "model": MODEL, "prompt_hash": prompt_hash(arm),
        "duration_ms": int((time.time() - t0) * 1000),
    }
    done_f.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


def first_rank(row: dict, gold: dict, spans: dict) -> int | None:
    """반환된 ID 순서대로 훑어, gold_span 과 문자 구간이 겹치는 첫 순위를 돌려준다.
    spans 와 gold_spans 는 반드시 같은 좌표계(raw)여야 한다."""
    for rank, uid in enumerate(row.get("ranked_chunk_ids") or [], 1):
        sp = spans.get(uid)
        if sp and any(sp[0] < end and sp[1] > start for start, end in gold["gold_spans"]):
            return rank
    return None


def metrics(rows: list[dict], gold_by: dict, spans: dict, arms: list[str]) -> dict:
    """arms 는 "BASE#1" 처럼 arm#rep 키를 받는다."""
    out = {}
    for arm in arms:
        arm_rows = {r["qid"]: r for r in rows if f'{r["arm"]}#{r.get("rep",1)}' == arm}
        ranks = [first_rank(arm_rows.get(qid, {}), g, spans) for qid, g in gold_by.items()]
        n = max(len(ranks), 1)
        answered = [q for q in gold_by if q in arm_rows]
        out[arm] = {
            "n": n,
            "R@1": round(sum(r == 1 for r in ranks) / n, 4),
            "R@5": round(sum(r is not None and r <= 5 for r in ranks) / n, 4),
            "R@10": round(sum(r is not None and r <= 10 for r in ranks) / n, 4),
            "MRR@10": round(sum(1 / r if r and r <= 10 else 0 for r in ranks) / n, 4),
            "errors": sum(arm_rows.get(qid, {}).get("status") == "error" for qid in gold_by),
            "avg_calls": round(sum(arm_rows.get(q, {}).get("n_tool_calls", 0)
                                   for q in answered) / max(len(answered), 1), 2),
        }
    return out


def print_table(done: int, total: int, result: dict):
    print(f"\n== {done}/{total} 문항 완료 ==", flush=True)
    print(f"{'Arm':<12} {'R@1':>5} {'R@5':>5} {'R@10':>5} {'MRR':>6} {'err':>4} {'calls':>6}", flush=True)
    for arm, m in result.items():
        print(f"{arm:<12} {m['R@1']:>5.3f} {m['R@5']:>5.3f} {m['R@10']:>5.3f} "
              f"{m['MRR@10']:>6.3f} {m['errors']:>4} {m['avg_calls']:>6.2f}", flush=True)


def load_spans() -> dict[str, tuple[int, int]]:
    """chunk_id(c*) + element_id(e*) 의 raw 좌표 스팬.

    엘리먼트는 P섹션 코퍼스(elements_psection.jsonl, 5,408건, 중앙값 336자) 하나만
    읽는다. 구 elements.jsonl / elements_repaired_v1.jsonl 은 같은 e00000 형태의
    ID를 쓰면서 스팬이 다르므로, 함께 로드하면 서로 덮어써 채점이 조용히 틀어진다.
    구 코퍼스(중앙값 24자)는 등록 실험에서 태그를 끈 조건보다도 낮았던 입도라
    검색 대상에서 제외한다."""
    spans: dict[str, tuple[int, int]] = {}
    for path, key in (
        (OUT / "chunks.jsonl", "chunk_id"),
        (OUT / "elements_psection.jsonl", "element_id"),
    ):
        if not path.exists():
            raise SystemExit(f"[FATAL] 스팬 소스 없음: {path}")
        for row in load_jsonl(path):
            spans[row[key]] = (row["char_start"], row["char_end"])
    return spans


def resolve_path(raw: str) -> Path:
    """상대경로는 데이터 루트(hybrid-enrich) 기준으로 해석한다.

    코드는 noah/ 에 있고 실행도 noah/ 에서 하므로, `out/gold_train.jsonl` 을
    그대로 넘기면 noah/out/ 을 찾아 실패한다. 절대경로와 실제로 존재하는
    상대경로는 그대로 두고, 그 외에만 BASE 를 앞에 붙인다."""
    q = Path(raw)
    if q.is_absolute() or q.exists():
        return q
    return BASE / raw


def purge_sessions(root: Path, include_zero_calls: bool = False) -> dict:
    """재실행할 세션의 result.json 을 지운다.

    러너는 result.json 이 있는 세션을 건너뛰므로, 지우는 것이 곧 '이 세션만 다시
    돌려라'는 뜻이 된다. 성공 세션은 그대로 두므로 이미 쓴 비용은 보존된다.
    raw_stdout 등 진단 파일은 남겨 실패 분석에 쓸 수 있게 한다."""
    import shutil as _shutil
    stat = {"removed": 0, "error": 0, "zero": 0, "kept": 0}
    for rp in list(root.glob("*/*/result.json")) + list(root.glob("*/*/*/result.json")):
        try:
            row = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:
            rp.unlink(missing_ok=True)
            stat["removed"] += 1
            continue
        is_err = row.get("status") == "error"
        is_zero = include_zero_calls and row.get("n_tool_calls", 0) == 0
        if is_err or is_zero:
            # 실패 근거를 남기기 위해 세션 디렉터리째 지우지 않고 진단 파일은 보존,
            # 대신 이전 시도의 흔적은 failed/ 아래로 옮긴다.
            keep = rp.parent / "prev_attempt"
            keep.mkdir(exist_ok=True)
            for f in list(rp.parent.glob("raw_*.txt")) + [rp]:
                try:
                    _shutil.move(str(f), str(keep / f.name))
                except Exception:
                    pass
            (rp.parent / "calls.jsonl").unlink(missing_ok=True)
            (rp.parent / "channel_stats.jsonl").unlink(missing_ok=True)
            stat["removed"] += 1
            stat["error"] += int(is_err)
            stat["zero"] += int(is_zero and not is_err)
        else:
            stat["kept"] += 1
    return stat


def preflight(arms: list[str], min_dense_coverage: float = 0.95) -> None:
    """실험 시작 전에 각 arm 의 채널이 실제로 살아 있는지 확인한다.

    dense 채널은 ollama 가 죽어도 조용히 빠지고 note 만 남기므로(설계상 graceful
    degradation), 점검 없이 돌리면 hybrid 실험이 BM25 단독 실험으로 바뀐 채
    끝까지 진행된다. 실제로 실행 도중 ollama 가 내려간 적이 있다. 여기서 막는다."""
    sys.path.insert(0, str(HERE))
    from tools_rrf import ARMS, build_or_load_bm25, embed_query, OUT as T_OUT

    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"[FATAL] 알 수 없는 arm: {unknown}  (가능: {list(ARMS)})")

    for a in arms:
        p = build_system_prompt(a)
        exposed = [t for t in ("hybrid", "slot", "slotand", "grep", "read")
                   if f"tools_rrf.py {t}" in p]
        print(f"  {a:<10} 도구={'/'.join(exposed):<28} prompt={prompt_hash(a)}")

    needs_dense = False
    corpus_n: dict[str, int] = {}
    for arm in arms:
        for kind, view in ARMS[arm]["channels"]:
            if kind == "bm25":
                idx, src = build_or_load_bm25(view)
                if idx is None:
                    raise SystemExit(f"[FATAL] 뷰 없음: out/view_{view}.jsonl ({arm}/bm25)")
                corpus_n[view] = idx.n
                print(f"  bm25/{view:<5} OK  n={idx.n} ({src})")
    for arm in arms:
        for kind, view in ARMS[arm]["channels"]:
            if kind != "dense":
                continue
            needs_dense = True
            npy = T_OUT / f"vec_{view}.npy"
            ids = T_OUT / f"vec_{view}_ids.json"
            if not npy.exists() or not ids.exists():
                raise SystemExit(f"[FATAL] 덴스 벡터 없음: {npy.name} ({arm}/dense)\n"
                                 f"  먼저 실행: python embed_local.py --views {view}")
            n_ids = len(json.loads(ids.read_text(encoding="utf-8")))
            total = corpus_n.get(view, n_ids)
            cov = n_ids / max(total, 1)
            # 벡터가 코퍼스 일부만 덮으면 dense 채널이 그 일부 안에서만 후보를
            # 내놓는다. 에러가 안 나므로 눈치채기 어렵고, 결과는 하이브리드가
            # 아니라 "BM25 전체 + dense 일부"가 된다.
            if cov < min_dense_coverage:
                raise SystemExit(
                    f"[FATAL] 덴스 커버리지 부족: {view} {n_ids}/{total} ({cov:.1%})\n"
                    f"  dense 채널이 코퍼스 일부만 검색하게 되어 arm 비교가 무효가 된다.\n"
                    f"  먼저 실행: python embed_local.py --views {view}   (--limit 없이 전량)")
            print(f"  dense/{view:<4} OK  n={n_ids}/{total} ({cov:.1%})")

    if needs_dense:
        try:
            embed_query("사전 점검")
        except Exception as exc:
            raise SystemExit(f"[FATAL] ollama 질의 임베딩 실패: {exc}\n"
                             f"  dense 채널이 조용히 빠진 채로 실험이 진행되면 결과가 무효다.\n"
                             f"  ollama 를 기동한 뒤 다시 실행하라.")
        print("  ollama  OK  (질의 임베딩 정상)")

    if any(ARMS[a]["clm"] for a in arms):
        from clm_filesearch import CLMFileSearch
        n = CLMFileSearch().search("보험료 납입면제", limit=5)["n_candidates"]
        if n == 0:
            raise SystemExit("[FATAL] CLM 후보 0건 — 태그/엘리먼트 파일을 확인하라.")
        print(f"  CLM     OK  (샘플 질의 후보 {n}건)")


def check_coord_space(gold: list[dict]) -> None:
    """골드가 raw 좌표인지 확인한다. 정규화 좌표 골드로 채점하면 수치가 전부
    무의미해지므로, 조용히 틀리느니 여기서 멈추는 편이 낫다."""
    bad = [g["qid"] for g in gold if g.get("coord_space") != "raw"]
    if bad:
        raise SystemExit(
            f"[FATAL] 골드 좌표계가 raw 가 아니다 ({len(bad)}건, 예: {bad[:3]}). "
            f"build_unified_gold.py 를 다시 실행해 coord_space=raw 골드를 만들어라."
        )


def main():
    global SESSION_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(OUT / "gold_train.jsonl"))
    ap.add_argument("--arms", default="BASE,V6V9")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--out", default=str(OUT / "agentic_rrf_results.jsonl"))
    ap.add_argument("--session-root", default=str(SESSION_ROOT))
    ap.add_argument("--core-only", dest="core_only", action="store_true", default=False)
    ap.add_argument("--reps", type=int, default=1,
                    help="arm 당 반복 실행 횟수. 2 이상이면 실행 간 분산을 잴 수 있다")
    ap.add_argument("--retry-failed", action="store_true",
                    help="실패 세션만 삭제해 재실행 대상으로 만든다 (성공분은 재사용)")
    ap.add_argument("--retry-zero-calls", action="store_true",
                    help="--retry-failed 에 더해, 도구를 한 번도 안 쓴 세션도 재실행한다")
    args = ap.parse_args()

    SESSION_ROOT = resolve_path(args.session_root)

    gold_path = resolve_path(args.gold)
    if not gold_path.exists():
        raise SystemExit(f"[FATAL] 골드셋 없음: {gold_path}\n"
                         f"  먼저 실행: python build_unified_gold.py --split train")
    gold = load_jsonl(gold_path)
    check_coord_space(gold)
    if args.core_only:
        gold = [g for g in gold if g.get("core_retrieval")]
    gold = [g for g in gold if g.get("gold_spans")]
    if args.limit:
        gold = gold[:args.limit]
    gold_by = {g["qid"]: g for g in gold}

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    spans = load_spans()

    print(f"골드셋: {gold_path}  ({len(gold)} 문항, core_only={args.core_only}, coord_space=raw)")
    print(f"스팬 로드: {len(spans)}개 unit (chunks.jsonl + elements_psection.jsonl)")
    print(f"arms: {', '.join(arms)}  workers={args.workers}  timeout={args.timeout}s")
    print(f"세션: {SESSION_ROOT}")
    print("사전 점검:")
    preflight(arms)

    SESSION_ROOT.mkdir(parents=True, exist_ok=True)

    if args.retry_failed or args.retry_zero_calls:
        purged = purge_sessions(SESSION_ROOT, include_zero_calls=args.retry_zero_calls)
        print(f"재실행 대상 정리: {purged['removed']}건 삭제 "
              f"(실패 {purged['error']}, 도구0회 {purged['zero']}), "
              f"{purged['kept']}건 재사용")

    results: list[dict] = []
    total = len(gold)
    for batch_start in range(0, total, args.batch_size):
        batch = gold[batch_start:batch_start + args.batch_size]
        jobs = [(g["qid"], g["question"], arm, rep)
                for g in batch for arm in arms for rep in range(1, args.reps + 1)]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_one, qid, q, arm, args.timeout, rep)
                    for qid, q, arm, rep in jobs]
            for fut in as_completed(futs):
                results.append(fut.result())
        done = min(batch_start + len(batch), total)
        keys = [f"{a}#{r}" for a in arms for r in range(1, args.reps + 1)]
        print_table(done, total, metrics(results, {g["qid"]: g for g in gold[:done]}, spans, keys))

    out_path = resolve_path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        for r in sorted(results, key=lambda x: (x["qid"], x["arm"], x.get("rep", 1))):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()
