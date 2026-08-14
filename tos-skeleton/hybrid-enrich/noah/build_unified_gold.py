#!/usr/bin/env python3
"""noah_qaset CSV를 250212 MD에 재앵커링하여 split별 gold set 생성.

출력: out/gold_train.jsonl / out/gold_test.jsonl
각 행: {qid, question, gold_spans, n_spans, task_type, core_retrieval, 사업구분, coord_space}
gold_spans: [[char_start, char_end], ...] — 출처 인용문의 MD 내 위치 (RAW 좌표계)

성능 노트: 문서(~2.7M자)를 CSV 행마다(그리고 인용문마다) 통째로 re.sub 로
재정규화하고 str.find 로 훑던 예전 구현은 사실상 O(n^2)에 가까워 수백 개
행에서도 타임아웃이 났다. 이번 버전은
  1) csv 모듈로 멀티라인 필드(출처)를 한 번에 올바르게 파싱하고
  2) 문서 정규화를 단 한 번만 수행하며
  3) n-gram(4자) 인덱스로 인용문의 앵커 위치를 O(1)에 가깝게 좁힌 뒤
     후보 위치만 직접 비교(verify)한다.

좌표계 노트(중요): 인용문 매칭은 공백에 둔감해야 하므로 **정규화 문서**에서
수행한다. 그러나 out/elements_repaired_v1.jsonl, out/chunks.jsonl 의
char_start/char_end 는 모두 **RAW 문서** 좌표다. 두 좌표계는 이 문서에서
최대 56,667자까지 벌어지므로, 정규화 좌표를 그대로 기록하면 downstream
recall 채점이 서로 다른 좌표계를 비교하는 무의미한 수치가 된다.
따라서 정규화를 한 번의 문자 단위 패스로 수행하면서 norm2raw 매핑
(정규화 i번째 문자 -> 원본 인덱스)을 함께 만들고, find_span 결과를
RAW 좌표로 되돌린 뒤에 기록한다. coord_space 필드로 이를 명시한다.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 코드는 noah/ 에, 데이터는 그 상위 hybrid-enrich/out 에 있다.
HERE = Path(__file__).resolve().parent   # noah/ — 코드 위치
BASE = HERE.parent                       # hybrid-enrich/ — 데이터 루트
OUT = BASE / "out"
DOC = BASE / "noah_qaset" / "판매약관_(간편)신한통합건강보장보험 원(ONE)(무배당, 해약환급금 미지급형)_250212.md"

SPLITS = {
    "train": (BASE / "noah_qaset" / "정답셋_train_v3_single_chunk.csv", OUT / "gold_train.jsonl"),
    "test": (BASE / "noah_qaset" / "정답셋_test_v3_single_chunk.csv", OUT / "gold_test.jsonl"),
}

NGRAM = 4  # 앵커로 쓸 n-gram 길이. 한글 음절 조합 수가 많아 4자면 충분히 선별적.

_WS = re.compile(r"\s")  # re.sub(r"\s+", ...) 와 정확히 같은 공백 정의를 쓰기 위해 문자 단위로 재사용


def normalize(text: str) -> str:
    """짧은 문자열(인용문)용 정규화. 문서 전체에는 normalize_with_map 을 쓴다."""
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", text).strip())


def normalize_with_map(raw: str) -> tuple[str, list[int]]:
    """raw(이미 NFC) 를 한 번의 문자 단위 패스로 정규화하면서 norm2raw 매핑을 만든다.

    - 연속 공백은 " " 한 칸으로 접고, 그 공백은 런(run)의 **첫** 공백 문자의 raw 인덱스에 매핑한다.
    - 앞뒤 공백은 기존 .strip() 과 동일하게 제거한다(접힌 뒤에는 최대 한 칸씩만 남는다).
    - 접은 뒤 NFC 를 다시 걸지 않는다. raw 를 이미 NFC 로 만들어 두었고, 여기서
      재정규화하면 문자 수가 바뀌어 norm2raw 정렬이 깨질 수 있기 때문이다.
    """
    chars: list[str] = []
    norm2raw: list[int] = []
    in_ws = False
    ws_start = 0
    is_ws = _WS.match
    for i, ch in enumerate(raw):
        if is_ws(ch):
            if not in_ws:
                in_ws = True
                ws_start = i
            continue
        if in_ws:
            in_ws = False
            if chars:  # 선행 공백(strip 대상)은 버린다
                chars.append(" ")
                norm2raw.append(ws_start)
        chars.append(ch)
        norm2raw.append(i)
    # 후행 공백 런은 애초에 append 하지 않았으므로 별도 처리 불필요
    return "".join(chars), norm2raw


def build_ngram_index(text: str, n: int) -> dict[str, list[int]]:
    """text 내 모든 길이-n 부분문자열 -> 시작 위치 리스트. 한 번만 구축."""
    index: dict[str, list[int]] = {}
    for i in range(len(text) - n + 1):
        index.setdefault(text[i : i + n], []).append(i)
    return index


def _search_exact(doc: str, index: dict[str, list[int]], n: int, query: str) -> tuple[int, int] | None:
    """query 를 doc 안에서 정확히 찾는다. n-gram 인덱스로 후보 위치를 좁혀 검증한다."""
    length = len(query)
    if length < n:
        pos = doc.find(query)
        return (pos, pos + length) if pos >= 0 else None

    # query 여러 지점에서 앵커 n-gram을 뽑아, 후보가 가장 적은 것을 사용
    offsets = sorted({0, length // 4, length // 2, (3 * length) // 4, length - n})
    best_positions: list[int] | None = None
    best_offset = 0
    for off in offsets:
        if off < 0 or off + n > length:
            continue
        positions = index.get(query[off : off + n])
        if not positions:
            # 이 n-gram이 문서 어디에도 없으면 query 전체도 있을 수 없음
            return None
        if best_positions is None or len(positions) < len(best_positions):
            best_positions, best_offset = positions, off

    if not best_positions:
        return None

    doc_len = len(doc)
    for pos in best_positions:
        start = pos - best_offset
        end = start + length
        if start < 0 or end > doc_len:
            continue
        if doc[start:end] == query:
            return (start, end)
    return None


def find_span(doc: str, index: dict[str, list[int]], n: int, quote: str) -> tuple[int, int] | None:
    nq = normalize(quote)
    if len(nq) < 5:
        return None

    span = _search_exact(doc, index, n, nq)
    if span:
        return span

    max_trim = min(30, len(nq) // 2)
    for trim in range(5, max_trim):
        if len(nq) <= 2 * trim:
            break
        trimmed = nq[trim:-trim]
        span = _search_exact(doc, index, n, trimmed)
        if span:
            return span
    return None


def parse_evidence(raw: str) -> list[str]:
    """출처 필드: 줄마다 "인용문 | 페이지번호" 형식. 페이지번호는 버리고 인용문만 추출."""
    parts = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line or line.startswith("---"):
            continue
        m = re.match(r"^(.+?)\s*\|\s*\d+\s*$", line)
        if m:
            parts.append(m.group(1).strip())
        elif "|" in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            parts.extend(cells)
        else:
            parts.append(line)
    return [p for p in parts if len(p) >= 5]


def main():
    ap = argparse.ArgumentParser(description="noah_qaset split별 gold set 생성 (RAW 좌표계)")
    ap.add_argument("--split", required=True, choices=sorted(SPLITS), help="train 또는 test")
    ap.add_argument("--out", default=None, help="출력 경로 override (기본: out/gold_{split}.jsonl)")
    args = ap.parse_args()

    qa_csv, default_out = SPLITS[args.split]
    out_path = Path(args.out) if args.out else default_out

    doc_raw = unicodedata.normalize("NFC", DOC.read_text(encoding="utf-8").replace("\r\n", "\n"))
    doc_norm, norm2raw = normalize_with_map(doc_raw)
    print(f"문서 길이: raw {len(doc_raw)} / norm {len(doc_norm)}")
    index = build_ngram_index(doc_norm, NGRAM)

    rows = []
    with qa_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            no = (rec.get("no") or "").strip()
            biz = (rec.get("사업구분") or "").strip()
            question = (rec.get("질문") or "").strip()
            evidence_raw = rec.get("출처") or ""
            task_type = (rec.get("task_type") or "").strip()
            core = (rec.get("core_retrieval") or "").strip()
            qid = (rec.get("qid") or "").strip() or (f"v3-offline-{int(no):04d}" if no.isdigit() else no)

            gold_spans = []
            for ev in parse_evidence(evidence_raw):
                span = find_span(doc_norm, index, NGRAM, ev)
                if span:
                    ns, ne = span
                    # 정규화 좌표 -> RAW 좌표. 끝점은 마지막 문자의 raw 인덱스 + 1.
                    gold_spans.append([norm2raw[ns], norm2raw[ne - 1] + 1])

            rows.append({
                "qid": qid,
                "question": question,
                "gold_spans": gold_spans,
                "n_spans": len(gold_spans),
                "task_type": task_type,
                "core_retrieval": core.lower() == "true",
                "사업구분": biz,
                "coord_space": "raw",
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_core = sum(1 for r in rows if r["core_retrieval"])
    n_with_spans = sum(1 for r in rows if r["gold_spans"])
    rate = (n_with_spans / len(rows) * 100) if rows else 0.0
    print(f"[{args.split}] 총 {len(rows)} 문항 / core_retrieval {n_core} / 앵커 성공 {n_with_spans} ({rate:.1f}%)")
    print(f"출력: {out_path}")


if __name__ == "__main__":
    main()
