#!/usr/bin/env python3
"""원(ONE) 판매약관 PDF 전 버전 → 정규화 텍스트 추출.

pdftotext(결정적)로 뽑은 뒤, 버전 간 diff에 방해되는 요소를 제거한다:
- 페이지 구분(form feed)과 페이지 번호 줄
- 반복되는 페이지 헤더/푸터 (전 페이지에서 동일하게 나타나는 짧은 줄)
- 연속 공백/빈 줄 정규화
"""
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

SRC = Path("/Users/seyoung/workspace/braincrew/project/shinhan/07_원본_파싱데이터/원ONE_상품문서")
OUT = Path("/Users/seyoung/workspace/braincrew/experiments/tos-skeleton/data/raw_text")

PAGE_NO = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")


def version_id(name: str) -> str:
    """파일명 → 짧은 버전 ID (예: 대면_250605, TM_251104, 260101)."""
    stem = re.sub(r"\.pdf$", "", name)
    m = re.search(r"_((?:대면|TM)_[\w.]+|\d{6}(?:_v[\d.]+)?|\d{8}|v[\d.]+)$", stem)
    return m.group(1) if m else stem[-20:]


def extract(pdf: Path) -> list[str]:
    txt = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), "-"],
        capture_output=True, text=True, check=True,
    ).stdout
    pages = txt.split("\f")
    # 반복 헤더/푸터 후보: 페이지 첫/끝 2줄에 등장하는 줄이 전체 페이지의 60% 이상에서 반복되면 제거
    edge_lines = Counter()
    for p in pages:
        lines = [l.strip() for l in p.splitlines() if l.strip()]
        for l in set(lines[:2] + lines[-2:]):
            if len(l) < 60:
                edge_lines[l] += 1
    repeated = {l for l, c in edge_lines.items() if c >= max(3, len(pages) * 0.6)}

    out = []
    for p in pages:
        for line in p.splitlines():
            s = unicodedata.normalize("NFC", line.rstrip())
            st = s.strip()
            if not st or PAGE_NO.match(st) or st in repeated:
                continue
            s = re.sub(r"[ \t]+", " ", st)
            out.append(s)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # macOS 파일명은 NFD라 한글 glob이 안 먹음 → 전체 나열 후 NFC로 비교
    pdfs = sorted(
        p for d in (SRC, SRC / "_versions") for p in d.glob("*.pdf")
        if unicodedata.normalize("NFC", p.name).startswith("판매약관")
    )
    for pdf in pdfs:
        vid = version_id(unicodedata.normalize("NFC", pdf.name))
        lines = extract(pdf)
        dest = OUT / f"{vid}.txt"
        dest.write_text("\n".join(lines), encoding="utf-8")
        print(f"{vid:24s} {len(lines):6d} lines  <- {pdf.name[:60]}")


if __name__ == "__main__":
    sys.exit(main())
