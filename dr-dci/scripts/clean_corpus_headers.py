"""청크형 판례 corpus의 중복 섹션 헤더를 in-place 제거한다.

AI Hub 판례 build가 각 섹션을 ``[판시사항]\n{원문}`` 으로 감쌌는데 원문이 이미
``【판시사항】`` 로 시작해 ``[판시사항]\n【판시사항】`` 중복이 ~50% 청크에 남았다.
전체 재빌드(재청킹) 없이, 이미 만들어진 corpus.jsonl의 각 청크 text에서 라벨 바로
아래 동일 full-width 헤더만 제거한다(청크 _id·parent_id·경계는 불변).

  python clean_corpus_headers.py data/aihub/full/corpus.jsonl

원본은 <corpus>.bak 로 백업하고, 임시파일→교체(atomic)로 안전하게 기록한다.
본문 내 정당한 【…】(인용/강조)는 건드리지 않는다.
"""
import json
import os
import re
import sys
from pathlib import Path

# [X]\n【X】  (라벨 바로 아래 동일 이름의 full-width 헤더) → [X] 만 남김
DUP_HEADER = re.compile(r'\[([^\]\n]{1,20})\]\n【\s*\1\s*】[ \t]*\n?')


def clean_text(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = DUP_HEADER.sub(r'[\1]\n', text)
    return text


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python clean_corpus_headers.py <corpus.jsonl>")
        return 1
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"not found: {path}")
        return 1

    backup = path.with_suffix(path.suffix + ".bak")
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = changed = 0
    with open(path, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            doc = json.loads(line)
            t = doc.get("text", "")
            c = clean_text(t)
            if c != t:
                doc["text"] = c
                changed += 1
            n += 1
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

    if not backup.exists():
        os.replace(path, backup)          # 원본 1회 백업
    else:
        os.remove(path)
    os.replace(tmp, path)                  # 정제본으로 교체
    print(f"cleaned {changed}/{n} chunks ({changed/n*100:.1f}%) in {path.name}")
    print(f"backup: {backup.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
