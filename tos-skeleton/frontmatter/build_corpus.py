#!/usr/bin/env python3
"""에이전틱 서치용 코퍼스 물화.

corpus_plain/   : 특약폴더/조파일.md (INDEX 없음)
corpus_indexed/ : 동일 + INDEX.md (frontmatter 명함첩)
파일명 매핑은 out/corpus_map.json에 저장.
"""
import json
import re
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from gate2 import ns, BASE
from gate2b import load_all

CORPUS = BASE / "frontmatter" / "corpus"

DISEASE = {'암':'암','뇌혈관':'뇌혈관질환','심장':'허혈심장질환','치매':'치매','당뇨':'당뇨',
 '치아':'치아','장해':'장해','골절':'골절','수술':'수술','입원':'입원','간병':'간병',
 '유방':'여성유방암','전립선':'전립선암','갑상선':'갑상선암','재해':'재해','생활비':'생활비'}


def safe(name, maxlen=80):
    s = re.sub(r"[/\\:*?\"<>|]", "_", name).strip()
    return s[:maxlen]


def main():
    units, chunks, bm, rider_names, rider_bm, ALWAYS, mapped = load_all()
    fm = json.loads((BASE / "frontmatter" / "out" / "260507.frontmatter.json").read_text())
    cov = {c["rider"]: c for c in fm["coverage"]}

    for arm in ("plain", "indexed"):
        d = CORPUS / f"corpus_{arm}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    cmap = {}
    for i, c in enumerate(chunks):
        rel = Path(safe(c["unit"])) / f"{safe(c['no'])}_{safe(c['title'], 40)}.md"
        cmap[str(i)] = {"unit": c["unit"], "no": c["no"], "rel": str(rel)}
        for arm in ("plain", "indexed"):
            p = CORPUS / f"corpus_{arm}" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            body = f"# {c['unit']} | {c['no']} {c['title']}\n\n{c['text']}\n"
            if p.exists():  # 동일 (조,제목)이 케이스 차이로 충돌하면 이어붙임
                body = p.read_text() + "\n" + c["text"] + "\n"
            p.write_text(body, encoding="utf-8")

    # INDEX.md — frontmatter 명함첩 (특약명+태그+급부명, 압축판)
    lines = ["# 특약 색인 (frontmatter)\n",
             "형식: 특약폴더명 [태그] :: 급부명들\n"]
    for c in fm["composition"]:
        tags = sorted({v for k, v in DISEASE.items() if k in c["rider"]})
        bens = ", ".join(b["name"] for b in cov.get(c["rider"], {}).get("benefits", []))
        lines.append(f"- {safe(c['rider'])} [{','.join(tags)}] :: {bens}")
    lines.append("\n주계약·공통 사항은 특약이 아닌 폴더(신한(간편가입)통합건강보험...)에 있음")
    (CORPUS / "corpus_indexed" / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")

    (BASE / "frontmatter" / "out" / "corpus_map.json").write_text(
        json.dumps(cmap, ensure_ascii=False))
    n_files = sum(1 for _ in (CORPUS / "corpus_plain").rglob("*.md"))
    print(f"청크 파일 {n_files}개 × 2안, INDEX.md {len(fm['composition'])}줄")


if __name__ == "__main__":
    main()
