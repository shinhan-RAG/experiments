#!/usr/bin/env python3
"""청크 메타데이터(벡터용) + 시멘틱 태그(grep용) 생성.
입력은 chunks.jsonl과 aliases.json뿐이다. QA 파일 접근 금지 (guard 포함)."""
import json, re, os, hashlib, sys

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")

# --- QA 접근 가드: 이 프로세스에서 QA 경로 open 시도시 즉시 실패 ---
_real_open = open
def _guarded_open(path, *a, **kw):
    if "QA_set/정답셋" in str(path) or "qa100_gold" in str(path):
        raise PermissionError(f"enrichment 생성 중 QA 접근 금지: {path}")
    return _real_open(path, *a, **kw)
import builtins
builtins.open = _guarded_open

ARTICLE = re.compile(r"제\s?(\d+(?:-\d+)?)\s?조(?:\s?\(([^)]{1,40})\)|\s+([^\n(]{1,30}))?")
CLAUSE_HEAD = re.compile(r"^(제\s?\d+\s?[편관절항]|부표\s?[\d-]+|\[부표[^\]]*\]|\[별표[^\]]*\])", re.M)

def table_headers(text):
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("|") and s.count("|") >= 2:
            cells = [c.strip() for c in s.strip("|").split("|")]
            cells = [c for c in cells if c and not set(c) <= set("-: ")]
            if cells:
                return cells[:8]
    return []

def main():
    aliases = json.load(_real_open(os.path.join(BASE, "aliases.json"), encoding="utf-8"))
    aliases.pop("_comment", None)
    alias_hash = hashlib.sha256(json.dumps(aliases, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]

    chunks = [json.loads(l) for l in _real_open(os.path.join(OUT, "chunks.jsonl"), encoding="utf-8")]
    meta_f = _real_open(os.path.join(OUT, "chunk_metadata.jsonl"), "w", encoding="utf-8")
    tag_f = _real_open(os.path.join(OUT, "semantic_tags.jsonl"), "w", encoding="utf-8")

    for c in chunks:
        text = c["text"]
        # 조항 라벨
        arts = []
        for m in ARTICLE.finditer(text):
            label = f"제{m.group(1)}조"
            title = (m.group(2) or m.group(3) or "").strip()
            arts.append(label + (f"({title})" if title else ""))
        arts = list(dict.fromkeys(arts))[:5]
        # 구조 헤드 (부표/별표/편/관)
        struct = list(dict.fromkeys(m.group(1).strip() for m in CLAUSE_HEAD.finditer(text)))[:5]
        # alias 매칭: 청크에 canonical 또는 변형이 등장하면 양방향 표현 모두 부여
        hit_alias = []
        for canon, alts in aliases.items():
            if canon in text or any(a in text for a in alts):
                hit_alias.append(canon)
                hit_alias.extend(alts)
        hit_alias = list(dict.fromkeys(hit_alias))[:20]

        headers = table_headers(text) if c["element_type"] == "table" else []

        # 메타데이터 (벡터 색인용 직렬화 대상)
        meta = {
            "chunk_id": c["chunk_id"],
            "contract_scope": c["contract_scope"],
            "section_path": c["section_path"],
            "articles": arts,
            "structures": struct,
            "aliases": hit_alias,
            "element_type": c["element_type"],
        }
        meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        # 시멘틱 태그 (grep 뷰용 직렬화 대상)
        tag = {
            "chunk_id": c["chunk_id"],
            "element_type": c["element_type"],
            "contract_scope": c["contract_scope"],
            "section_path": c["section_path"],
            "article_label": arts,
            "table_headers": headers,
            "structures": struct,
            "aliases": hit_alias,
        }
        tag_f.write(json.dumps(tag, ensure_ascii=False) + "\n")

    meta_f.close(); tag_f.close()
    manifest = {"alias_hash": alias_hash, "generator": "rule-based structural v1", "n": len(chunks)}
    json.dump(manifest, _real_open(os.path.join(OUT, "enrichment_manifest.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False))

if __name__ == "__main__":
    main()
