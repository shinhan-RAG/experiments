#!/usr/bin/env python3
"""시멘틱 태그: 엘리먼트 단위 생성 (파일서치용). QA 접근 가드 포함."""
import json, re, os, hashlib
import builtins

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out")

_real_open = open
def _guarded_open(path, *a, **kw):
    if "QA_set/정답셋" in str(path) or "qa100_gold" in str(path):
        raise PermissionError(f"태그 생성 중 QA 접근 금지: {path}")
    return _real_open(path, *a, **kw)
builtins.open = _guarded_open

ARTICLE = re.compile(r"제\s?(\d+(?:-\d+)?)\s?조(?:의\d+)?(?:\s?\(([^)]{1,40})\))?")
STRUCT = re.compile(r"(제\s?\d+\s?[편관절]|부표\s?[\d-]+|\[부표[^\]]*\]|\[별표[^\]]*\])")

def table_meta(text):
    headers, row_labels, title = [], [], ""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for l in lines:
        if l.startswith("|"):
            cells = [c.strip() for c in l.strip("|").split("|")]
            cells = [c for c in cells if c and not set(c) <= set("-: ")]
            if cells and not headers:
                headers = cells[:8]
            elif cells:
                row_labels.append(cells[0])
        elif not title:
            title = l[:60]
    return title, headers, list(dict.fromkeys(row_labels))[:12]

def main():
    aliases = json.load(_real_open(os.path.join(BASE, "configs", "aliases.json"), encoding="utf-8"))
    aliases.pop("_comment", None)
    elems = [json.loads(l) for l in _real_open(os.path.join(OUT, "elements.jsonl"), encoding="utf-8")]
    with _real_open(os.path.join(OUT, "element_tags.jsonl"), "w", encoding="utf-8") as f:
        for e in elems:
            text = e["text"]
            arts = []
            for m in ARTICLE.finditer(text):
                arts.append(f"제{m.group(1)}조" + (f"({m.group(2).strip()})" if m.group(2) else ""))
            arts = list(dict.fromkeys(arts))[:5]
            structs = list(dict.fromkeys(m.group(1).strip() for m in STRUCT.finditer(text)))[:5]
            hit_alias = []
            for canon, alts in aliases.items():
                if canon in text or any(a in text for a in alts):
                    hit_alias.append(canon)
                    hit_alias.extend(alts)
            hit_alias = list(dict.fromkeys(hit_alias))[:20]
            tag = {"element_id": e["element_id"], "element_type": e["element_type"],
                   "contract_scope": e["contract_scope"], "article_label": arts,
                   "structures": structs, "aliases": hit_alias}
            if e["element_type"] == "table":
                t, h, r = table_meta(text)
                tag["table_title"], tag["column_headers"], tag["row_labels"] = t, h, r
            f.write(json.dumps(tag, ensure_ascii=False) + "\n")
    manifest = json.load(_real_open(os.path.join(OUT, "enrichment_manifest.json"), encoding="utf-8"))
    manifest["element_tags"] = "rule-based structural v2 (element-level)"
    manifest["n_elements"] = len(elems)
    json.dump(manifest, _real_open(os.path.join(OUT, "enrichment_manifest.json"), "w"), ensure_ascii=False, indent=2)
    print("element tags:", len(elems))

if __name__ == "__main__":
    main()
