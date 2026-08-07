"""
AI-Hub 의료+법률 문서이해 코퍼스(라벨 zip) → dr-dci 청크형 corpus.jsonl.

- 라벨 zip을 스트리밍(raw_decode)해서 도메인별 앞 N개 문서 추출
- 문서 text를 문장 경계 기준 ~CHUNK_CHARS 청크로 분할, 각 청크에 parent_id 부여
- Noah parent-aware 로더 호환: corpus._id=청크, corpus.parent_id=원본문서
- augmentation용 원본 라벨(category/keyword/NE)은 parent_meta.jsonl 사이드카로 저장

출력(data/aihub/<variant>/):
  corpus.jsonl       {_id, parent_id, title, text, domain, category, element_type}
  parent_meta.jsonl  {parent_id, domain, category, keyword[], ne_types{}}  ← 3단계 augmentation용
  corpus_stats.json
"""
import zipfile, json, re, sys
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
OUT_BASE = BASE / "data" / "aihub"
SRC = {
    "의료": "/Users/seyoung/Downloads/01-1.정식개방데이터/Training/02.라벨링데이터/TL_01.의료.zip",
    "법률": "/Users/seyoung/Downloads/01-1.정식개방데이터/Training/02.라벨링데이터/TL_02.법률.zip",
}
CHUNK_CHARS = int(__import__("os").environ.get("CHUNK_CHARS", "512"))
MIN_CHARS = 80
SENT_SPLIT = re.compile(r'(?<=[다요음\.。」』】])\s+|\n{1,}')


def stream_records(zip_path, limit):
    """{"totalcount":N,"data":[{...}]} zip → 앞 limit개 레코드 (raw_decode 스트리밍).
    UTF-8 멀티바이트가 1MB 경계에서 잘려도 손상 없도록 incremental decoder 사용."""
    import codecs
    z = zipfile.ZipFile(zip_path)
    f = z.open(z.namelist()[0])
    dec = json.JSONDecoder()
    bdec = codecs.getincrementaldecoder("utf-8")()
    buf = ""; started = False; count = 0
    while count < limit:
        chunk = f.read(1 << 20)
        buf += bdec.decode(chunk, not chunk)   # final=True on EOF
        if not chunk:
            break
        if not started:
            j = buf.find('[', buf.find('"data"'))
            if j == -1:
                buf = buf[-50:]; continue
            buf = buf[j + 1:]; started = True
        while count < limit:
            buf = buf.lstrip(" \t\r\n,")
            if not buf or buf[0] == ']':
                break
            try:
                obj, idx = dec.raw_decode(buf)
            except json.JSONDecodeError:
                break
            yield obj; count += 1; buf = buf[idx:]


def sentence_spans(raw):
    """원본 raw의 문장 (start,end) 오프셋 목록 (연속, 전체 커버)."""
    spans, prev = [], 0
    for mo in SENT_SPLIT.finditer(raw):
        end = mo.start()
        if end > prev:
            spans.append((prev, end))
        prev = mo.end()
    if prev < len(raw):
        spans.append((prev, len(raw)))
    return spans


def chunk_offsets(raw):
    """문장 경계 기준 ~CHUNK_CHARS 청크의 (start,end) 오프셋 목록.
    청크 text = raw[start:end] (원본 슬라이스라 오프셋이 정확)."""
    sents = sentence_spans(raw)
    chunks = []
    cs = ce = None
    for (s, e) in sents:
        if e - s > CHUNK_CHARS:                      # 초장문 문장 → 강제 슬라이스
            if cs is not None:
                chunks.append((cs, ce)); cs = None
            for i in range(s, e, CHUNK_CHARS):
                chunks.append((i, min(i + CHUNK_CHARS, e)))
            continue
        if cs is None:
            cs, ce = s, e
        elif e - cs > CHUNK_CHARS:                    # 넘치면 지금까지를 청크로
            chunks.append((cs, ce)); cs, ce = s, e
        else:
            ce = e
    if cs is not None:
        chunks.append((cs, ce))
    return [(s, e) for (s, e) in chunks if e - s >= MIN_CHARS]


def main():
    n_per_domain = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    variant = sys.argv[2] if len(sys.argv) > 2 else "smoke"
    out_dir = OUT_BASE / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    parent_meta = []
    stats = {"variant": variant, "docs": 0, "chunks": 0, "by_domain": {}}
    fc = open(out_dir / "corpus.jsonl", "w", encoding="utf-8")
    # 원본 parent 텍스트를 data/aihub/parents.jsonl 로 저장 (모든 단계의 단일 기준)
    fp = open(OUT_BASE / "parents.jsonl", "w", encoding="utf-8")
    for domain, zp in SRC.items():
        d_docs = d_chunks = 0
        for rec in stream_records(zp, n_per_domain):
            pid = rec.get("book_id") or f"{domain}_{d_docs}"
            raw = rec.get("text", "")
            cats = rec.get("category", "")
            offs = chunk_offsets(raw)
            if not offs:
                continue
            for ci, (s, e) in enumerate(offs):
                row = {"_id": f"{pid}_{ci:03d}", "parent_id": pid,
                       "title": f"{domain} · {cats}", "text": raw[s:e],
                       "char_start": s, "char_end": e,          # 원본 내 위치
                       "domain": domain, "category": cats, "split_method": "text"}
                fc.write(json.dumps(row, ensure_ascii=False) + "\n")
                d_chunks += 1
            ne_types = Counter(ne.get("type", "?") for ne in (rec.get("NE") or []))
            parent_meta.append({"parent_id": pid, "domain": domain, "category": cats,
                                "keyword": rec.get("keyword", []),
                                "ne_types": dict(ne_types.most_common(10)),
                                "n_chunks": len(offs)})
            fp.write(json.dumps({"parent_id": pid, "domain": domain,
                                 "category": cats, "text": raw}, ensure_ascii=False) + "\n")
            d_docs += 1
        stats["by_domain"][domain] = {"docs": d_docs, "chunks": d_chunks}
        stats["docs"] += d_docs; stats["chunks"] += d_chunks
        print(f"  [{domain}] 문서 {d_docs:,} → 청크 {d_chunks:,}")
    fc.close(); fp.close()

    with open(out_dir / "parent_meta.jsonl", "w", encoding="utf-8") as f:
        for m in parent_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(out_dir / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n총 문서 {stats['docs']:,} / 청크 {stats['chunks']:,}  → {out_dir}")
    print("도메인별:", stats["by_domain"])


if __name__ == "__main__":
    main()
