"""
AI-Hub 의료+법률 → element 단위 corpus (512자 청크와 별도 variant).

element 정의:
- 법률: 【판시사항】【판결요지】【참조조문】【이유】 등 섹션 마커로 분할 → 각 섹션 = element
- 의료: 구조 마커 없음 → 문장 그룹(~SENT_GROUP 문장)으로 근사
너무 긴 element(법률 '이유' 등)는 MAX_ELEM_CHARS로 상한 슬라이스, 너무 짧으면 병합.

512 청크와 동일 parent_id 공유 → 두 청킹을 같은 QA(정답 span)로 평가·비교 가능.
출력: data/aihub/element/{corpus.jsonl, parent_meta.jsonl, corpus_stats.json}
"""
import zipfile, json, re, statistics as st
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "aihub" / "element"
SRC = {
    "의료": "/Users/seyoung/Downloads/01-1.정식개방데이터/Training/02.라벨링데이터/TL_01.의료.zip",
    "법률": "/Users/seyoung/Downloads/01-1.정식개방데이터/Training/02.라벨링데이터/TL_02.법률.zip",
}
N_PER_DOMAIN = 5000
SECTION = re.compile(r'(?=【[^】]{1,20}】)')      # 섹션 마커 '앞'에서 분할
SENT = re.compile(r'(?<=[다요음\.。」』】])\s+|\n{1,}')
SENT_GROUP = 3            # 의료: 문장 N개 = 1 element
MAX_ELEM_CHARS = 900      # 초장문 element 상한
MIN_ELEM_CHARS = 80


def stream_records(zip_path, limit):
    import codecs
    z = zipfile.ZipFile(zip_path); f = z.open(z.namelist()[0])
    dec = json.JSONDecoder(); buf = ""; started = False; count = 0
    bdec = codecs.getincrementaldecoder("utf-8")()
    while count < limit:
        b = f.read(1 << 20)
        buf += bdec.decode(b, not b)
        if not b:
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


def cap_off(s, e):
    """긴 element (start,end)를 MAX_ELEM_CHARS 이하 오프셋들로 슬라이스."""
    if e - s <= MAX_ELEM_CHARS:
        return [(s, e)]
    return [(i, min(i + MAX_ELEM_CHARS, e)) for i in range(s, e, MAX_ELEM_CHARS)]


def elements_legal(raw):
    """섹션 마커 '앞'에서 분할한 (start,end) 오프셋. 짧은 조각은 뒤에 병합."""
    starts = [mo.start() for mo in MARK.finditer(raw)]
    bounds = sorted(set([0] + starts + [len(raw)]))
    segs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] > bounds[i]]
    merged, cur = [], None
    for (s, e) in segs:
        if cur is None:
            cur = [s, e]
        elif cur[1] - cur[0] < MIN_ELEM_CHARS:      # 직전 조각이 짧으면 이어붙임
            cur[1] = e
        else:
            merged.append(tuple(cur)); cur = [s, e]
    if cur is not None:
        if merged and cur[1] - cur[0] < MIN_ELEM_CHARS:
            merged[-1] = (merged[-1][0], cur[1])
        else:
            merged.append(tuple(cur))
    out = []
    for (s, e) in merged:
        out += cap_off(s, e)
    return [(s, e) for (s, e) in out if e - s >= MIN_ELEM_CHARS]


def elements_medical(raw):
    """문장 SENT_GROUP개씩 묶은 (start,end) 오프셋."""
    ss = sentence_spans(raw)
    out = []
    for i in range(0, len(ss), SENT_GROUP):
        grp = ss[i:i + SENT_GROUP]
        out += cap_off(grp[0][0], grp[-1][1])
    return [(s, e) for (s, e) in out if e - s >= MIN_ELEM_CHARS]


MARK = re.compile(r'【[^】]{1,20}】')


def sentence_spans(raw):
    spans, prev = [], 0
    for mo in SENT.finditer(raw):
        end = mo.start()
        if end > prev:
            spans.append((prev, end))
        prev = mo.end()
    if prev < len(raw):
        spans.append((prev, len(raw)))
    return spans


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fc = open(OUT / "corpus.jsonl", "w", encoding="utf-8")
    parent_meta = []
    stats = {"variant": "element", "docs": 0, "chunks": 0, "by_domain": {}}
    for domain, zp in SRC.items():
        splitter = elements_legal if domain == "법률" else elements_medical
        d_docs = d_el = 0; elens = []
        for rec in stream_records(zp, N_PER_DOMAIN):
            pid = rec.get("book_id") or f"{domain}_{d_docs}"
            raw = rec.get("text", "")
            offs = splitter(raw)
            if not offs:
                continue
            for ei, (s, e) in enumerate(offs):
                fc.write(json.dumps({
                    "_id": f"{pid}_e{ei:03d}", "parent_id": pid,
                    "title": f"{domain} · {rec.get('category','')}", "text": raw[s:e],
                    "char_start": s, "char_end": e,
                    "domain": domain, "category": rec.get("category", ""),
                    "split_method": "section" if domain == "법률" else "sent_group",
                }, ensure_ascii=False) + "\n")
                elens.append(e - s); d_el += 1
            ne = Counter(x.get("type", "?") for x in (rec.get("NE") or []))
            parent_meta.append({"parent_id": pid, "domain": domain,
                                "category": rec.get("category", ""),
                                "keyword": rec.get("keyword", []),
                                "ne_types": dict(ne.most_common(10)), "n_elements": len(offs)})
            d_docs += 1
        stats["by_domain"][domain] = {"docs": d_docs, "elements": d_el,
                                      "avg_len": int(st.mean(elens)) if elens else 0}
        stats["docs"] += d_docs; stats["chunks"] += d_el
        print(f"  [{domain}] 문서 {d_docs:,} → element {d_el:,} (평균 {int(st.mean(elens)) if elens else 0}자)")
    fc.close()
    with open(OUT / "parent_meta.jsonl", "w", encoding="utf-8") as f:
        for m in parent_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(OUT / "corpus_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n총 문서 {stats['docs']:,} / element {stats['chunks']:,}  → {OUT}")


if __name__ == "__main__":
    main()
