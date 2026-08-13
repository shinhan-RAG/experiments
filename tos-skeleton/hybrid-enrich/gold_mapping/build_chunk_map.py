#!/usr/bin/env python3
"""line-span overlap으로 elements_rev4(P-section split) -> chunks_fixed600(fixed600 split)
브릿지 맵 생성. element/chunk 각각 line_start/line_end 필드를 기준으로 겹치는 청크를 모은다."""
import argparse, json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_map(elements, chunks):
    chunks_sorted = sorted(chunks, key=lambda c: (c.get("line_start", 0), c["chunk_id"]))
    chunk_map = {}
    for el in elements:
        es, ee = el.get("line_start"), el.get("line_end")
        hits = []
        if es is not None and ee is not None:
            for c in chunks_sorted:
                cs, ce = c.get("line_start"), c.get("line_end")
                if cs is None or ce is None:
                    continue
                if max(es, cs) <= min(ee, ce):
                    hits.append(c["chunk_id"])
        chunk_map[el["element_id"]] = hits
    return chunk_map


def print_coverage(elements, chunks, chunk_map):
    total = len(elements)
    mapped = sum(1 for v in chunk_map.values() if v)
    unmapped = total - mapped
    pct = (unmapped / total * 100) if total else 0.0
    covered_chunks = {cid for ids in chunk_map.values() for cid in ids}
    print(f"elements: total={total} mapped={mapped} unmapped={unmapped} ({pct:.1f}%) "
          f"chunks_covered={len(covered_chunks)}/{len(chunks)}")


def validate_gold(gold_path, ref_path, chunk_map):
    """재앵커링 gold(element 기준)와 참조 gold(fixed600 chunk 기준)의 겹침률 확인."""
    anchored = load_jsonl(gold_path)
    ref = {r.get("qid"): r for r in load_jsonl(ref_path)}
    rates, matched, skipped = [], 0, 0
    for row in anchored:
        r = ref.get(row.get("qid"))
        if r is None:
            skipped += 1
            continue
        gold_elem_ids = row.get("gold") or row.get("gold_element_ids") or []
        derived = set()
        for eid in gold_elem_ids:
            derived.update(chunk_map.get(eid, []))
        ref_chunks = set()
        for g in r.get("chunkset_mappings", {}).get("fixed600", []):
            ref_chunks.update(g.get("chunk_ids", []))
        if not ref_chunks:
            ref_chunks.update(r.get("gold_chunk_ids", []))
        if not ref_chunks:
            skipped += 1
            continue
        matched += 1
        rates.append(len(derived & ref_chunks) / len(ref_chunks))
    avg = sum(rates) / len(rates) if rates else 0.0
    print(f"validate-gold: matched={matched} skipped={skipped} avg_overlap_rate={avg:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--elements", default=str((BASE / "../semtag-schema/out/v2ds/elements_rev4.jsonl").resolve()))
    ap.add_argument("--chunks", default=str((BASE / "metajson-v6/meta-search-v4/out/chunks_fixed600.jsonl").resolve()))
    ap.add_argument("--output", default=str((BASE / "out/chunk_map_rev4_to_fixed600.json").resolve()))
    ap.add_argument("--validate-gold", nargs=2, metavar=("GOLD", "REF"),
                     help="재앵커링 gold jsonl(v2ds30_noah_v3_train_anchored_full.jsonl 등)과 "
                          "참조 gold jsonl(qa_gold_v3.jsonl) 경로를 받아 overlap rate 검증")
    args = ap.parse_args()

    elements = load_jsonl(Path(args.elements))
    chunks = load_jsonl(Path(args.chunks))
    chunk_map = build_map(elements, chunks)
    print_coverage(elements, chunks, chunk_map)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunk_map, f, ensure_ascii=False, indent=2)
    print(f"저장: {out_path}")

    if args.validate_gold:
        gold_path, ref_path = args.validate_gold
        validate_gold(Path(gold_path), Path(ref_path), chunk_map)


if __name__ == "__main__":
    main()
